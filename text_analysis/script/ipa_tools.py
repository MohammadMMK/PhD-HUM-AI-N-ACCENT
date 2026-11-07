"""
ipa_tools.py - text -> IPA, phoneme substitution, Levenshtein distance.

No settings live here; everything you choose is in make_conditions.py.

    sentences(text)                 split a text into sentences
    to_ipa(sentence)                Kokoro's own Italian G2P (espeak), the exact IPA Kokoro will read
    normalize(ipa, mapping)         e.g. every ɪ -> i
    word_spans(ipa) / units(word)   words of a sentence / phoneme units of a word
    substitute(ipa, choose)         apply substitutions word by word, returns (new ipa, list of changes)
    consistent(...) / inconsistent(...)
    lev(a, b, cost)                 Levenshtein distance between two unit lists

PHONEME UNIT = one base symbol + the diacritics that follow it (ː ʰ ʲ ̃). Stress marks (ˈ ˌ) are kept
in the string but are not part of any unit. So  tː -> tʲː  is ONE substitution (t -> tʲ), and the
geminate tː is one unit. Italian affricates come out of misaki as single symbols (ʦ ʣ ʧ ʤ), and ss as
'S', so a rule on t or s never touches half an affricate.
"""
import os
import random
import re

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", r"C:\Program Files\eSpeak NG\espeak-ng.exe")

STRESS = set("ˈˌ")
DIACRITICS = {"ː": "long", "ʰ": "aspirated", "ʲ": "palatalized", "\u0303": "nasalized"}
BOUNDARY = set(" ;:,.!?\u2014\u2026\"()\u201c\u201d")
MAX_TOKENS = 510

# every symbol Kokoro-82M knows (anything else is silently dropped from the speech)
KOKORO_VOCAB = set(';:,.!?\u2014\u2026"()\u201c\u201d \u0303ʣʥʦʨ\u1d5d\uab67AIOQSTWY\u1d4a'
                   'abcdefhijklmnopqrstuvwxyzɑɐɒæβɔɕçɖðʤəɚɛɜɟɡɥɨɪʝɯɰŋɳɲɴøɸθœɹɾɻʁɽʂʃʈʧʊʋʌɣɤχʎʒʔˈˌːʰʲ↓→↗↘\u1d7b')


# ------------------------------------------------------------------------------------------------
# text -> IPA
# ------------------------------------------------------------------------------------------------
def sentences(text):
    """One sentence per item, split after . ! ? … (punctuation kept). Line breaks count as spaces."""
    text = re.sub(r"\s+", " ", text.lstrip("\ufeff")).strip()
    parts = re.split(r"([.!?\u2026]+[\"\u201d»]?)", text)
    out = []
    for i in range(0, len(parts), 2):
        s = (parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")).strip()
        if s:
            out.append(s)
    return out


_G2P = None


def to_ipa(sentence):
    """The phoneme string Kokoro itself produces for this sentence (Italian, espeak backend)."""
    global _G2P
    if _G2P is None:
        from kokoro import KPipeline                      # imported here: only needed for G2P
        _G2P = KPipeline(lang_code="i", model=False, repo_id="hexgrad/Kokoro-82M")
    ps = _G2P.g2p(sentence)[0]
    if len(ps) > MAX_TOKENS:
        print(f"WARNING: sentence has {len(ps)} symbols (> {MAX_TOKENS}); Kokoro will cut it: {sentence[:60]}...")
    return ps


def normalize(ipa, mapping):
    """Replace symbols everywhere, e.g. {"ɪ": "i"} so the i-rule also catches unstressed i."""
    for old, new in mapping.items():
        ipa = ipa.replace(old, new)
    return ipa


def not_in_vocab(symbols):
    return sorted({s for s in symbols if any(c not in KOKORO_VOCAB for c in s)})


# ------------------------------------------------------------------------------------------------
# words and phoneme units
# ------------------------------------------------------------------------------------------------
def word_spans(ipa):
    """[(start, end)] of every word (maximal run of non-boundary symbols)."""
    return [m.span() for m in re.finditer(r"[^" + re.escape("".join(BOUNDARY)) + r"]+", ipa)]


def words(ipa):
    return [ipa[a:b] for a, b in word_spans(ipa)]


def unit_spans(word):
    """[(start, end)] of every phoneme unit in a word: base symbol + following diacritics, no stress."""
    spans = []
    for i, c in enumerate(word):
        if c in STRESS:
            continue
        if c in DIACRITICS and spans:
            spans[-1] = (spans[-1][0], i + 1)
        elif c not in DIACRITICS:
            spans.append((i, i + 1))
    return spans


def units(word):
    """'tˈutːo' -> ['t', 'u', 'tː', 'o']"""
    return [word[a:b] for a, b in unit_spans(word)]


def base(unit):
    return unit[0]


# ------------------------------------------------------------------------------------------------
# substitution
# ------------------------------------------------------------------------------------------------
def substitute(ipa, choose):
    """Rewrite one sentence. choose(word_index, base_symbol, unit_index) -> replacement or None (keep).
    The replacement takes the place of the base symbol; the original diacritics stay (tː + tʲ -> tʲː).
    Returns (new_ipa, changes); one change per substituted unit:
    {word, unit, old, new, native_word, word_ipa}."""
    out, changes, last = [], [], 0
    for wi, (a, b) in enumerate(word_spans(ipa)):
        out.append(ipa[last:a])
        w = ipa[a:b]
        new_w, pos, w_changes = [], 0, []
        for ui, (ua, ub) in enumerate(unit_spans(w)):
            new = choose(wi, w[ua], ui)
            if new is None or new == w[ua]:
                continue
            new_w.append(w[pos:ua] + new)
            pos = ua + 1                                   # keep the diacritics after the base
            w_changes.append(dict(word=wi, unit=ui, old=w[ua:ub], new=new + w[ua + 1:ub]))
        new_w.append(w[pos:])
        new_w = "".join(new_w)
        for c in w_changes:
            c.update(native_word=w, word_ipa=new_w)
        changes += w_changes
        out.append(new_w)
        last = b
    out.append(ipa[last:])
    return "".join(out), changes


def consistent(sentences_ipa, rule):
    """Every occurrence of each phoneme in `rule` ({old: new}) is replaced by the same new phoneme."""
    out, changes = [], []
    for si, ipa in enumerate(sentences_ipa):
        new, ch = substitute(ipa, lambda wi, b, ui: rule.get(b))
        out.append(new)
        changes += [dict(sentence=si, **c) for c in ch]
    return out, changes


def _bag_draw(alts, rng):
    """Balanced random draws: alternatives come out of a shuffled bag that is refilled when empty,
    so after n draws every alternative has been used n/len(alts) times (+-1)."""
    bag = []

    def draw():
        if not bag:
            bag.extend(rng.sample(alts, len(alts)))
        return bag.pop()
    return draw


def inconsistent(sentences_ipa, spec, balanced=True, rng=None):
    """Inconsistent substitution over a list of sentences (a section, or all sections of a story).

    spec     : {old: dict(alts=[...], share=s)}
               share = 1 : every occurrence of `old` is changed; each WORD draws one alternative (all
                           occurrences in that word get the same one, so geminates stay coherent)
               share < 1 : decided per OCCURRENCE: exactly round(share * occurrences) occurrences are drawn
                           at random over the whole text, each with the same chance, whatever else is in
                           the word (a word can get several vowels changed, or only one of its two e's);
                           each changed occurrence draws its own alternative
    balanced : True -> the alternatives are used equally often (shuffled bag); False -> independent draws
    Returns (new_sentences, changes, report) with
    report[old] = dict(target, changed, words, occurrences, candidate_words)."""
    rng = rng or random.Random()
    tokens = {old: [] for old in spec}                          # (sentence, word, unit) of every occurrence
    for si, ipa in enumerate(sentences_ipa):
        for wi, w in enumerate(words(ipa)):
            for ui, u in enumerate(units(w)):
                if base(u) in spec:
                    tokens[base(u)].append((si, wi, ui))

    choice, target = {}, {}
    for old, s in spec.items():
        alts = list(s["alts"])
        draw = _bag_draw(alts, rng) if balanced else (lambda: rng.choice(alts))
        toks = tokens[old]
        target[old] = int(round(s["share"] * len(toks)))
        if s["share"] >= 1:
            per_word = {}
            for t in toks:
                if t[:2] not in per_word:
                    per_word[t[:2]] = draw()
                choice[t] = per_word[t[:2]]
        else:
            for t in sorted(rng.sample(toks, target[old])):
                choice[t] = draw()

    out, changes = [], []
    for si, ipa in enumerate(sentences_ipa):
        new, ch = substitute(ipa, lambda wi, b, ui, si=si: choice.get((si, wi, ui)))
        out.append(new)
        changes += [dict(sentence=si, **c) for c in ch]

    report = {}
    for old in spec:
        mine = [c for c in changes if base(c["old"]) == old]
        report[old] = dict(target=target[old], changed=len(mine),
                           words=len({(c["sentence"], c["word"]) for c in mine}),
                           occurrences=len(tokens[old]), candidate_words=len({t[:2] for t in tokens[old]}))
    return out, changes, report


# ------------------------------------------------------------------------------------------------
# Levenshtein
# ------------------------------------------------------------------------------------------------
def lev(a, b, cost=None, indel=1.0):
    """Levenshtein distance between two unit lists. cost(x, y) for a substitution (default 1)."""
    cost = cost or (lambda x, y: 0.0 if x == y else 1.0)
    prev = [j * indel for j in range(len(b) + 1)]
    for i in range(1, len(a) + 1):
        cur = [i * indel] + [0.0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + indel, cur[j - 1] + indel, prev[j - 1] + cost(a[i - 1], b[j - 1]))
        prev = cur
    return prev[-1]


def pair_key(x, y):
    """(old, new) with the diacritics both units share removed: ('tː', 'tʲː') -> ('t', 'tʲ')."""
    common = set(x[1:]) & set(y[1:])
    return x[0] + "".join(c for c in x[1:] if c not in common), y[0] + "".join(c for c in y[1:] if c not in common)


def weighted_cost(weights, default=1.0):
    """Substitution cost from {(old, new): weight}; pairs not listed cost `default`."""
    def cost(x, y):
        if x == y:
            return 0.0
        return float(weights.get(pair_key(x, y), default))
    return cost


if __name__ == "__main__":                                       # quick self-test, no G2P needed
    assert units("tˈutːo") == ["t", "u", "tː", "o"]
    new, ch = substitute("tˈutːo kˈasa.", lambda wi, b, ui: {"t": "tʲ", "k": "x"}.get(b))
    assert new == "tʲˈutʲːo xˈasa.", new
    assert [c["old"] + ">" + c["new"] for c in ch] == ["t>tʲ", "tː>tʲː", "k>x"]
    assert pair_key("tː", "tʲː") == ("t", "tʲ")
    assert lev(units("tˈutːo"), units("tʲˈutʲːo")) == 2
    print("ipa_tools ok")
