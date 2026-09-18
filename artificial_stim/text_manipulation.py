"""
Text → phonemes → manipulated phonemes, for Kokoro (misaki) Italian stimuli.

This module is the *symbolic* half of the pipeline. It turns text into the exact
phoneme strings Kokoro would use, checks them against the model vocab, and
rewrites them with substitution rules. It never loads the acoustic model — only
the espeak-backed G2P, and that lazily, on first use — so importing it is cheap.
Synthesis and the interactive visualiser live in functions.py.

    text_to_ipa_chunks(text)              text  -> list of IPA chunks (1 sentence each)
    check_ipa(ipa)                        every char must be a real Kokoro token
    manipulate(ipa, rules)                apply substitution rules
    substitution_stats(before, after)     what actually changed, by comparison
    print_stats(stats) / describe_rules(rules)

ONE RULE KIND: THE WEIGHTED SUBSTITUTION
Everything is expressed as "this phoneme becomes that one, with probability p":
aspiration is {"t": "tʰ"}, palatalization {"t": "tʲ"}, nasalization {"a": "ã"},
Spanish spirantization {"b": "β", "d": "ð", "ɡ": "ɣ"}.

    {"map": {"t": {"ʈ": 70, "t": 30}}}    70% of the /t/ become ʈ, 30% stay t
    {"map": {"t": "ʈ", "d": "ɖ"}}         deterministic (probability 1)
    {"old": "t", "choices": {"ʈ": 1}}     single-phoneme shorthand
    {"old": "t", "new": "ʈ"}              single deterministic shorthand

  old/new may be a vocab char, a token id (int), or a multi-char string:
  {"map": {"tʃ": "ʨ"}} rewrites a 2-codepoint affricate, {"map": {"t": "pʰ"}}
  substitutes an aspirated p. A precomposed character is decomposed for you,
  so "ã" and "a" + combining tilde are the same rule. Weights are fractions
  (0.3) or percentages (30);
  they are normalised per phoneme. Include the phoneme itself among the choices
  to give it a chance of staying put.

  OPTIONS (per rule)
    scope          "all" (default) | "outside_cluster": fire only on singletons,
                   i.e. where the target is not adjacent to another consonant.
    same_geminate  default True: both halves of tː / kk get ONE draw.
    same_in_word   default False: set True and every occurrence of the phoneme
                   inside a word reuses the first draw made in that word.
    seed           per-rule seed, overriding manipulate(seed=...) for this rule.

  Rules are applied IN ORDER, each to the output of the last, so they chain:
  {"map": {"t": "d"}} followed by {"map": {"d": "ð"}} turns t into ð.

AFFRICATE SAFETY (protect_affricates=True, the default)
  espeak-Italian writes affricates as TWO codepoints: tʃ ts dʒ dz. A
  single-char key never fires on either half, so they stay intact; use a
  multi-char key ({"tʃ": "ʨ"}) to rewrite one on purpose.

See example_text_manipulation.py for a runnable walkthrough.
"""

from __future__ import annotations

import os
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
os.environ.setdefault("PHONEMIZER_ESPEAK_PATH",    r"C:\Program Files\eSpeak NG\espeak-ng.exe")

LANG_CODE  = "i"                      # misaki/espeak Italian
REPO       = "hexgrad/Kokoro-82M"     # only to silence misaki's repo_id warning
MAX_TOKENS = 510                      # Kokoro's per-chunk limit


# =============================================================================
# 1. TEXT -> PHONEMES
# =============================================================================
_G2P = None

def get_g2p():
    """The espeak-backed Italian G2P pipeline (model=False: no acoustic model).
    Built on first call and cached, so importing this module stays cheap."""
    global _G2P
    if _G2P is None:
        from kokoro import KPipeline
        _G2P = KPipeline(lang_code=LANG_CODE, model=False, repo_id=REPO)
    return _G2P


def one_sentence_chunks(text):
    """One sentence per chunk. Never merges; splits only on . ! ? … (keeps the punctuation)."""
    parts = re.split(r'([.!?…]+)', text)               # keep delimiters
    sents = []
    for i in range(0, len(parts), 2):
        s = parts[i].strip()
        if i + 1 < len(parts):
            s += parts[i + 1]                          # re-attach its . ! ? …
        s = s.strip()
        if s:
            sents.append(s)
    return sents


def text_to_ipa_chunks(text):
    """Editable IPA, one sentence per chunk, using the pipeline's own g2p."""
    out = []
    for s in one_sentence_chunks(text):
        ps = get_g2p().g2p(s)[0]                       # exact phonemes the model would use
        if len(ps) > MAX_TOKENS:                       # safety: a single overlong sentence
            print(f"⚠ sentence > {MAX_TOKENS} tokens ({len(ps)}), will truncate: {s[:60]}…")
            ps = ps[:MAX_TOKENS]
        out.append(ps)
    return out


# =============================================================================
# 2. KOKORO VOCAB / VALIDATION
# =============================================================================
VOCAB = {';':1,':':2,',':3,'.':4,'!':5,'?':6,'\u2014':9,'\u2026':10,'"':11,'(':12,')':13,
'\u201c':14,'\u201d':15,' ':16,'\u0303':17,'ʣ':18,'ʥ':19,'ʦ':20,'ʨ':21,'\u1d5d':22,'\uab67':23,
'A':24,'I':25,'O':31,'Q':33,'S':35,'T':36,'W':39,'Y':41,'\u1d4a':42,'a':43,'b':44,'c':45,'d':46,
'e':47,'f':48,'h':50,'i':51,'j':52,'k':53,'l':54,'m':55,'n':56,'o':57,'p':58,'q':59,'r':60,'s':61,
't':62,'u':63,'v':64,'w':65,'x':66,'y':67,'z':68,'ɑ':69,'ɐ':70,'ɒ':71,'æ':72,'β':75,'ɔ':76,'ɕ':77,
'ç':78,'ɖ':80,'ð':81,'ʤ':82,'ə':83,'ɚ':85,'ɛ':86,'ɜ':87,'ɟ':90,'ɡ':92,'ɥ':99,'ɨ':101,'ɪ':102,
'ʝ':103,'ɯ':110,'ɰ':111,'ŋ':112,'ɳ':113,'ɲ':114,'ɴ':115,'ø':116,'ɸ':118,'θ':119,'œ':120,'ɹ':123,
'ɾ':125,'ɻ':126,'ʁ':128,'ɽ':129,'ʂ':130,'ʃ':131,'ʈ':132,'ʧ':133,'ʊ':135,'ʋ':136,'ʌ':138,'ɣ':139,
'ɤ':140,'χ':142,'ʎ':143,'ʒ':147,'ʔ':148,'ˈ':156,'ˌ':157,'ː':158,'ʰ':162,'ʲ':164,'↓':169,'→':171,
'↗':172,'↘':173,'\u1d7b':177}

INV = {tid: ch for ch, tid in VOCAB.items()}           # token id -> char


def _check_scan(s):
    kept, tokens, dropped = [], [], []
    for pos, ch in enumerate(s):
        tid = VOCAB.get(ch)
        if tid is None:
            dropped.append((pos, ch))
        else:
            kept.append(ch); tokens.append(tid)
    return ''.join(kept), tokens, dropped


def check_ipa(ipa):
    """Print what the model will actually see. `ipa` may be one string OR a list
    of chunks. Returns True when every character is a real vocab token."""
    chunks = [ipa] if isinstance(ipa, str) else list(ipa)
    clean = True
    for n, s in enumerate(chunks):
        seen, tokens, dropped = _check_scan(s)
        print(f"[chunk {n}] {s!r}")
        print(f"   model sees : {seen!r}")
        print(f"   tokens     : {tokens}")
        if dropped:
            clean = False
            for pos, ch in dropped:
                print(f"   DROPPED @{pos}: {_char_help(ch)}")
        else:
            print("   ok")
        print()
    print("=> ALL CLEAN" if clean else "=> SOME CHARS DROPPED (see above)")
    return clean


def _char_help(ch):
    """Describe an out-of-vocab character, pointing at its decomposition when
    THAT is in the vocab (the classic 'ã' vs 'a' + combining tilde trap)."""
    d = unicodedata.normalize('NFD', ch)
    fix = f"  -> use {' + '.join(repr(x) for x in d)}" if len(d) > 1 and all(x in VOCAB for x in d) else ""
    return f"U+{ord(ch):04X} {unicodedata.name(ch, '?')!r} {ch!r}{fix}"


# =============================================================================
# 3. PHONEME INVENTORY
# =============================================================================
VOWELS = set('aeiouy') | set('ɑɐɒæɔəɚɛɜɨɪɯøœʊʌɤ') | {'\u1d7b','\u1d4a'} | set('AIOQWY')
CONSONANTS = set('bcdfhjklmnpqrstvwxz') | set('βɕçɖðʤɟɡɥʝɰŋɳɲɴɸθɹɾɻʁɽʂʃʈʧʋɣχʎʒʔ') | set('ʣʥʦʨꭧ')
SKIP     = {'ˈ','ˌ','ː','ʰ','ʲ','\u0303','\u1d5d','↓','→','↗','↘'}
BOUNDARY = {' ',';',':',',','.','!','?','\u2014','\u2026','"','(',')','\u201c','\u201d'}
AFFRICATE_PAIRS = {('t','ʃ'), ('t','s'), ('d','ʒ'), ('d','z')}
LEADING  = {'ˈ', 'ˌ'}                                  # stress marks: precede their phoneme
TRAILING = SKIP - LEADING                              # length, aspiration, tilde …: attach to the left
BASE     = VOWELS | CONSONANTS                         # a phoneme "carrier" character


def phoneme_units(s):
    """Split an IPA chunk into phoneme units covering it exactly: a base
    character plus the diacritics trailing it ('tː', 'ã', 'tʲː'), while stress
    marks, spaces and punctuation stay units of their own. This is the grain at
    which substitutions are reported."""
    units = []
    for ch in s:
        if ch in TRAILING and units and units[-1][0] in BASE:
            units[-1] += ch
        else:
            units.append(ch)
    return units


def _prev_phon(s, i):
    """Nearest phoneme before i, skipping diacritics; None across a word boundary."""
    j = i - 1
    while j >= 0:
        c = s[j]
        if c in BOUNDARY: return None
        if c in SKIP: j -= 1; continue
        return c
    return None


def _next_phon(s, i):
    """Nearest phoneme after i, skipping diacritics; None across a word boundary."""
    j = i + 1
    while j < len(s):
        c = s[j]
        if c in BOUNDARY: return None
        if c in SKIP: j += 1; continue
        return c
    return None


def _in_cluster(s, start, end):
    """True if the span s[start:end] is adjacent to a consonant on either side."""
    return (_prev_phon(s, start) in CONSONANTS) or (_next_phon(s, end - 1) in CONSONANTS)


def _is_affricate_member(s, i):
    """True if the single char at i is the stop OR the fricative half of a
    2-codepoint affricate (tʃ ts dʒ dz)."""
    c = s[i]
    return ((c, _next_phon(s, i)) in AFFRICATE_PAIRS) or ((_prev_phon(s, i), c) in AFFRICATE_PAIRS)


# =============================================================================
# 4. THE RULE
# =============================================================================
_RULE_KEYS    = {"map", "old", "new", "choices", "scope", "same_geminate", "same_in_word", "seed"}
_LEGACY_KINDS = {"substitute", "substitute_prob"}
_SCOPES       = ("all", "outside_cluster")


def _norm(x):
    """Token id -> its vocab char; string -> unchanged (may be multi-char).
    Raises if a character is not something the model can read."""
    if isinstance(x, bool):                            # bool is an int: catch the typo
        raise TypeError(f"expected a phoneme or a token id, got {x!r}")
    if isinstance(x, int):
        if x not in INV:
            raise ValueError(f"token id {x} not in vocab")
        return INV[x]
    if not isinstance(x, str) or not x:
        raise TypeError(f"expected a phoneme string or a token id, got {x!r}")
    if all(ch in VOCAB for ch in x):
        return x
    nfd = unicodedata.normalize('NFD', x)              # 'ã' -> 'a' + combining tilde
    if all(ch in VOCAB for ch in nfd):
        return nfd
    bad = next(ch for ch in nfd if ch not in VOCAB)
    raise ValueError(f"{x!r} contains a character the model cannot read: {_char_help(bad)}")


def _norm_choices(choices):
    """{out: weight} | [(out, weight), …] | a single out  ->  ((out, p), …),
    with probabilities summing to 1. Weights may be fractions or percentages."""
    if isinstance(choices, (str, int)):
        return ((_norm(choices), 1.0),)
    items = list(choices.items()) if isinstance(choices, dict) else list(choices)
    if not items:
        raise ValueError("empty choices")
    outs, ws = [], []
    for out, w in items:
        if w < 0:
            raise ValueError(f"negative weight {w} for {out!r}")
        outs.append(_norm(out)); ws.append(float(w))
    total = sum(ws)
    if total <= 0:
        raise ValueError("weights sum to 0")
    return tuple((o, w / total) for o, w in zip(outs, ws))


@dataclass
class Rule:
    """One weighted substitution — {old phoneme: ((new, p), …)} plus its options.
    Build it from a dict with Rule.make(); the module docstring lists the
    accepted shorthands."""
    map: Dict[str, Tuple[Tuple[str, float], ...]]
    scope: str = "all"
    same_geminate: bool = True
    same_in_word: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        self.map = {_norm(k): _norm_choices(v) for k, v in self.map.items()}
        if not self.map:
            raise ValueError("rule has an empty map")
        if self.scope not in _SCOPES:
            raise ValueError(f"scope must be one of {_SCOPES}, got {self.scope!r}")

    # -- construction ---------------------------------------------------------
    @classmethod
    def make(cls, spec):
        """Rule | dict -> Rule."""
        if isinstance(spec, cls):
            return spec
        spec = dict(spec)
        kind = spec.pop("kind", None)                  # tolerated: older rule dicts carry it
        if kind is not None and kind not in _LEGACY_KINDS:
            raise ValueError(
                f"unknown rule kind {kind!r}. This engine has ONE kind, the weighted "
                f"substitution, because every other one is a special case of it: "
                f"aspirate -> {{'map': {{'t': 'tʰ'}}}}, palatalize -> {{'map': {{'t': 'tʲ'}}}}, "
                f"nasal_final -> {{'map': {{'a': 'ã'}}}}, spanish -> {{'map': {{'b': 'β', 'd': 'ð'}}}}.")
        unknown = set(spec) - _RULE_KEYS
        if unknown:
            raise ValueError(f"unknown rule option(s) {sorted(unknown)}; allowed: {sorted(_RULE_KEYS)}")
        if "map" in spec:
            raw = spec.pop("map")
        elif "old" in spec:
            old = spec.pop("old")
            if "choices" in spec:
                raw = {old: spec.pop("choices")}
            elif "new" in spec:
                raw = {old: spec.pop("new")}
            else:
                raise ValueError('a rule with "old" needs "new" or "choices"')
        else:
            raise ValueError('a rule needs "map", or "old" with "new"/"choices"')
        return cls(map=raw, **spec)

    # -- introspection --------------------------------------------------------
    @property
    def keys(self):
        """The rule's targets, longest first (the scan must try those first)."""
        return sorted(self.map, key=len, reverse=True)

    def describe(self):
        """One human-readable line, e.g. 't→{ʈ 70% | t 30%} [same in word]'."""
        parts = []
        for old, outs in self.map.items():
            if len(outs) == 1 and outs[0][1] == 1.0:
                parts.append(f"{old}→{outs[0][0]}")
            else:
                parts.append(f"{old}→{{" + " | ".join(f"{o} {p * 100:.0f}%" for o, p in outs) + "}")
        if self.same_in_word:
            flags = ["same in word"]
        elif self.same_geminate:
            flags = ["same geminate"]
        else:
            flags = ["independent"]
        if self.scope != "all":
            flags.append(self.scope)
        return ", ".join(parts) + f" [{', '.join(flags)}]"

    # -- application ----------------------------------------------------------
    def apply(self, s, mask=None, rng=None, protect_affricates=True):
        """Rewrite one IPA chunk. Returns (new chunk, alteration mask)."""
        return _Application(self, s, mask, rng, protect_affricates).run()


class _Application:
    """One Rule applied to one chunk: a longest-match scan that carries the
    alteration mask forward and remembers earlier draws, so geminates and
    repeats inside a word can reuse them."""

    def __init__(self, rule, s, mask=None, rng=None, protect_affricates=True):
        self.rule = rule
        self.s = s
        self.in_mask = [False] * len(s) if mask is None else mask
        self.rng = rng or random.Random()
        self.protect = protect_affricates
        self.keys = rule.keys
        self.word_id = self._word_ids(s)
        self._last = {}                                # key -> (end index, piece) of its last firing
        self._in_word = {}                             # (word id, key) -> piece chosen in that word

    @staticmethod
    def _word_ids(s):
        """Word index of every character (a boundary char starts a new word)."""
        ids, w = [], 0
        for ch in s:
            if ch in BOUNDARY:
                w += 1
            ids.append(w)
        return ids

    def run(self):
        out, mask, i, n = [], [], 0, len(self.s)
        while i < n:
            key = next((k for k in self.keys if self.s.startswith(k, i)), None)
            if key is None or not self._fires(key, i):
                out.append(self.s[i]); mask.append(self.in_mask[i]); i += 1
                continue
            end = i + len(key)
            piece = self._pick(key, i)
            out.append(piece)
            # a draw that picked the phoneme itself is not an alteration
            mask.extend(self.in_mask[i:end] if piece == key else [True] * len(piece))
            i = end
        return ''.join(out), mask

    def _fires(self, key, i):
        end = i + len(key)
        if len(key) == 1 and self.protect and _is_affricate_member(self.s, i):
            return False                               # keep tʃ / ts / dʒ / dz intact
        if self.rule.scope == "outside_cluster" and _in_cluster(self.s, i, end):
            return False                               # singletons only
        return True

    def _pick(self, key, i):
        r, wid = self.rule, self.word_id[i]
        if r.same_in_word and (wid, key) in self._in_word:
            piece = self._in_word[(wid, key)]          # this word already decided
        elif r.same_geminate and key in self._last and \
                all(c in SKIP for c in self.s[self._last[key][0]:i]):
            piece = self._last[key][1]                 # previous firing ends right before i
        else:
            outs, ws = zip(*r.map[key])
            piece = self.rng.choices(outs, weights=ws, k=1)[0]
        self._last[key] = (i + len(key), piece)
        self._in_word[(wid, key)] = piece
        return piece


def as_rules(rules):
    """Rule dicts (or Rules) -> list of Rule."""
    return [Rule.make(r) for r in rules]


def describe_rules(rules):
    """One human-readable line for a whole rule list (the HTML header uses it)."""
    return "; ".join(r.describe() for r in as_rules(rules))


# =============================================================================
# 5. MANIPULATE
# =============================================================================
def manipulate(ipa, rules, protect_affricates=True, return_mask=False, seed=None):
    """Apply substitution rules to IPA.

    ipa    : one chunk (str) or a list of chunks, e.g. text_to_ipa_chunks() output.
    rules  : list of rule dicts / Rule objects, applied IN ORDER — every rule
             sees the previous rule's output.
    seed   : int for reproducible draws (None = different every run). A rule's
             own "seed" overrides it for that rule.
    return_mask=True also returns a bool mask per chunk, True on every character
             a rule altered or introduced (pass it to synth_aligned(masks=…) and
             the visualiser highlights those phonemes).

    Returns result, or (result, mask); one value per chunk, or a list of them
    when `ipa` is a list. For what the rules actually did, compare the two:
    substitution_stats(ipa, result, rules).
    """
    single = isinstance(ipa, str)
    chunks = [ipa] if single else list(ipa)
    rules = as_rules(rules)
    shared = random.Random(seed)
    rngs = [random.Random(r.seed) if r.seed is not None else shared for r in rules]

    out_strs, out_masks = [], []
    for s in chunks:
        mask = [False] * len(s)
        for rule, rng in zip(rules, rngs):
            s, mask = rule.apply(s, mask, rng, protect_affricates)
        out_strs.append(s)
        out_masks.append(mask)
    if not return_mask:
        return out_strs[0] if single else out_strs
    return (out_strs[0], out_masks[0]) if single else (out_strs, out_masks)


# =============================================================================
# 6. STATISTICS, BY COMPARING BEFORE WITH AFTER
# =============================================================================
EMPTY = '∅'                                            # stands for an inserted / deleted phoneme


def aligned_pairs(before, after):
    """(old, new) phoneme-unit pairs between two versions of one chunk.

    Unchanged units pair with themselves; a rewritten unit pairs with its
    replacement; EMPTY stands for an insertion or a deletion. A run of changes
    that shifted the unit count is reported as one joined pair, which is what
    makes {"tʃ": "ʨ"} read as tʃ→ʨ instead of two half-changes."""
    a, b = phoneme_units(before), phoneme_units(after)
    pairs = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == 'equal':
            pairs += [(u, u) for u in a[i1:i2]]
        elif tag == 'replace' and (i2 - i1) == (j2 - j1):
            pairs += list(zip(a[i1:i2], b[j1:j2]))
        else:
            pairs.append((''.join(a[i1:i2]) or EMPTY, ''.join(b[j1:j2]) or EMPTY))
    return pairs


def _strip_shared_diacritics(old, new):
    """Drop the diacritics both sides carry, so a geminate tː→tʲː is reported as
    the t→tʲ it is. Never strips either side down to nothing."""
    while len(old) > 1 and len(new) > 1 and old[-1] == new[-1] and old[-1] in TRAILING:
        old, new = old[:-1], new[:-1]
    return old, new


def substitution_stats(before, after, rules=None, label=None):
    """Compare the original IPA with the manipulated IPA and report what
    actually happened. JSON-friendly, ready for
    audiovisualize_interactive(stats=…):

        [{"label": str, "changed": int,
          "groups": [{"old": str, "total": int,
                      "rows": [{"new": str, "n": int, "p": float|None,
                                "changed": bool}]}]}]

    `total` is how often that phoneme occurred, `n` how often it came out as
    `new`, `p` the probability the rules asked for (None when the rules say
    nothing about that pair). A phoneme is listed when it changed somewhere or
    when a rule targets it, so an option that never fired stays visible at 0.

    Because this reads only the two strings, it reports the NET effect: rules
    that chained (t→d, then d→ð) show up as t→ð. Pass `rules` to get the target
    probabilities and a header label.
    """
    b_chunks = [before] if isinstance(before, str) else list(before)
    a_chunks = [after] if isinstance(after, str) else list(after)
    if len(b_chunks) != len(a_chunks):
        raise ValueError(f"before has {len(b_chunks)} chunks, after has {len(a_chunks)}")

    counts = Counter()
    for b, a in zip(b_chunks, a_chunks):
        for old, new in aligned_pairs(b, a):
            counts[_strip_shared_diacritics(old, new)] += 1

    declared = _declared_options(as_rules(rules) if rules else [])
    olds = {old for old, new in counts if old != new} | set(declared)
    occurrences = Counter(old for (old, _), n in counts.items() for _ in range(n))

    groups = []
    for old in sorted(olds, key=lambda o: (-occurrences[o], o)):
        seen = {new: n for (o, new), n in counts.items() if o == old}
        for new in declared.get(old, {}):              # keep declared-but-unfired options visible
            seen.setdefault(new, 0)
        if len(phoneme_units(old)) > 1:
            # a multi-unit target (an affricate, a whole syllable) only aligns as
            # one pair where it fired — count the ones that stayed put directly
            kept = sum(b.count(old) for b in b_chunks) - sum(n for new, n in seen.items() if new != old)
            if kept > 0:
                seen[old] = kept
        rows = [{"new": new, "n": n, "p": declared.get(old, {}).get(new), "changed": new != old}
                for new, n in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))]
        groups.append({"old": old, "total": sum(r["n"] for r in rows), "rows": rows})

    return [{
        "label":   label if label is not None else (describe_rules(rules) if rules else ""),
        "changed": sum(r["n"] for g in groups for r in g["rows"] if r["changed"]),
        "groups":  groups,
    }]


def _declared_options(rules):
    """{old: {new: p}} over a rule list. A phoneme that several rules touch gets
    no probabilities: once rules chain, no single number is the target."""
    out, touched_twice = {}, set()
    for rule in rules:
        for old, outs in rule.map.items():
            if old in out:
                touched_twice.add(old)
            out.setdefault(old, {}).update(dict(outs))
    return {old: ({} if old in touched_twice else opts) for old, opts in out.items()}


def print_stats(stats):
    """Text view of substitution_stats() output."""
    for entry in stats:
        print(f"{entry['label'] or 'substitutions'}   ({entry['changed']} changes)")
        if not entry["groups"]:
            print("    nothing changed")
        for g in entry["groups"]:
            print(f"    {g['old']} ×{g['total']}")
            for row in g["rows"]:
                pct = f"{100 * row['n'] / g['total']:.0f}%" if g["total"] else "–"
                tgt = f" (target {100 * row['p']:.0f}%)" if row["p"] is not None else ""
                mark = " " if row["changed"] else "="
                print(f"      {mark} {g['old']} → {row['new']}: {row['n']}  {pct}{tgt}")


__all__ = [
    # text -> phonemes
    "one_sentence_chunks", "text_to_ipa_chunks", "get_g2p", "LANG_CODE", "MAX_TOKENS",
    # vocab / validation
    "VOCAB", "INV", "check_ipa",
    # phoneme inventory
    "VOWELS", "CONSONANTS", "BASE", "LEADING", "TRAILING", "SKIP", "BOUNDARY",
    "AFFRICATE_PAIRS", "phoneme_units",
    # rules & manipulation
    "Rule", "as_rules", "describe_rules", "manipulate",
    # statistics
    "aligned_pairs", "substitution_stats", "print_stats", "EMPTY",
]
