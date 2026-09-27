import json
import os
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
from kokoro import KPipeline, KModel
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
import soundfile as sf
from IPython.display import display, Audio
import re, numpy as np

REPO = "hexgrad/Kokoro-82M"
LOCAL = Path("kokoro_model"); LOCAL.mkdir(exist_ok=True)
# check if the model files exist, if not download them
if not (LOCAL/"config.json").exists() or not (LOCAL/"kokoro-v1_0.pth").exists():
    print("Downloading model files...") 
# # First run only: pull the two files into a local folder
    hf_hub_download(REPO, "config.json",     local_dir=LOCAL)
    hf_hub_download(REPO, "kokoro-v1_0.pth", local_dir=LOCAL)

device = "cuda" if torch.cuda.is_available() else "cpu"


model = KModel(repo_id=REPO,
               config=str(LOCAL/"config.json"),
               model=str(LOCAL/"kokoro-v1_0.pth")).to(device).eval()

g2p = KPipeline(lang_code="i", model=False)   # espeak-backed Italian G2P, no model
gen = KPipeline(lang_code="i", model=model)   # generation, REUSES the one model


def one_sentence_chunks(text):
    """One sentence per chunk. Never merges; splits only on . ! ? … (keeps the punctuation)."""
    parts = re.split(r'([.!?…]+)', text)              # keep delimiters
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
        ps = g2p.g2p(s)[0]                             # exact phonemes the model would use
        if len(ps) > 510:                             # safety: a single overlong sentence
            print(f"⚠ sentence > 510 tokens ({len(ps)}), will truncate: {s[:60]}…")
            ps = ps[:510]
        out.append(ps)
    return out

def synth_chunks(ipa_chunks, voice, speed=1.0):
    audio = []
    phonems = []
    for ps in ipa_chunks:
        if not ps.strip():
            continue
        r = next(gen.generate_from_tokens(tokens=ps, voice=voice, speed=speed))
        phonems.append(r.phonemes)
        audio.append(r.audio.detach().cpu().numpy())
    return np.concatenate(audio), phonems







import unicodedata

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
    """ipa may be a single string OR a list/tuple of ipa chunks."""
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
                d = unicodedata.normalize('NFD', ch)
                fix = f"  -> use {' + '.join(repr(x) for x in d)}" if len(d) > 1 and all(x in VOCAB for x in d) else ""
                print(f"   DROPPED @{pos}: U+{ord(ch):04X} {unicodedata.name(ch,'?')!r} {ch!r}{fix}")
        else:
            print("   ok")
        print()
    print("=> ALL CLEAN" if clean else "=> SOME CHARS DROPPED (see above)")
    return clean   

"""
IPA manipulation engine for Kokoro (misaki) phoneme chunks.

Operates on text_to_ipa_chunks() output (list of IPA strings, one per chunk).
Every emitted character is a real Kokoro vocab token — verify with check_ipa().

RULE KINDS
  aspirate    : append ʰ (162) to target stops             (default targets: p t k)
  palatalize  : append ʲ (164) to target consonants         (default targets: ALL consonants)
  nasal_final : append ̃  (17) to WORD-FINAL vowels          (default targets: ALL vowels)
  substitute  : replace phonemes via a map (old -> new)      (old/new may be a CHAR or a TOKEN ID)
                `new` may itself be a multi-char string, e.g. "pʰ" or "tʲ", so you
                can substitute one phoneme with the aspirated/palatalized form of
                ANOTHER phoneme directly: {"old":"n","new":"pʰ"}.
                Optional per-rule flags append the marker for you instead of typing
                ʰ/ʲ by hand: aspirate=True appends ʰ, palatalize=True appends ʲ to
                every value in the map. E.g. {"old":"n","new":"p","aspirate":True}
                -> n becomes pʰ. Works with "map" (multiple pairs) too.
  retroflex   : preset map  t→ʈ d→ɖ n→ɳ s→ʂ r→ɽ
  japanese    : preset map  s→ɕ  ʃ→ɕ  tʃ/ʧ→ʨ  dʒ/ʤ→ʥ  p→ɸ
  spanish     : preset map  b→β  v→β  d→ð  ɡ→ɣ        (ɡ = U+0261, the script-g espeak emits)

COMMON OPTIONS (per rule)
  scope = "all"              -> every occurrence
        = "outside_cluster"  -> only singletons (target NOT adjacent to another consonant).
                                (e.g. Spanish: spirantize b d g only intervocalically.)
  targets = [...]            -> aspirate / palatalize / nasal_final only: restrict the
                                target set (chars or token ids).

NASAL_FINAL DETAIL
  "Word-final" = the vowel is followed, after skipping any diacritics (ː ʰ ʲ ̃ …),
  by a BOUNDARY char (space or punctuation) or by the end of the chunk.
  The tilde is inserted immediately after the vowel, so length marks survive:
      aː  ->  ãː        (a + ̃ + ː)
  In a word-final diphthong only the LAST vowel is nasalized:
      ai  ->  aĩ
  targets defaults to every vowel; pass e.g. targets="aeo" or targets=['a','ɛ']
  to restrict it to certain vowels only.

AFFRICATE SAFETY (protect_affricates=True, default)
  espeak-Italian writes affricates as TWO codepoints: tʃ, ts, dʒ, dz.
  A single-char rule (aspirate/palatalize/retroflex t, substitute s ...) will NOT fire
  on either half of such a pair, so tʃ/ts/dʒ/dz stay intact. To transform an affricate
  on purpose, give a MULTI-char key (e.g. "tʃ":"ʨ") — multi-char keys bypass the guard.
"""

import unicodedata

# ---- vocab (verified against hexgrad/Kokoro-82M config.json) -------------------
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
INV = {v: k for k, v in VOCAB.items()}                     # token id -> char

# ---- phoneme classes ----------------------------------------------------------
VOWELS = set('aeiouy') | set('ɑɐɒæɔəɚɛɜɨɪɯøœʊʌɤ') | {'\u1d7b','\u1d4a'} | set('AIOQWY')
CONSONANTS = set('bcdfhjklmnpqrstvwxz') | set('βɕçɖðʤɟɡɥʝɰŋɳɲɴɸθɹɾɻʁɽʂʃʈʧʋɣχʎʒʔ') | set('ʣʥʦʨꭧ') \
             | {'S'}   # misaki writes geminate ss as ONE token 'S' (e.g. complessa -> komplˈeSa)
SKIP     = {'ˈ','ˌ','ː','ʰ','ʲ','\u0303','\u1d5d','↓','→','↗','↘'}
BOUNDARY = {' ',';',':',',','.','!','?','\u2014','\u2026','"','(',')','\u201c','\u201d'}
AFFRICATE_PAIRS = {('t','ʃ'), ('t','s'), ('d','ʒ'), ('d','z')}
VOICELESS_STOPS = set('ptk')
NASAL_TILDE = '\u0303'                                     # combining tilde, token 17
PALATALIZE_TARGETS_NATURAL = "pbmɸβfvʋtdnszlrɾɹðθʦʣkɡŋɣɰqʁχhʔ"       # theoretical, full vocab
PALATALIZE_TARGETS_ITALIAN = "pbtdkɡfvszmnlrɾ"                       # what actually occurs in it_IT output

# ---- preset substitution maps -------------------------------------------------
PRESETS = {
    "retroflex": {'t':'ʈ', 'd':'ɖ', 'n':'ɳ', 's':'ʂ', 'r':'ɽ'},
    "japanese":  {'tʃ':'ʨ', 'dʒ':'ʥ', 'ʧ':'ʨ', 'ʤ':'ʥ', 's':'ɕ', 'ʃ':'ɕ', 'p':'ɸ'},
    "spanish":   {'b':'β', 'v':'β', 'd':'ð', 'ɡ':'ɣ'},     # 'ɡ' is U+0261 (script g)
}

# ---- neighbour / cluster / affricate helpers ----------------------------------
def _prev_phon(s, i):
    j = i - 1
    while j >= 0:
        c = s[j]
        if c in BOUNDARY: return None
        if c in SKIP: j -= 1; continue
        return c
    return None

def _next_phon(s, i):
    j = i + 1
    while j < len(s):
        c = s[j]
        if c in BOUNDARY: return None
        if c in SKIP: j += 1; continue
        return c
    return None

def _in_cluster(s, start, end):
    """True if span s[start:end] is adjacent to a consonant on either side."""
    return (_prev_phon(s, start) in CONSONANTS) or (_next_phon(s, end - 1) in CONSONANTS)

def _is_affricate_member(s, i):
    """True if single char at i is the stop OR fricative half of a 2-codepoint affricate."""
    c = s[i]
    return ((c, _next_phon(s, i)) in AFFRICATE_PAIRS) or ((_prev_phon(s, i), c) in AFFRICATE_PAIRS)

def _is_word_final(s, end):
    """True if the span ending at `end` is the last phoneme of its word:
    nothing follows but diacritics, then a boundary char or the end of the chunk.
    (_next_phon already skips diacritics and returns None at boundary/end.)"""
    return _next_phon(s, end - 1) is None

# ---- normalisation: accept a CHAR or a TOKEN ID -------------------------------
def _norm(x):
    """int -> its vocab char; str -> unchanged (may be a multi-char sequence)."""
    if isinstance(x, int):
        if x not in INV:
            raise ValueError(f"token id {x} not in vocab")
        return INV[x]
    return x

def _norm_map(m):
    return {_norm(k): _norm(v) for k, v in m.items()}

import random
from collections import Counter

# ---- probability helpers (for "substitute_prob") ------------------------------
def _norm_choices(choices, suffix=""):
    """choices: {out: weight} dict OR list of (out, weight) pairs.
    Outputs may be chars, multi-char strings, or token ids. Weights can be
    fractions (0.4) or percentages (40) — they are normalised to sum to 1.
    Returns (outputs, weights)."""
    items = list(choices.items()) if isinstance(choices, dict) else list(choices)
    if not items:
        raise ValueError("substitute_prob: empty choices")
    outs, ws = [], []
    for out, w in items:
        if w < 0:
            raise ValueError(f"substitute_prob: negative weight {w} for {out!r}")
        outs.append(_norm(out) + suffix)
        ws.append(float(w))
    total = sum(ws)
    if total <= 0:
        raise ValueError("substitute_prob: weights sum to 0")
    return outs, [w / total for w in ws]

# ---- core scan: longest-match, guarded ----------------------------------------
def _scan(s, keys, action, scope, protect_affricates, in_mask=None, guard=None, pass_pos=False,
          in_origin=None):
    """Longest-match scan. `in_mask` (optional) is a bool list aligned to `s`,
    carrying forward alteration flags from earlier rules in the same
    manipulate() call. `guard` (optional) is a callable (s, start, end) -> bool;
    the rule only fires where it returns True — used for positional conditions
    such as word-finality. Returns (output_string, out_mask).
    If the action returns the key unchanged (e.g. a probabilistic rule picked
    the phoneme itself), the incoming mask is kept, so it is NOT flagged.
    pass_pos=True: the action is called as action(key, s, i) instead of
    action(key), so it can look at where the match is (used by substitute_prob
    to keep geminates / repeated phonemes in a word consistent).
    `in_origin` (optional) is a list aligned to `s` of (start, end) spans into
    the ORIGINAL chunk; every output char gets the span it came from. All chars
    of a replacement piece share the union span of the chars they replaced
    (k -> kʲ: both k and ʲ point at the original k). Returns (out, mask, origin)."""
    if in_mask is None:
        in_mask = [False] * len(s)
    if in_origin is None:
        in_origin = [(j, j + 1) for j in range(len(s))]
    keys = sorted(set(keys), key=len, reverse=True)
    out, mask, origin, i, n = [], [], [], 0, len(s)
    while i < n:
        key = next((k for k in keys if s.startswith(k, i)), None)
        if key is None:
            out.append(s[i]); mask.append(in_mask[i]); origin.append(in_origin[i]); i += 1; continue
        end = i + len(key)
        fire = True
        if len(key) == 1 and protect_affricates and _is_affricate_member(s, i):
            fire = False                                   # protect single-char hits on tʃ/ts/dʒ/dz
        if fire and scope == "outside_cluster" and _in_cluster(s, i, end):
            fire = False                                   # singletons only
        if fire and guard is not None and not guard(s, i, end):
            fire = False                                   # positional condition failed
        if fire:
            piece = action(key, s, i) if pass_pos else action(key)
            out.append(piece)
            if piece == key:
                mask.extend(in_mask[i:end])                # unchanged -> keep previous flags
                origin.extend(in_origin[i:end])
            else:
                mask.extend([True] * len(piece))
                src = in_origin[i:end]
                span = (min(a for a, _ in src), max(b for _, b in src))
                origin.extend([span] * len(piece))         # whole piece inherits the replaced span
            i = end
        else:
            out.append(s[i]); mask.append(in_mask[i]); origin.append(in_origin[i]); i += 1
    return ''.join(out), mask, origin

# ---- one rule -> one chunk ----------------------------------------------------
def _counted(action, counter):
    """Wrap a _scan action so every firing is tallied as counter[(old, new)] += 1."""
    if counter is None:
        return action
    def wrapped(k, *ctx):
        piece = action(k, *ctx)
        counter[(k, piece)] += 1
        return piece
    return wrapped

def _apply_rule(s, rule, protect_affricates, in_mask=None, rng=None, counter=None, in_origin=None):
    kind  = rule["kind"]
    scope = rule.get("scope", "all")

    if kind in ("aspirate", "palatalize"):
        default = VOICELESS_STOPS if kind == "aspirate" else CONSONANTS
        targets = {_norm(t) for t in rule.get("targets", default)}
        suffix  = 'ʰ' if kind == "aspirate" else 'ʲ'
        return _scan(s, targets, _counted(lambda k: k + suffix, counter),
                     scope, protect_affricates, in_mask, in_origin=in_origin)

    if kind == "nasal_final":
        targets = {_norm(t) for t in rule.get("targets", VOWELS)}
        bad = targets - VOWELS
        if bad:
            raise ValueError(f"nasal_final targets must be vowels; got {sorted(bad)}")
        return _scan(s, targets, _counted(lambda k: k + NASAL_TILDE, counter),
                     scope, protect_affricates, in_mask,
                     guard=lambda st, a, b: _is_word_final(st, b), in_origin=in_origin)

    if kind in PRESETS or kind == "substitute":
        mapping = _rule_mapping(rule)
        return _scan(s, mapping.keys(), _counted(lambda k: mapping[k], counter),
                     scope, protect_affricates, in_mask, in_origin=in_origin)

    if kind == "substitute_prob":
        rng = rng or random.Random()
        table = _rule_prob_table(rule)
        same_geminate, same_in_word = _prob_flags(rule)
        # word id of every position in this chunk (a boundary char starts a new word)
        word_id, w = [], 0
        for ch in s:
            if ch in BOUNDARY:
                w += 1
            word_id.append(w)
        last = {}          # key -> (end position, piece) of the previous firing of that key
        word_pick = {}     # (word id, key) -> piece chosen for that phoneme in that word

        def pick(k, st, i):
            # 1) same_in_word: reuse the choice made for this phoneme earlier in the word
            if same_in_word and (word_id[i], k) in word_pick:
                piece = word_pick[(word_id[i], k)]
            # 2) same_geminate: the previous firing of this key ends right before i
            #    (only diacritics/stress marks in between) -> same choice
            elif same_geminate and k in last and \
                    all(c in SKIP for c in st[last[k][0]:i]):
                piece = last[k][1]
            # 3) otherwise: a fresh weighted random draw
            else:
                outs, ws = table[k]
                piece = rng.choices(outs, weights=ws, k=1)[0]
                if counter is not None:
                    counter[("#draws", k)] += 1
            last[k] = (i + len(k), piece)
            word_pick[(word_id[i], k)] = piece
            return piece

        return _scan(s, table.keys(), _counted(pick, counter),
                     scope, protect_affricates, in_mask, pass_pos=True, in_origin=in_origin)

    raise ValueError(f"unknown rule kind: {kind!r}")

def _rule_mapping(rule):
    """Deterministic {old: new} map for 'substitute' and preset rules."""
    kind = rule["kind"]
    if kind == "substitute":
        mapping = _norm_map(rule["map"] if "map" in rule else {rule["old"]: rule["new"]})
        suffix = ('ʰ' if rule.get("aspirate") else '') + ('ʲ' if rule.get("palatalize") else '')
        return {k: v + suffix for k, v in mapping.items()} if suffix else mapping
    return _norm_map(PRESETS[kind])

def _rule_prob_table(rule):
    """{old: (outputs, weights)} for 'substitute_prob' rules."""
    raw = rule["map"] if "map" in rule else {rule["old"]: rule["choices"]}
    suffix = ('ʰ' if rule.get("aspirate") else '') + ('ʲ' if rule.get("palatalize") else '')
    return {_norm(k): _norm_choices(ch, suffix) for k, ch in raw.items()}

def _prob_flags(rule):
    """(same_geminate, same_in_word) for a 'substitute_prob' rule.
    same_geminate (default True) : tt, kk ... always get ONE choice for both halves.
    same_in_word  (default False): every occurrence of the phoneme in a word reuses
                                   the first choice made in that word."""
    return bool(rule.get("same_geminate", True)), bool(rule.get("same_in_word", False))

# ---- substitution statistics (for the HTML legend) ----------------------------
def _build_stats(rules, counters):
    """Turn per-rule Counters {(old, new): n} into a JSON-friendly list:
    [{"label", "kind", "changed", "groups": [{"old", "total",
      "rows": [{"new", "n", "p" (target prob or None), "changed"}]}]}]
    Deterministic and probabilistic substitutions list every option, even at 0,
    so you can see when a mapping never fired."""
    out = []
    for rule, cnt in zip(rules, counters):
        kind = rule["kind"]
        groups = {}                                        # old -> list of rows
        draws  = {}                                        # old -> independent random draws
        if kind == "substitute_prob":
            for old, (outs, ws) in _rule_prob_table(rule).items():
                groups[old] = [{"new": o, "n": cnt.get((old, o), 0), "p": w, "changed": o != old}
                               for o, w in zip(outs, ws)]
                draws[old] = cnt.get(("#draws", old), 0)
        elif kind in PRESETS or kind == "substitute":
            for old, new in _rule_mapping(rule).items():
                groups[old] = [{"new": new, "n": cnt.get((old, new), 0), "p": None, "changed": new != old}]
        else:                                              # aspirate / palatalize / nasal_final: only hits
            for (old, new), n in sorted(cnt.items(), key=lambda kv: -kv[1]):
                if old == "#draws":
                    continue
                groups.setdefault(old, []).append({"new": new, "n": n, "p": None, "changed": new != old})
        gl = [{"old": old, "total": sum(r["n"] for r in rows), "rows": rows,
               "draws": draws.get(old)} for old, rows in groups.items()]
        out.append({
            "label":   describe_rules([rule]),
            "kind":    kind,
            "changed": sum(r["n"] for g in gl for r in g["rows"] if r["changed"]),
            "groups":  gl,
        })
    return out

def print_stats(stats):
    """Quick text view of manipulate(..., return_stats=True) output."""
    for i, r in enumerate(stats, 1):
        print(f"rule {i}: {r['label']}   ({r['changed']} changes)")
        for g in r["groups"]:
            if g.get("draws") is not None:
                print(f"    {g['old']}: {g['total']} occurrences, {g['draws']} random draws")
            for row in g["rows"]:
                pct = f"{100*row['n']/g['total']:.0f}%" if g["total"] else "–"
                tgt = f" (target {100*row['p']:.0f}%)" if row["p"] is not None else ""
                print(f"    {g['old']} → {row['new']}: {row['n']}  {pct}{tgt}")

# ---- public entry point -------------------------------------------------------
def manipulate(ipa, rules, protect_affricates=True, return_mask=False, seed=None,
               return_stats=False, return_origin=False):
    """ipa: str or list of chunks. rules: list of rule dicts, applied IN ORDER.
    return_mask=True : also return a parallel bool mask (or list of masks)
                       marking every character that any rule altered or introduced.
    return_stats=True: also return per-rule substitution counts (pass them to
                       audiovisualize_interactive(stats=...) for the legend).
    return_origin=True: also return, per chunk, a list aligned to the output
                       string of (start, end) spans into the ORIGINAL chunk.
                       Pass it to synth_aligned(origins=...) to transfer the
                       durations of the original text onto the manipulated one.
    Return order: result, [mask], [stats], [origin].
    seed: int for reproducible "substitute_prob" draws (None = different every run).
    A rule can also carry its own "seed" key, which overrides this for that rule.
    substitute_prob options: same_geminate=True (default) -> both halves of tt/kk
    get one choice; same_in_word=False (default) -> set True so every occurrence
    of the phoneme within a word reuses the first choice made in that word.
    Note: counts are what each rule did when it ran; a later rule can still
    transform the output of an earlier one (e.g. t→d then d→ð)."""
    single = isinstance(ipa, str)
    chunks = [ipa] if single else list(ipa)
    base_rng = random.Random(seed)
    rule_rngs = [random.Random(r["seed"]) if "seed" in r else base_rng for r in rules]
    counters = [Counter() for _ in rules]
    out_strs, out_masks, out_origins = [], [], []
    for s in chunks:
        mask = [False] * len(s)
        origin = [(j, j + 1) for j in range(len(s))]
        for rule, rng, cnt in zip(rules, rule_rngs, counters):
            s, mask, origin = _apply_rule(s, rule, protect_affricates, mask, rng, cnt, origin)
        out_strs.append(s)
        out_masks.append(mask)
        out_origins.append(origin)
    result = [out_strs[0] if single else out_strs]
    if return_mask:
        result.append(out_masks[0] if single else out_masks)
    if return_stats:
        result.append(_build_stats(rules, counters))
    if return_origin:
        result.append(out_origins[0] if single else out_origins)
    return result[0] if len(result) == 1 else tuple(result)

# ---- human-readable rule description (for the HTML header) --------------------
def describe_rules(rules):
    lines = []
    for r in rules:
        kind  = r["kind"]
        scope = r.get("scope", "all")
        sfx   = "" if scope == "all" else f" [{scope}]"
        if kind in PRESETS:
            lines.append(f"{kind}: " + ", ".join(f"{k}→{v}" for k, v in PRESETS[kind].items()) + sfx)
        elif kind == "substitute":
            mapping = r.get("map", {r.get("old"): r.get("new")})
            suffix = ('ʰ' if r.get("aspirate") else '') + ('ʲ' if r.get("palatalize") else '')
            mapping = {k: v + suffix for k, v in mapping.items()} if suffix else mapping
            lines.append("substitute: " + ", ".join(f"{k}→{v}" for k, v in mapping.items()) + sfx)
        elif kind == "substitute_prob":
            raw = r.get("map", {r.get("old"): r.get("choices")})
            suffix = ('ʰ' if r.get("aspirate") else '') + ('ʲ' if r.get("palatalize") else '')
            parts = []
            for k, ch in raw.items():
                outs, ws = _norm_choices(ch, suffix)
                opts = " | ".join(f"{o} {w*100:.0f}%" for o, w in zip(outs, ws))
                parts.append(f"{_norm(k)}→{{{opts}}}")
            gem, word = _prob_flags(r)
            flags = [f for f, on in (("same in word", word), ("same geminate", gem and not word)) if on]
            if not gem and not word:
                flags.append("independent")
            lines.append("substitute_prob: " + ", ".join(parts) + f" [{', '.join(flags)}]" + sfx)
        elif kind == "nasal_final":
            t = r.get("targets")
            which = "all vowels" if t is None else "".join(sorted(_norm(x) for x in t))
            lines.append(f"nasal_final: word-final {which} → +\u0303{sfx}")
        else:                                              # aspirate / palatalize
            t = r.get("targets")
            which = "default set" if t is None else "".join(sorted(_norm(x) for x in t))
            lines.append(f"{kind} on {which}{sfx}")
    return "; ".join(lines)



# =============================================================================
# UPDATED VISUALIZATION SECTION — replaces align_phonemes, synth_aligned,
# audiovisualize_interactive, and _TEMPLATE from your script.
# Everything above this in your file (model load, VOCAB, manipulate(), etc.)
# stays exactly as-is.
#
# WHAT CHANGED (signatures, so you can update call sites):
#   align_phonemes(...)         -> now returns (segments, seg_word_idx)   [was: segments]
#   synth_aligned(...)          -> now returns (audio, segments, seg_words) [was: (audio, segments)]
#   audiovisualize_interactive  -> gained two new optional kwargs: seg_words=None, text=None
#
# Timing math (spf, f2s, lead_trim_frames, cumulative frame accumulation) is
# byte-for-byte the same as before — the only addition is a word counter that
# increments on a literal ' ' character while walking `kept`, which is exactly
# the same loop that already existed.
# =============================================================================

import numpy as np
import base64, io, json, soundfile as sf
from IPython.display import HTML
from html import escape as _html_escape

SR = 24000
FRAME_SAMPLES = 600
BASE     = VOWELS | CONSONANTS
TRAILING = {'ː','ʰ','ʲ','\u0303','\u1d5d','↓','→','↗','↘'}
LEADING  = {'ˈ','ˌ'}


# how misaki's single-letter tokens are shown in the player (the IPA string is unchanged)
LABEL_DISPLAY = {'S': 'sː', 'A': 'eɪ', 'I': 'aɪ', 'W': 'aʊ', 'O': 'oʊ', 'Q': 'əʊ', 'Y': 'ɔɪ', ' ': '␣'}


def token_kind(c):
    """Visual category of one Kokoro token."""
    if c == ' ':        return 'space'
    if c in BASE:       return 'phone'
    if c in TRAILING:   return 'diacritic'
    if c in LEADING:    return 'stress'
    return 'punct'


def align_phonemes(phonemes, pred_dur, n_samples=None, sr=SR, lead_trim_frames=0, altered_mask=None):
    """One segment per Kokoro token — nothing merged, nothing hidden.

    Every token the model received gets its own segment with its own predicted
    duration: phonemes, diacritics (ʲ ʰ ː ...), stress marks (ˈ ˌ), spaces,
    punctuation, and the BOS/EOS tokens Kokoro adds around each chunk.
    So dʲ is two segments, d and ʲ, each with its own duration.

    altered_mask (optional): bool list the same length as `phonemes`, from
    manipulate(..., return_mask=True).
    Returns (segments, seg_word_idx, seg_altered):
      segments     : [(token, start_s, end_s, info)], info = {"kind", "fr", "ms"}
                     kind in phone / diacritic / stress / punct / space / bos / eos
      seg_word_idx : word index per token; None for space / bos / eos
                     (punctuation belongs to the word it is attached to)
      seg_altered  : True where a manipulation rule changed/introduced the token"""
    kept, kept_altered = [], []
    for idx, c in enumerate(phonemes):
        if c in VOCAB:
            kept.append(c)
            kept_altered.append(bool(altered_mask[idx]) if altered_mask is not None else False)
    dur = [int(x) for x in pred_dur]
    assert len(dur) == len(kept) + 2, \
        f"pred_dur ({len(dur)}) != kept phonemes+2 ({len(kept)+2}) — filtered string mismatch"
    spf = (n_samples / sum(dur)) if n_samples else FRAME_SAMPLES
    f2s = lambda f: max(0.0, f - lead_trim_frames) * spf / sr
    f2ms = lambda f: round(f * spf / sr * 1000, 1)
    start_f, acc = [], 0
    for d in dur:
        start_f.append(acc); acc += d

    segs, seg_word_idx, seg_altered = [], [], []
    def add(tok, k, kind, w, alt):
        segs.append((tok, f2s(start_f[k]), f2s(start_f[k] + dur[k]),
                     {"kind": kind, "fr": dur[k], "ms": f2ms(dur[k])}))
        seg_word_idx.append(w); seg_altered.append(alt)

    add('BOS', 0, 'bos', None, False)
    word_idx = 0
    for k, c in enumerate(kept, start=1):
        kind = token_kind(c)
        if kind == 'space':
            add(c, k, kind, None, kept_altered[k - 1])
            word_idx += 1
        else:
            add(c, k, kind, word_idx, kept_altered[k - 1])
    add('EOS', len(dur) - 1, 'eos', None, False)
    return segs, seg_word_idx, seg_altered


def transfer_durations(orig_ps, manip_ps, origin, ref_dur, diacritic_weight=0.3):
    """Map per-token durations of the ORIGINAL chunk onto the MANIPULATED chunk.

    orig_ps  : original IPA string (what the reference voice spoke)
    manip_ps : manipulated IPA string
    origin   : from manipulate(..., return_origin=True) for this chunk
               (None = identity, i.e. manip_ps == orig_ps)
    ref_dur  : reference pred_dur (frames), len = kept(orig_ps) + 2 (BOS/EOS)

    Each group of manipulated tokens that replaced a group of original tokens
    gets EXACTLY the frames of what it replaced (k=5 -> kʲ = 4+1, total 5), so
    every word boundary stays locked to the reference timing.
    Inside a group, frames are split by weight: base phonemes 1.0, diacritics
    (ʲ ʰ ː ̃ ˈ ...) `diacritic_weight`. Every token gets >= 1 frame when the
    group has enough frames; otherwise diacritics may get 0 so the total holds.
    Returns a list of ints, len = kept(manip_ps) + 2, sum == sum(ref_dur)."""
    ref_dur = [int(x) for x in ref_dur]
    if origin is None:
        assert manip_ps == orig_ps, "origin=None requires identical strings"
        origin = [(j, j + 1) for j in range(len(orig_ps))]
    assert len(origin) == len(manip_ps), (len(origin), len(manip_ps))

    # frames of every ORIGINAL char (0 for chars the model drops)
    src_dur, k = [0] * len(orig_ps), 1
    for j, c in enumerate(orig_ps):
        if c in VOCAB:
            src_dur[j] = ref_dur[k]; k += 1
    assert k == len(ref_dur) - 1, \
        f"ref pred_dur ({len(ref_dur)}) != kept original tokens+2 ({k + 1})"

    # group consecutive manipulated chars whose source spans overlap
    groups = []                                            # [a, b, [out idx]]
    for oi, (a, b) in enumerate(origin):
        if groups and a < groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], b); groups[-1][2].append(oi)
        else:
            assert not groups or a >= groups[-1][1], "non-monotonic origin"
            groups.append([a, b, [oi]])
    # deleted source chars (substitution to '') -> give their frames to the previous group
    if groups:
        groups[0][0] = 0
        for g, nxt in zip(groups, groups[1:]):
            g[1] = nxt[0]
        groups[-1][1] = len(orig_ps)

    def weight(c):
        return diacritic_weight if (c in TRAILING or c in LEADING) else 1.0

    new, carry, last_slot = [], 0, None
    for a, b, outs in groups:
        total = sum(src_dur[a:b]) + carry
        kept = [oi for oi in outs if manip_ps[oi] in VOCAB]
        if not kept:                                       # nothing audible here -> pass frames on
            carry = total; continue
        carry = 0
        w = [weight(manip_ps[oi]) for oi in kept]
        ideal = [total * x / sum(w) for x in w]
        alloc = [int(v) for v in ideal]
        rest = total - sum(alloc)                          # largest remainder
        for j in sorted(range(len(kept)), key=lambda j: ideal[j] - alloc[j], reverse=True)[:rest]:
            alloc[j] += 1
        if total >= len(kept):                             # Kokoro never predicts 0: avoid zeros if possible
            for j in range(len(alloc)):
                if alloc[j] == 0:
                    donor = max(range(len(alloc)), key=lambda q: alloc[q])
                    alloc[donor] -= 1; alloc[j] += 1
        last_slot = len(new) + len(alloc) - 1
        new += alloc
    if carry and last_slot is not None:
        new[last_slot] += carry
    out = [ref_dur[0]] + new + [ref_dur[-1]]
    assert sum(out) == sum(ref_dur), (sum(out), sum(ref_dur))
    return out


# ---- cached, decoder-free duration prediction + forced synthesis -------------
# Kokoro splits into: text encoders (depend only on the phoneme string),
# duration predictor (string + voice), decoder (string + voice + durations,
# by far the most expensive). Each stage is cached on exactly what it depends
# on, so nothing is computed twice across versions/voices in one session:
#   _ENC_CACHE   [ps]                    -> BERT + text-encoder outputs
#   _DUR_CACHE   [(voice, ps, speed)]    -> predicted durations (no decoder run)
#   _AUDIO_CACHE [(voice, ps, durations)]-> synthesized chunk audio
# A chunk that is identical in two versions (same string, same timing) is
# therefore synthesized once and is bit-identical in both files.
import hashlib
_ENC_CACHE, _DUR_CACHE, _AUDIO_CACHE = {}, {}, {}


def clear_synth_cache():
    """Free all cached encoder outputs, durations and chunk audio."""
    _ENC_CACHE.clear(); _DUR_CACHE.clear(); _AUDIO_CACHE.clear()


def _voice_pack(voice):
    """-> (voice pack on the model device, stable cache key).
    `voice` can be a tensor (e.g. torch.load('custom_voices/x.pt')) or a name."""
    pack = voice if torch.is_tensor(voice) else gen.load_voice(voice)
    if torch.is_tensor(voice):
        key = "sha1:" + hashlib.sha1(voice.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
    else:
        key = "name:" + str(voice)
    return pack.to(model.device), key


@torch.no_grad()
def _encode(ps):
    """Voice-independent part of KModel.forward_with_tokens, cached per string."""
    enc = _ENC_CACHE.get(ps)
    if enc is not None:
        return enc
    dev = model.device
    ids = [i for i in (model.vocab.get(p) for p in ps) if i is not None]
    assert len(ids) + 2 <= getattr(model, "context_length", 512), f"chunk too long ({len(ids)} tokens)"
    input_ids = torch.LongTensor([[0, *ids, 0]]).to(dev)
    L = torch.full((1,), input_ids.shape[-1], device=dev, dtype=torch.long)
    tm = torch.gt(torch.arange(input_ids.shape[-1], device=dev).unsqueeze(0) + 1, L.unsqueeze(1))
    bert_dur = model.bert(input_ids, attention_mask=(~tm).int())
    enc = {"n": input_ids.shape[-1], "L": L, "tm": tm,
           "d_en": model.bert_encoder(bert_dur).transpose(-1, -2),
           "t_en": model.text_encoder(input_ids, L, tm)}
    _ENC_CACHE[ps] = enc
    return enc


@torch.no_grad()
def predict_durations(ps, pack, speed=1.0):
    """Kokoro's own per-token durations (frames, incl. BOS/EOS) for string `ps`
    spoken by voice `pack`, WITHOUT running the decoder. Identical to the
    pred_dur that normal generation returns."""
    enc = _encode(ps)
    s = pack[len(ps) - 1][:, 128:]                       # same style row as KPipeline.infer
    d = model.predictor.text_encoder(enc["d_en"], s, enc["L"], enc["tm"])
    x, _ = model.predictor.lstm(d)
    duration = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1) / speed
    return torch.round(duration).clamp(min=1).long().reshape(-1).cpu().tolist()


def _durations_cached(ps, pack, vkey, speed):
    key = (vkey, ps, float(speed))
    if key not in _DUR_CACHE:
        _DUR_CACHE[key] = predict_durations(ps, pack, speed)
    return _DUR_CACHE[key]


@torch.no_grad()
def synth_forced(ps, pack, dur):
    """Synthesize `ps` with voice `pack` using the GIVEN durations `dur`
    (len = kept tokens + 2). Pitch/energy and timbre come from the voice."""
    dev = model.device
    enc = _encode(ps)
    ref_s = pack[len(ps) - 1]
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(enc["d_en"], s, enc["L"], enc["tm"])
    dur = torch.as_tensor(dur, device=dev).long().clamp(min=0).reshape(-1)
    n = enc["n"]
    assert dur.shape[0] == n, f"durations ({dur.shape[0]}) != tokens incl. BOS/EOS ({n})"
    idx = torch.repeat_interleave(torch.arange(n, device=dev), dur)
    aln = torch.zeros((n, idx.shape[0]), device=dev)
    aln[idx, torch.arange(idx.shape[0], device=dev)] = 1
    aln = aln.unsqueeze(0)
    F0, N = model.predictor.F0Ntrain(d.transpose(-1, -2) @ aln, s)
    audio = model.decoder(enc["t_en"] @ aln, F0, N, ref_s[:, :128]).squeeze()
    return audio.cpu().numpy()


def synth_aligned(ipa_chunks, voice, speed=1.0, sr=SR, masks=None,
                  dur_voice=None, dur_source=None, original_chunks=None, origins=None,
                  diacritic_weight=0.3, use_cache=True):
    """Synthesize `ipa_chunks` with `voice`; returns (audio, segments, seg_words, seg_altered).

    masks : list parallel to ipa_chunks, from manipulate(..., return_mask=True).

    TIMING
      dur_voice  : voice whose predicted durations drive the timing
                   (None = `voice` times itself, i.e. normal Kokoro generation).
      dur_source : "original"    -> dur_voice speaks `original_chunks` (the
                                    UNMANIPULATED text); durations are remapped
                                    onto ipa_chunks via `origins`, so every
                                    version lands on identical word boundaries.
                   "manipulated" -> dur_voice speaks ipa_chunks ITSELF (errors
                                    included); its durations are reused 1:1.
                   None (default)-> "original" if original_chunks is given,
                                    else "manipulated".
      original_chunks : unmanipulated IPA chunks, parallel to ipa_chunks.
      origins    : from manipulate(..., return_origin=True). None = identity.
      diacritic_weight : share of a replaced token's frames given to added
                   diacritics (k -> kʲ). Only used for dur_source="original".
      speed      : applies to the timing voice's prediction.
      use_cache  : reuse encoder outputs / durations / chunk audio computed
                   earlier in this session (see clear_synth_cache())."""
    if dur_source is None:
        dur_source = "original" if original_chunks is not None else "manipulated"
    assert dur_source in ("original", "manipulated"), dur_source
    if dur_source == "original":
        assert original_chunks is not None and len(original_chunks) == len(ipa_chunks), \
            "dur_source='original' needs original_chunks, parallel to ipa_chunks"
        assert origins is None or len(origins) == len(ipa_chunks)
    if not use_cache:
        clear_synth_cache()

    pack, vkey = _voice_pack(voice)
    dpack, dkey = _voice_pack(dur_voice) if dur_voice is not None else (pack, vkey)

    audios, segments, seg_words, seg_altered = [], [], [], []
    t0, word_offset = 0.0, 0
    for ci, ps in enumerate(ipa_chunks):
        if not ps.strip():
            continue
        if dur_source == "manipulated":
            dur = _durations_cached(ps, dpack, dkey, speed)
        else:
            orig = original_chunks[ci]
            dur = transfer_durations(orig, ps, origins[ci] if origins is not None else None,
                                     _durations_cached(orig, dpack, dkey, speed), diacritic_weight)
        akey = (vkey, ps, tuple(dur))
        a = _AUDIO_CACHE.get(akey)
        if a is None:
            a = synth_forced(ps, pack, dur)
            _AUDIO_CACHE[akey] = a
        chunk_mask = masks[ci] if masks is not None else None
        segs, widx, altd = align_phonemes(ps, dur, n_samples=len(a), sr=sr, altered_mask=chunk_mask)
        segments += [(lab, s + t0, e + t0, info) for lab, s, e, info in segs]
        seg_words += [None if w is None else w + word_offset for w in widx]
        seg_altered += altd
        real = [w for w in widx if w is not None]
        word_offset += (max(real) + 1) if real else 0
        audios.append(a); t0 += len(a) / sr
    return np.concatenate(audios), segments, seg_words, seg_altered


def audiovisualize_interactive(audio, segments, sr=24000, out_html=None, per_row=None,
                                seg_words=None, text=None, seg_altered=None,
                                title=None, rules=None, stats=None):
    """title (optional): short heading shown at the top of the player.
    stats (optional): from manipulate(..., return_stats=True). Renders a legend
    with the number of changes for each substitution (and actual vs target %
    for substitute_prob rules).
    rules (optional): the same `rules` list you passed to manipulate() — it's
    turned into a one-line human-readable summary and shown under the title.
    Falls back gracefully (rule kinds only) if describe_rules() isn't in
    scope yet, so this never raises just because the engine module wasn't
    imported first."""
    audio = np.asarray(audio, dtype=np.float32)
    buf = io.BytesIO(); sf.write(buf, audio, sr, format='WAV')
    b64 = base64.b64encode(buf.getvalue()).decode()
    N = 2000
    step = max(1, len(audio)//N)
    env = np.abs(audio[:step*(len(audio)//step)].reshape(-1, step)).max(axis=1)
    env = (env/(env.max() or 1)).round(3).tolist()
    segs = []
    for seg in segments:                                   # (tok, s, e) or (tok, s, e, info)
        tok, s, e = seg[:3]
        info = seg[3] if len(seg) > 3 and isinstance(seg[3], dict) else {}
        g = {"l": ''.join(LABEL_DISPLAY.get(c, c) for c in tok) if tok not in ("BOS", "EOS") else tok,
             "t": tok, "k": info.get("kind", "phone"),
             "s": round(s, 4), "e": round(e, 4),
             "ms": info.get("ms", round((e - s) * 1000, 1)), "fr": info.get("fr")}
        segs.append(g)
    dur = len(audio)/sr
    if seg_words is not None and text is not None:
        real_words = [w for w in seg_words if w is not None]
        n_words = (max(real_words) + 1) if real_words else 0
        word_labels = text.split()
        if len(word_labels) != n_words:
            print(f"⚠ word count mismatch: text has {len(word_labels)} words, "
                  f"phoneme string implies {n_words} — labels may be misaligned")
        for i, seg in enumerate(segs):
            wi = seg_words[i]
            seg["w"] = wi
            seg["wl"] = (word_labels[wi] if wi < len(word_labels) else "") if wi is not None else None
    if seg_altered is not None:
        if len(seg_altered) != len(segs):
            print(f"⚠ seg_altered length ({len(seg_altered)}) != segments ({len(segs)}) — skipping alter-highlighting")
        else:
            for i, seg in enumerate(segs):
                seg["alt"] = bool(seg_altered[i])

    rules_text = ""
    if rules:
        try:
            rules_text = describe_rules(rules)
        except NameError:
            rules_text = "; ".join(r.get("kind", "?") for r in rules)

    html = _TEMPLATE.replace("__B64__", b64).replace("__SEGS__", json.dumps(segs)) \
                    .replace("__ENV__", json.dumps(env)).replace("__DUR__", str(dur)) \
                    .replace("__TITLE__", json.dumps(title or "")) \
                    .replace("__RULES__", json.dumps(rules_text)) \
                    .replace("__STATS__", json.dumps(stats or []))
    if out_html:
        out_dir = os.path.join(os.getcwd(), "output_online")
        os.makedirs(out_dir, exist_ok=True)

        full_path = os.path.join(out_dir, out_html)
        print(f"Writing interactive HTML to: {full_path}")
        page = ('<!DOCTYPE html>\n<html><head><meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                f'<title>{_html_escape(title or "Kokoro")}</title></head>\n<body>\n'
                + html + '\n</body></html>\n')
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(page)

    return html


_TEMPLATE = r"""
<div id="pv" style="font-family:system-ui,sans-serif;max-width:1080px">
<h2 id="pageTitle" style="margin:0 0 4px;font-size:20px;font-weight:700;color:#111;"></h2>
<div id="rulesBox" style="font-size:12.5px;color:#555;margin-bottom:12px;padding:7px 11px;background:#f3f4f6;border-radius:6px;display:none;max-height:60px;overflow-y:auto;"></div>
<div id="statsBox" style="display:none;margin:0 0 12px;padding:9px 12px;border:1px solid #e5e7eb;border-radius:8px;background:#fffdf7;font-size:13px;color:#333;">
    <div style="font-weight:700;margin-bottom:6px;color:#111;">Substitutions <span id="statsTotal" style="font-weight:400;color:#666;"></span></div>
    <div id="statsBody"></div>
</div>
<audio id="au" src="data:audio/wav;base64,__B64__"></audio>
<div style="display:flex;gap:8px;align-items:center;margin-bottom:2px">
    <button id="pp" style="padding:6px 14px;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer">▶ play</button>
    <input id="sp" type="range" min="0.5" max="1.5" step="0.05" value="1" style="width:120px">
    <span id="spl" style="font-size:12px;color:#555">1.00×</span>
    <span id="tt" style="font-size:12px;color:#555;margin-left:auto">0.00 / __DUR__s</span>
</div>
<div id="legend" style="font-size:11px;color:#666;margin:2px 0 8px;display:none;">
    <span style="border-bottom:3px solid #f59e0b;padding:0 3px;color:#b45309;font-weight:700;">phoneme</span>
    &nbsp;= altered by manipulation rule
</div>
<div id="durNote" style="font-size:11px;color:#666;margin:0 0 8px;display:none;">
    Every Kokoro token is shown with its predicted duration in ms (hover for frames): phonemes, diacritics (ʲ ː …), stress marks (ˈ), punctuation, ␣ = word gap, BOS/EOS = chunk start/end. Word totals include the word's own tokens only.
</div>
<canvas id="wf" width="1040" height="80" style="width:100%;height:80px;background:#0b1020;border-radius:6px;cursor:pointer"></canvas>
<div id="ph" style="display:flex;flex-wrap:wrap;align-items:flex-start;gap:0;margin-top:16px;"></div>
</div>
<script>
(function(){
const segs=__SEGS__, env=__ENV__, DUR=__DUR__;
const TITLE=__TITLE__, RULES=__RULES__, STATS=__STATS__;
const au=document.getElementById('au'), pp=document.getElementById('pp');
const ph=document.getElementById('ph'), wf=document.getElementById('wf'), ctx=wf.getContext('2d');
const tt=document.getElementById('tt'), sp=document.getElementById('sp'), spl=document.getElementById('spl');
const legend=document.getElementById('legend');
let view=[0,DUR];

if (TITLE) document.getElementById('pageTitle').textContent = TITLE;
if (RULES) {
    const rb = document.getElementById('rulesBox');
    rb.textContent = 'Rules applied: ' + RULES;
    rb.style.display = 'block';
}

if (segs.some(g=>g.alt)) legend.style.display='block';
if (segs.some(g=>g.ms!==undefined)) document.getElementById('durNote').style.display='block';

// ---- substitution-count legend ----
if (STATS && STATS.length) {
    const box=document.getElementById('statsBox'), body=document.getElementById('statsBody');
    const el=(tag,css,txt)=>{ const e=document.createElement(tag); if(css) e.style.cssText=css; if(txt!==undefined) e.textContent=txt; return e; };
    let grand=0;
    STATS.forEach((r,ri)=>{
        grand+=r.changed;
        const blk=el('div', 'margin:0 0 8px;' + (ri ? 'padding-top:7px;border-top:1px dashed #e5e7eb;' : ''));
        const head=el('div','font-size:12px;color:#555;margin-bottom:3px;');
        head.appendChild(el('b','color:#111;', 'Rule '+(ri+1)+': '));
        head.appendChild(document.createTextNode(r.label+'  '));
        head.appendChild(el('span','color:#b45309;font-weight:700;', '('+r.changed+' change'+(r.changed===1?'':'s')+')'));
        blk.appendChild(head);
        if (!r.groups.length) blk.appendChild(el('div','color:#999;font-size:12px;margin-left:12px;','no matches'));
        const isProb = r.kind==='substitute_prob';
        let shared=null;                                   // deterministic rules: all chips on one line
        r.groups.forEach(g=>{
            const row = (!isProb && shared) ? shared
                      : el('div','display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:3px 0 3px 12px;');
            if (!isProb) shared=row;
            if (isProb) {
                const lab = el('span','font-size:12px;color:#666;min-width:64px;', g.old+' ×'+g.total);
                if (g.draws !== null && g.draws !== undefined && g.draws !== g.total) {
                    lab.textContent += ' ('+g.draws+' draw'+(g.draws===1?'':'s')+')';
                    lab.title = 'geminates / repeats in a word reused an earlier choice, so there were fewer independent random draws than occurrences';
                }
                row.appendChild(lab);
            }
            g.rows.forEach(o=>{
                const pct = g.total ? Math.round(100*o.n/g.total) : 0;
                const chip=el('span','display:inline-flex;align-items:center;gap:6px;padding:3px 8px;border-radius:6px;'
                    + (o.changed ? 'background:#fef3c7;border:1px solid #fcd34d;' : 'background:#f3f4f6;border:1px solid #e5e7eb;'));
                chip.appendChild(el('span','font-size:16px;', g.old+' → '+o.new));
                chip.appendChild(el('b', o.changed ? 'color:#b45309;' : 'color:#555;', String(o.n)));
                if (o.p !== null) {
                    chip.appendChild(el('span','font-size:11px;color:#666;', pct+'% (target '+Math.round(100*o.p)+'%)'));
                    const bar=el('span','display:inline-block;width:46px;height:6px;background:#e5e7eb;border-radius:3px;position:relative;overflow:hidden;');
                    bar.appendChild(el('span','position:absolute;left:0;top:0;bottom:0;width:'+pct+'%;background:'+(o.changed?'#f59e0b':'#9ca3af')+';'));
                    bar.appendChild(el('span','position:absolute;top:0;bottom:0;width:2px;background:#111;left:calc('+Math.round(100*o.p)+'% - 1px);'));
                    chip.appendChild(bar);
                }
                if (!o.changed) chip.title='kept as itself (not highlighted)';
                row.appendChild(chip);
            });
            if (!row.parentNode) blk.appendChild(row);
        });
        body.appendChild(blk);
    });
    document.getElementById('statsTotal').textContent='— '+grand+' change'+(grand===1?'':'s')+' in total';
    box.style.display='block';
}
// units = word boxes (tokens with the same word index) + standalone tokens (space / BOS / EOS)
let groups=[];
segs.forEach((g,i)=>{
    const last=groups[groups.length-1];
    const inWord = g.w!==undefined && g.w!==null;
    if (inWord && last && last.w===g.w) { last.segIdx.push(i); last.e=g.e; last.ms+=g.ms; }
    else groups.push({w: inWord ? g.w : null, wl: inWord ? g.wl : null, s:g.s, e:g.e, ms:g.ms, segIdx:[i]});
});
const KIND_STYLE={
    phone:     'color:#111;',
    diacritic: 'color:#4b5563;',
    stress:    'color:#4b5563;',
    punct:     'color:#6b7280;',
    space:     'color:#9ca3af;',
    bos:       'color:#9ca3af;font-size:12px;',
    eos:       'color:#9ca3af;font-size:12px;',
};
const KIND_NAME={phone:'phoneme',diacritic:'diacritic',stress:'stress mark',punct:'punctuation / pause',
                 space:'word gap',bos:'chunk start (BOS)',eos:'chunk end (EOS)'};
function makeChip(i){
    const g=segs[i], altered=!!g.alt;
    const c=document.createElement('span');
    c.dataset.i=i; c.dataset.alt=altered?'1':'0'; c.dataset.k=g.k;
    c.style.cssText='display:inline-flex;flex-direction:column;align-items:center;vertical-align:top;'
        + 'padding:2px 4px;margin:0 1px;border-radius:6px;cursor:pointer;transition:.05s;min-width:14px;'
        + (altered ? 'color:#b45309;font-weight:700;' : (KIND_STYLE[g.k]||''));
    c.dataset.col = c.style.color;
    const lab=document.createElement('span');
    lab.textContent=g.l;
    if (altered) lab.style.borderBottom='3px solid #f59e0b';
    const ms=document.createElement('span');
    ms.textContent=Math.round(g.ms);
    ms.style.cssText='font-size:11px;font-weight:400;letter-spacing:0;opacity:.75;margin-top:1px;font-variant-numeric:tabular-nums;';
    c.appendChild(lab); c.appendChild(ms);
    c.title=(KIND_NAME[g.k]||g.k)+(altered?' · manipulated':'')+(g.t!==g.l&&g.k!=='space'?' · Kokoro token '+g.t:'')
        +'\n'+(g.fr!==null&&g.fr!==undefined ? g.fr+' frames = ' : '')+g.ms+' ms';
    c.onclick=()=>{ au.currentTime=g.s; au.play(); };
    return c;
}
groups.forEach((grp, gi)=>{
    if (grp.w===null) {                                    // standalone token between words
        const box=document.createElement('div');
        box.style.cssText='display:inline-block;vertical-align:top;margin:2px 6px 12px 0;padding:6px 2px;'
            + 'font-size:18px;border:1px dashed #e5e7eb;border-radius:8px;';
        box.appendChild(makeChip(grp.segIdx[0]));
        ph.appendChild(box);
        return;
    }
    const wrap=document.createElement('div');
    wrap.dataset.gi=gi;
    wrap.style.cssText='display:inline-block;vertical-align:top;margin:2px 6px 12px 0;padding:6px 8px;border-radius:9px;background:#f8f9fb;border:1.5px solid #e5e7eb;transition:.08s;';
    const wl=document.createElement('div');
    wl.textContent = grp.wl || '·';
    wl.style.cssText='font-size:23px;font-weight:700;color:#111;margin-bottom:4px;letter-spacing:.2px;';
    const wd=document.createElement('span');
    wd.textContent='  '+Math.round(grp.ms)+' ms';
    wd.style.cssText='font-size:11px;font-weight:400;color:#888;letter-spacing:0;';
    wl.appendChild(wd);
    wrap.appendChild(wl);
    const prow=document.createElement('div');
    prow.style.cssText='font-size:21px;white-space:nowrap;';
    grp.segIdx.forEach(i=>prow.appendChild(makeChip(i)));
    wrap.appendChild(prow);
    ph.appendChild(wrap);
});
const chips=[...ph.querySelectorAll('[data-i]')];
const wordWraps=[...ph.querySelectorAll('[data-gi]')];
function drawWave(){ const W=wf.width,H=wf.height; ctx.clearRect(0,0,W,H);
    const [a,b]=view, n=env.length;
    segs.forEach(g=>{ if(g.e<a||g.s>b)return; const x=(g.s-a)/(b-a)*W, w=(g.e-g.s)/(b-a)*W;
        ctx.fillStyle=(g.s<=au.currentTime&&au.currentTime<g.e)?'#2563eb55':'#ffffff10'; ctx.fillRect(x,0,Math.max(w,1),H);});
    ctx.strokeStyle='#ffffff30';
    groups.forEach(g=>{ if(g.w===null||g.s<a||g.s>b)return; const x=(g.s-a)/(b-a)*W;
        ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); });
    ctx.strokeStyle='#7dd3fc'; ctx.beginPath();
    for(let x=0;x<W;x++){ const t=a+(b-a)*x/W, idx=Math.floor(t/DUR*n); const v=env[Math.max(0,Math.min(n-1,idx))]||0;
        ctx.moveTo(x,H/2-v*H/2); ctx.lineTo(x,H/2+v*H/2);} ctx.stroke();
    const cx=(au.currentTime-a)/(b-a)*W; ctx.strokeStyle='#f43f5e'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(cx,0); ctx.lineTo(cx,H); ctx.stroke(); ctx.lineWidth=1;
}
function tick(){ const t=au.currentTime; let act=-1;
    for(let i=0;i<segs.length;i++){ if(segs[i].s<=t && t<segs[i].e){act=i;break;} }
    chips.forEach((c,i)=>{ const on=(+c.dataset.i===act); const altered=c.dataset.alt==='1';
        c.style.background=on?'#2563eb':'transparent';
        c.style.color=on?'#fff':c.dataset.col; });
    wordWraps.forEach(w=>{ w.style.borderColor='#e5e7eb'; w.style.background='#f8f9fb'; });
    if (act>=0){
        const wrap=chips[act].closest('[data-gi]');
        if (wrap){ wrap.style.borderColor='#2563eb'; wrap.style.background='#eef2ff'; }
    }
    tt.textContent=t.toFixed(2)+' / '+DUR.toFixed(2)+'s'; drawWave();
    if(!au.paused) requestAnimationFrame(tick);
}
pp.onclick=()=>{ au.paused?au.play():au.pause(); };
au.onplay=()=>{pp.textContent='⏸ pause'; tick();};
au.onpause=()=>{pp.textContent='▶ play';};
au.onended=()=>{pp.textContent='▶ play';};
sp.oninput=()=>{ au.playbackRate=+sp.value; spl.textContent=(+sp.value).toFixed(2)+'×'; };
wf.onclick=e=>{ const r=wf.getBoundingClientRect(); const f=(e.clientX-r.left)/r.width;
    au.currentTime=view[0]+(view[1]-view[0])*f; if(au.paused) drawWave(); };
wf.onwheel=e=>{ e.preventDefault(); const r=wf.getBoundingClientRect(); const f=(e.clientX-r.left)/r.width;
    const c=view[0]+(view[1]-view[0])*f, z=e.deltaY<0?0.8:1.25; let w=(view[1]-view[0])*z;
    w=Math.max(0.3,Math.min(DUR,w)); let a=c-f*w, b=a+w; a=Math.max(0,a); b=Math.min(DUR,a+w);
    view=[a,b]; drawWave(); };
drawWave();
})();
</script>
"""

# =============================================================================
# LISTENING TEST — participant page
#   build_listening_test(audio, segments, seg_words, text, seg_altered, ...)
#   One sentence at a time: listen -> text appears -> tap the words not
#   understood -> next. Answers are sent (optionally) to a Google Sheet and can
#   always be downloaded as CSV at the end.
# =============================================================================

# TEST_UI / _audio_data_uri / the participant-page template and its writer all
# live in experiment.py, which has NO kokoro/torch dependency, so building or
# rebuilding a participant page never needs the model. build_listening_test()
# below is only the "live session" path (straight off a synth_aligned() call);
# build_listening_test_from_html() in experiment.py is the "saved file" path —
# use that one whenever you can, especially for adding/editing conditions later.
from experiment import TEST_UI, _audio_data_uri, _write_test_page  # noqa: F401
from experiment import load_visualization, build_listening_test_from_html  # noqa: F401 (re-exported for convenience)


def _test_chunks(audio, segments, seg_words, text, seg_altered, sr):
    """Cut the synthesized story back into its sentences (using the BOS/EOS
    tokens that synth_aligned emits) and pair each with its words."""
    sents = one_sentence_chunks(text)
    spans, cur = [], None
    for i, seg in enumerate(segments):
        tok, s, e = seg[:3]
        if tok == "BOS":
            cur = {"s": s, "altered": {}}
        w = seg_words[i] if seg_words is not None else None
        if cur is not None and w is not None:
            cur["altered"][w] = cur["altered"].get(w, False) or bool(seg_altered and seg_altered[i])
        if tok == "EOS" and cur is not None:
            cur["e"] = e; spans.append(cur); cur = None
    if not spans:
        raise ValueError("no BOS/EOS tokens in segments — pass the output of the current synth_aligned()")
    if len(spans) != len(sents):
        raise ValueError(f"{len(spans)} audio chunks but {len(sents)} sentences in `text` — "
                         "pass the same text you gave text_to_ipa_chunks()")
    out = []
    for j, (sp, sent) in enumerate(zip(spans, sents)):
        words = sent.split()
        keys = sorted(sp["altered"])
        if len(keys) != len(words):
            print(f"⚠ sentence {j+1}: {len(words)} written words vs {len(keys)} spoken words "
                  f"— 'altered' flags may be shifted: {sent[:60]!r}")
        altered = [int(sp["altered"][keys[k]]) if k < len(keys) else 0 for k in range(len(words))]
        a = audio[int(round(sp["s"] * sr)):int(round(sp["e"] * sr))]
        out.append({"words": words, "altered": altered, "audio": a})
    return out


def build_listening_test(audio, segments, seg_words, text, seg_altered=None, sr=SR,
                         out_html="listening_test.html", out_dir="listening_test",
                         test_id="story", version="v1", submit_url=None, max_plays=1,
                         lang="en", ui=None, completion_url=None, audio_format="mp3"):
    """Self-contained participant page (one HTML file, audio embedded).

    audio, segments, seg_words, seg_altered : output of synth_aligned()
    text        : the SAME text you passed to text_to_ipa_chunks(); its
                  sentences are what participants read.
    test_id     : name of the study/story, stored with every answer
    version     : which version this file is (e.g. "clean"), stored with every answer
    submit_url  : Google Apps Script web-app URL that appends answers to a sheet
                  (None = participants download a CSV and send it to you)
    max_plays   : listens allowed per sentence before the text appears (None = unlimited)
    lang        : "en" or "it" interface text; `ui` = dict to override any string
    completion_url : optional link shown at the end (e.g. a Prolific completion URL)

    The participant code is read from the URL (?pid=..., or Prolific's
    ?PROLIFIC_PID=...) or typed on the first screen.
    Answers are one row per word: test, version, participant, session, chunk,
    word_pos, word, word_clean, altered (0/1, from your rules), not_understood
    (0/1), plays, listen_ms, response_ms, time."""
    chunks = _test_chunks(np.asarray(audio, dtype=np.float32), segments, seg_words, text, seg_altered, sr)
    payload = [{"audio": _audio_data_uri(c["audio"], sr, audio_format),
                "words": c["words"], "altered": c["altered"]} for c in chunks]
    return _write_test_page(payload, out_html, out_dir, test_id, version, submit_url,
                            max_plays, lang, ui, completion_url)