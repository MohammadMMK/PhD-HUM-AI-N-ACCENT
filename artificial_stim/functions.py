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
CONSONANTS = set('bcdfhjklmnpqrstvwxz') | set('βɕçɖðʤɟɡɥʝɰŋɳɲɴɸθɹɾɻʁɽʂʃʈʧʋɣχʎʒʔ') | set('ʣʥʦʨꭧ')
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

# ---- core scan: longest-match, guarded ----------------------------------------
def _scan(s, keys, action, scope, protect_affricates, in_mask=None, guard=None):
    """Longest-match scan. `in_mask` (optional) is a bool list aligned to `s`,
    carrying forward alteration flags from earlier rules in the same
    manipulate() call. `guard` (optional) is a callable (s, start, end) -> bool;
    the rule only fires where it returns True — used for positional conditions
    such as word-finality. Returns (output_string, out_mask)."""
    if in_mask is None:
        in_mask = [False] * len(s)
    keys = sorted(set(keys), key=len, reverse=True)
    out, mask, i, n = [], [], 0, len(s)
    while i < n:
        key = next((k for k in keys if s.startswith(k, i)), None)
        if key is None:
            out.append(s[i]); mask.append(in_mask[i]); i += 1; continue
        end = i + len(key)
        fire = True
        if len(key) == 1 and protect_affricates and _is_affricate_member(s, i):
            fire = False                                   # protect single-char hits on tʃ/ts/dʒ/dz
        if fire and scope == "outside_cluster" and _in_cluster(s, i, end):
            fire = False                                   # singletons only
        if fire and guard is not None and not guard(s, i, end):
            fire = False                                   # positional condition failed
        if fire:
            piece = action(key)
            out.append(piece)
            mask.extend([True] * len(piece))
            i = end
        else:
            out.append(s[i]); mask.append(in_mask[i]); i += 1
    return ''.join(out), mask

# ---- one rule -> one chunk ----------------------------------------------------
def _apply_rule(s, rule, protect_affricates, in_mask=None):
    kind  = rule["kind"]
    scope = rule.get("scope", "all")

    if kind in ("aspirate", "palatalize"):
        default = VOICELESS_STOPS if kind == "aspirate" else CONSONANTS
        targets = {_norm(t) for t in rule.get("targets", default)}
        suffix  = 'ʰ' if kind == "aspirate" else 'ʲ'
        return _scan(s, targets, lambda k: k + suffix, scope, protect_affricates, in_mask)

    if kind == "nasal_final":
        targets = {_norm(t) for t in rule.get("targets", VOWELS)}
        bad = targets - VOWELS
        if bad:
            raise ValueError(f"nasal_final targets must be vowels; got {sorted(bad)}")
        return _scan(s, targets, lambda k: k + NASAL_TILDE, scope, protect_affricates,
                     in_mask, guard=lambda st, a, b: _is_word_final(st, b))

    if kind in PRESETS or kind == "substitute":
        mapping = (rule["map"] if "map" in rule else {rule["old"]: rule["new"]}) \
                  if kind == "substitute" else PRESETS[kind]
        mapping = _norm_map(mapping)
        if kind == "substitute":
            suffix = ('ʰ' if rule.get("aspirate") else '') + ('ʲ' if rule.get("palatalize") else '')
            if suffix:
                mapping = {k: v + suffix for k, v in mapping.items()}
        return _scan(s, mapping.keys(), lambda k: mapping[k], scope, protect_affricates, in_mask)

    raise ValueError(f"unknown rule kind: {kind!r}")

# ---- public entry point -------------------------------------------------------
def manipulate(ipa, rules, protect_affricates=True, return_mask=False):
    """ipa: str or list of chunks. rules: list of rule dicts, applied IN ORDER.
    return_mask=False (default): identical behavior to before — returns just
    the manipulated string (or list of strings).
    return_mask=True: also returns a parallel bool mask (or list of masks)
    marking every character that any rule altered or introduced."""
    single = isinstance(ipa, str)
    chunks = [ipa] if single else list(ipa)
    out_strs, out_masks = [], []
    for s in chunks:
        mask = [False] * len(s)
        for rule in rules:
            s, mask = _apply_rule(s, rule, protect_affricates, mask)
        out_strs.append(s)
        out_masks.append(mask)
    if return_mask:
        return (out_strs[0], out_masks[0]) if single else (out_strs, out_masks)
    return out_strs[0] if single else out_strs

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

SR = 24000
FRAME_SAMPLES = 600
BASE     = VOWELS | CONSONANTS
TRAILING = {'ː','ʰ','ʲ','\u0303','\u1d5d','↓','→','↗','↘'}
LEADING  = {'ˈ','ˌ'}


def align_phonemes(phonemes, pred_dur, n_samples=None, sr=SR, lead_trim_frames=0, altered_mask=None):
    """altered_mask (optional): bool list the same length as `phonemes`, e.g.
    from manipulate(..., return_mask=True). If given, each emitted segment
    gets an `altered` flag (True if any character contributing to it — base
    or a merged trailing diacritic — was flagged)."""
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
    start_f, acc = [], 0
    for d in dur:
        start_f.append(acc); acc += d
    segs, pending, pending_altered = [], None, False
    word_idx = 0
    seg_word_idx, seg_altered = [], []
    for k, c in enumerate(kept, start=1):
        s, e = start_f[k], start_f[k] + dur[k]
        c_alt = kept_altered[k - 1]
        if c in BASE:
            segs.append([c, pending if pending is not None else s, e])
            seg_word_idx.append(word_idx)
            seg_altered.append(c_alt or pending_altered)
            pending, pending_altered = None, False
        elif c in TRAILING and segs:
            segs[-1][0] += c; segs[-1][2] = e
            if c_alt:
                seg_altered[-1] = True
        elif c in LEADING:
            pending = s if pending is None else pending
            pending_altered = pending_altered or c_alt
        else:
            pending, pending_altered = None, False
            if c == ' ':
                word_idx += 1
    return [(lab, f2s(s), f2s(e)) for lab, s, e in segs], seg_word_idx, seg_altered


def synth_aligned(ipa_chunks, voice, speed=1.0, sr=SR, masks=None):
    """masks (optional): list parallel to ipa_chunks, each entry the bool
    mask for that chunk from manipulate(..., return_mask=True)."""
    audios, segments, seg_words, seg_altered = [], [], [], []
    t0, word_offset = 0.0, 0
    for ci, ps in enumerate(ipa_chunks):
        if not ps.strip():
            continue
        r = next(gen.generate_from_tokens(tokens=ps, voice=voice, speed=speed))
        a = r.audio.detach().cpu().numpy()
        chunk_mask = masks[ci] if masks is not None else None
        segs, widx, altd = align_phonemes(r.phonemes, r.pred_dur, n_samples=len(a), sr=sr,
                                           altered_mask=chunk_mask)
        segments += [(lab, s + t0, e + t0) for lab, s, e in segs]
        seg_words += [w + word_offset for w in widx]
        seg_altered += altd
        word_offset += (widx[-1] + 1) if widx else 0
        audios.append(a); t0 += len(a) / sr
    return np.concatenate(audios), segments, seg_words, seg_altered


def audiovisualize_interactive(audio, segments, sr=24000, out_html=None, per_row=None,
                                seg_words=None, text=None, seg_altered=None,
                                title=None, rules=None):
    """title (optional): short heading shown at the top of the player.
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
    segs = [{"l": lab, "s": round(s, 4), "e": round(e, 4)} for lab, s, e in segments]
    dur = len(audio)/sr
    if seg_words is not None and text is not None:
        n_words = (max(seg_words) + 1) if seg_words else 0
        word_labels = text.split()
        if len(word_labels) != n_words:
            print(f"⚠ word count mismatch: text has {len(word_labels)} words, "
                  f"phoneme string implies {n_words} — labels may be misaligned")
        for i, seg in enumerate(segs):
            wi = seg_words[i]
            seg["w"] = wi
            seg["wl"] = word_labels[wi] if wi < len(word_labels) else ""
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
                    .replace("__RULES__", json.dumps(rules_text))
    if out_html:
        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)

        full_path = os.path.join(out_dir, out_html)
        print(f"Writing interactive HTML to: {full_path}")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(html)

    return html


_TEMPLATE = r"""
<div id="pv" style="font-family:system-ui,sans-serif;max-width:1080px">
<h2 id="pageTitle" style="margin:0 0 4px;font-size:20px;font-weight:700;color:#111;"></h2>
<div id="rulesBox" style="font-size:12.5px;color:#555;margin-bottom:12px;padding:7px 11px;background:#f3f4f6;border-radius:6px;display:none;max-height:60px;overflow-y:auto;"></div>
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
<canvas id="wf" width="1040" height="80" style="width:100%;height:80px;background:#0b1020;border-radius:6px;cursor:pointer"></canvas>
<div id="ph" style="display:flex;flex-wrap:wrap;align-items:flex-start;gap:0;margin-top:16px;"></div>
</div>
<script>
(function(){
const segs=__SEGS__, env=__ENV__, DUR=__DUR__;
const TITLE=__TITLE__, RULES=__RULES__;
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
let groups=[];
if (segs.length && segs[0].w !== undefined) {
    segs.forEach((g,i)=>{
        const last=groups[groups.length-1];
        if (last && last.w===g.w) { last.segIdx.push(i); last.e=g.e; }
        else { groups.push({w:g.w, wl:g.wl, s:g.s, e:g.e, segIdx:[i]}); }
    });
} else {
    groups = segs.map((g,i)=>({w:i, wl:null, s:g.s, e:g.e, segIdx:[i]}));
}
groups.forEach((grp, gi)=>{
    const wrap=document.createElement('div');
    wrap.dataset.gi=gi;
    wrap.style.cssText='display:inline-block;vertical-align:top;margin:2px 10px 12px 0;padding:6px 10px;border-radius:9px;background:#f8f9fb;border:1.5px solid #e5e7eb;transition:.08s;';
    if (grp.wl !== null) {
        const wl=document.createElement('div');
        wl.textContent = grp.wl || '·';
        wl.style.cssText='font-size:23px;font-weight:700;color:#111;margin-bottom:4px;letter-spacing:.2px;';
        wrap.appendChild(wl);
    }
    const prow=document.createElement('div');
    prow.style.cssText='font-size:21px;letter-spacing:1.5px;white-space:nowrap;';
    grp.segIdx.forEach(i=>{
        const s=document.createElement('span');
        s.textContent=segs[i].l; s.dataset.i=i;
        const altered = !!segs[i].alt;
        s.dataset.alt = altered ? '1' : '0';
        s.style.cssText='padding:2px 6px;margin:0 1px;border-radius:6px;cursor:pointer;transition:.05s;'
            + (altered ? 'border-bottom:3px solid #f59e0b;color:#b45309;font-weight:700;' : '');
        if (altered) s.title = 'manipulated phoneme';
        s.onclick=()=>{ au.currentTime=segs[i].s; au.play(); };
        prow.appendChild(s);
    });
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
    groups.forEach(g=>{ if(g.s<a||g.s>b)return; const x=(g.s-a)/(b-a)*W;
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
        c.style.color=on?'#fff':(altered?'#b45309':'#111'); });
    wordWraps.forEach(w=>{ w.style.borderColor='#e5e7eb'; w.style.background='#f8f9fb'; });
    if (act>=0){
        const wrap=chips[act].closest('[data-gi]');
        if (wrap){ wrap.style.borderColor='#2563eb'; wrap.style.background='#eef2ff';
            wrap.scrollIntoView({block:'nearest',inline:'nearest',behavior:'smooth'}); }
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