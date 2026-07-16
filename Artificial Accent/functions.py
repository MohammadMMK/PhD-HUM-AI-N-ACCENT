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
  substitute  : replace phonemes via a map (old -> new)      (old/new may be a CHAR or a TOKEN ID)
  retroflex   : preset map  t→ʈ d→ɖ n→ɳ s→ʂ r→ɽ
  japanese    : preset map  s→ɕ  ʃ→ɕ  tʃ/ʧ→ʨ  dʒ/ʤ→ʥ  p→ɸ
  spanish     : preset map  b→β  v→β  d→ð  ɡ→ɣ        (ɡ = U+0261, the script-g espeak emits)

COMMON OPTIONS (per rule)
  scope = "all"              -> every occurrence
        = "outside_cluster"  -> only singletons (target NOT adjacent to another consonant).
                                (e.g. Spanish: spirantize b d g only intervocalically.)
  targets = [...]            -> aspirate/palatalize only: restrict the target set (chars or ids).

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
def _scan(s, keys, action, scope, protect_affricates):
    keys = sorted(set(keys), key=len, reverse=True)        # longest match wins
    out, i, n = [], 0, len(s)
    while i < n:
        key = next((k for k in keys if s.startswith(k, i)), None)
        if key is None:
            out.append(s[i]); i += 1; continue
        end = i + len(key)
        fire = True
        if len(key) == 1 and protect_affricates and _is_affricate_member(s, i):
            fire = False                                   # protect single-char hits on tʃ/ts/dʒ/dz
        if fire and scope == "outside_cluster" and _in_cluster(s, i, end):
            fire = False                                   # singletons only
        if fire:
            out.append(action(key)); i = end
        else:
            out.append(s[i]); i += 1
    return ''.join(out)

# ---- one rule -> one chunk ----------------------------------------------------
def _apply_rule(s, rule, protect_affricates):
    kind  = rule["kind"]
    scope = rule.get("scope", "all")

    if kind in ("aspirate", "palatalize"):
        default = VOICELESS_STOPS if kind == "aspirate" else CONSONANTS
        targets = {_norm(t) for t in rule.get("targets", default)}
        suffix  = 'ʰ' if kind == "aspirate" else 'ʲ'
        return _scan(s, targets, lambda k: k + suffix, scope, protect_affricates)

    if kind in PRESETS or kind == "substitute":
        mapping = (rule["map"] if "map" in rule else {rule["old"]: rule["new"]}) \
                  if kind == "substitute" else PRESETS[kind]
        mapping = _norm_map(mapping)
        return _scan(s, mapping.keys(), lambda k: mapping[k], scope, protect_affricates)

    raise ValueError(f"unknown rule kind: {kind!r}")

# ---- public entry point -------------------------------------------------------
def manipulate(ipa, rules, protect_affricates=True):
    """ipa: str or list of chunks. rules: list of rule dicts, applied IN ORDER."""
    single = isinstance(ipa, str)
    chunks = [ipa] if single else list(ipa)
    result = []
    for s in chunks:
        for rule in rules:
            s = _apply_rule(s, rule, protect_affricates)
        result.append(s)
    return result[0] if single else result