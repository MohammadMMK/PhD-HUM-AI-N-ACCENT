# ============================================================
# read_with_voices.py  (v2)
#
# Loads the 3 saved voice embeddings (Female / Male / Mix), makes the
# Darker / Brighter versions of each by post-processing, optionally transfers
# Italian duration / pitch contour / energy from a reference voice (any of the
# three), and writes ONE self-contained HTML page with a voice selector.
#
#   Voice : Female / Male / Mix          -> the 3 embeddings (EMBEDDINGS)
#   Style : Original / Darker / Brighter -> post-processing   (STYLES)
#
# Usage:
#   python read_with_voices.py "Ciao, oggi parliamo di piante."
#   python read_with_voices.py --file story1.txt --out story1.html
#   python read_with_voices.py "Ciao." --ref male          # prosody from the Male voice
#   python read_with_voices.py "Ciao." --ref none          # each voice keeps its own prosody
#
# Needs next to this file: italian_voice_bank3.py, blend_voices.py,
# functions.py, accent_tools.py
# ============================================================

import argparse
import base64
import html
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from italian_voice_bank3 import ItalianVoiceBank
from accent_tools import merge_results, SR

# ============================================================
# SETTINGS - edit these
# ============================================================

# 1) The 3 saved embeddings: label -> path of the .pt file
EMBEDDINGS = {
    "Female": "custom_voices/nicola_10.pt",
    "Male":   "custom_voices/nicola_90.pt",
    "Mix":    "custom_voices/nicola_50.pt",
}

# 2) Post-processing per style (applied to every embedding)
#    semitones   : pitch level shift (re-decoded inside Kokoro)
#    pitch_range : pitch excursion factor, keep 0.8-1.2
#    formant     : vocal-tract length (Praat), <1 darker / >1 brighter, keep 0.90-1.10
STYLES = {
    "Original": dict(semitones=0.0,  pitch_range=1.00, formant=1.00),
    "Darker":   dict(semitones=-2.0, pitch_range=0.85, formant=0.93),
    "Brighter": dict(semitones=+2.0, pitch_range=1.15, formant=1.07),
}

# 3) Prosody transfer
#    REF: which embedding gives duration / contour / energy to ALL voices
#         "Female" | "Male" | "Mix" | None (each voice keeps its own prosody)
#    Or set a different reference per voice in REF_PER_VOICE, e.g. {"Male": "Female"}
REF = None
REF_PER_VOICE: dict[str, str | None] = {}

#    What is transferred and how (same options as accent_tools.render_condition)
TRANSFER = dict(
    duration=False,          # word / phoneme timings
    pitch=False,             # F0 contour
    energy=False,            # loudness contour
    dur_mode="raw",         # duration transfer mode
    dur_scope="sentence",   # duration matching scope
    f0_mode="meanstd",      # contour mapped onto the target voice's own mean/std
    energy_mode="meanstd",
    norm_scope="text",
)

# ============================================================


def load_embedding(path: str | Path) -> torch.Tensor:
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):                         # tolerate {"voice": tensor} style saves
        t = next(v for v in t.values() if torch.is_tensor(v))
    return t


def render_all(bank: ItalianVoiceBank, text: str, embeddings: dict, styles: dict,
               ref: str | None, ref_per_voice: dict, transfer: dict) -> dict[str, np.ndarray]:
    """Return {"Female|Darker": audio, ...}. Each embedding is decoded once,
    its styles are derived from that render."""
    bank.transfer = {**bank.transfer, **transfer}
    cond = bank.prepare_text(text, "reader")
    tensors = {label: load_embedding(p) for label, p in embeddings.items()}

    out = {}
    for label, tensor in tensors.items():
        r = ref_per_voice.get(label, ref)
        if r is not None and r not in tensors:
            raise SystemExit(f"Reference '{r}' is not one of {list(tensors)}")
        name = label.lower()
        print(f"=== {label} (prosody from {r or 'itself'}) ===")
        if r is None or r == label:
            # own prosody, or reference is itself -> no transfer needed
            base_results = bank.render_base(cond, name, tensor, None)
        else:
            # reference given as (name, tensor), same form as the target voice
            base_results = bank.render_base(cond, name, tensor, (r.lower(), tensors[r]))
        for style, spec in styles.items():
            res = bank.make_variant(base_results, tensor, f"{name}_{style.lower()}", **spec)
            out[f"{label}|{style}"] = merge_results(res)["audio"]
            print(f"   {style}: {spec}")
    return out


def wav_base64(audio: np.ndarray, sr: int = SR) -> str:
    buf = io.BytesIO()
    sf.write(buf, np.clip(np.asarray(audio, dtype=np.float32), -1, 1), sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------
# HTML
# ------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Italian Voice Reader</title>
<style>
  :root { --bg:#f6f4ef; --card:#fff; --ink:#1d1d1b; --muted:#6b6a65; --line:#e2dfd7;
          --accent:#1f6f5c; --accent-ink:#fff; --chip:#efece5; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#151614; --card:#1e201d; --ink:#ecebe6; --muted:#9a9890; --line:#2f322d;
            --accent:#4fb596; --accent-ink:#0f1a16; --chip:#262923; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:760px; margin:0 auto; padding:32px 16px 48px; }
  h1 { font-size:1.4rem; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 24px; font-size:.95rem; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px; }
  .row { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:14px; }
  .row label { width:52px; color:var(--muted); font-size:.85rem; text-transform:uppercase; letter-spacing:.04em; }
  .seg { display:inline-flex; background:var(--chip); border-radius:10px; padding:3px; gap:3px; }
  .seg button { border:0; background:transparent; color:var(--ink); font:inherit; font-size:.95rem;
                padding:7px 16px; border-radius:8px; cursor:pointer; }
  .seg button[aria-pressed="true"] { background:var(--accent); color:var(--accent-ink); font-weight:600; }
  .player { display:flex; align-items:center; gap:12px; margin-top:6px; flex-wrap:wrap; }
  audio { flex:1 1 260px; min-width:0; }
  .now, .info { font-size:.85rem; color:var(--muted); }
  .dl { font-size:.85rem; color:var(--accent); text-decoration:none; }
  .text { margin-top:20px; font-size:1.12rem; line-height:1.75; white-space:pre-wrap; }
</style>
</head>
<body>
<main>
  <h1>Italian Voice Reader</h1>
  <p class="sub">Pick a voice and a style; switching keeps your place in the text.</p>
  <div class="card">
    <div class="row"><label>Voice</label><div class="seg" id="voice"></div></div>
    <div class="row"><label>Style</label><div class="seg" id="style"></div></div>
    <div class="player">
      <audio id="audio" controls preload="auto"></audio>
      <span class="now" id="now"></span>
      <a class="dl" id="dl">Download</a>
    </div>
    <p class="info">__INFO__</p>
    <div class="text">__TEXT__</div>
  </div>
</main>
<script>
const AUDIO = __AUDIO__;            // {"Female|Darker": base64 wav, ...}
const VOICES = __VOICES__, STYLES = __STYLES__;
const audio = document.getElementById("audio");
const urls = {};
for (const [k, b64] of Object.entries(AUDIO)) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  urls[k] = URL.createObjectURL(new Blob([bytes], {type: "audio/wav"}));
}
const state = {voice: VOICES[0], style: STYLES[0]};
function seg(id, options, key) {
  const box = document.getElementById(id);
  for (const o of options) {
    const b = document.createElement("button");
    b.textContent = o; b.dataset.value = o;
    b.onclick = () => { state[key] = o; select(); };
    box.appendChild(b);
  }
}
function select() {
  for (const [id, key] of [["voice","voice"],["style","style"]])
    for (const b of document.querySelectorAll("#" + id + " button"))
      b.setAttribute("aria-pressed", b.dataset.value === state[key]);
  const k = state.voice + "|" + state.style;
  const t = audio.currentTime, playing = !audio.paused;
  if (audio.dataset.k !== k) {
    audio.dataset.k = k; audio.src = urls[k];
    audio.addEventListener("loadedmetadata", () => {
      audio.currentTime = Math.min(t, audio.duration || t);
      if (playing) audio.play();
    }, {once: true});
  }
  document.getElementById("now").textContent = state.voice + " · " + state.style;
  const dl = document.getElementById("dl");
  dl.href = urls[k]; dl.download = (state.voice + "_" + state.style).toLowerCase() + ".wav";
}
seg("voice", VOICES, "voice"); seg("style", STYLES, "style"); select();
</script>
</body>
</html>
"""


def write_html(text: str, audios: dict, voices: list, styles: list, info: str, out_path: Path) -> Path:
    page = (PAGE
            .replace("__TEXT__", html.escape(text.replace("###", "").strip()))
            .replace("__INFO__", html.escape(info))
            .replace("__AUDIO__", json.dumps({k: wav_base64(a) for k, a in audios.items()}))
            .replace("__VOICES__", json.dumps(voices))
            .replace("__STYLES__", json.dumps(styles)))
    out_path.write_text(page, encoding="utf-8")
    return out_path


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Read a text with 3 embeddings x 3 styles into one HTML page.")
    ap.add_argument("text", nargs="?", help="the text to read (or use --file)")
    ap.add_argument("--file", help="text file to read (### block separators are fine)")
    ap.add_argument("--out", default="reader.html", help="output HTML file")
    ap.add_argument("--ref", help="override REF: female | male | mix | none")
    args = ap.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = args.text or ("Irene Valli lavorava come botanica all'università di Bologna, "
                             "e da molti anni studiava le piante che crescono vicino al mare.")

    ref = REF
    if args.ref:
        ref = None if args.ref.lower() == "none" else args.ref.capitalize()

    bank = ItalianVoiceBank(out_dir=Path(args.out).parent or ".")
    audios = render_all(bank, text, EMBEDDINGS, STYLES, ref, REF_PER_VOICE, TRANSFER)

    on = [k for k in ("duration", "pitch", "energy") if TRANSFER.get(k)]
    info = (f"Prosody ({', '.join(on) or 'nothing'}) from: {ref}" if ref else "Each voice uses its own prosody")
    path = write_html(text, audios, list(EMBEDDINGS), list(STYLES), info, Path(args.out))
    print(f"\nDone: open {path.resolve()}")


if __name__ == "__main__":
    main()
