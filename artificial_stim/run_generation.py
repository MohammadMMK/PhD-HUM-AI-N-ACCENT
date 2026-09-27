"""
STEP 1 — GENERATION (needs the "kokoro" conda env).

For every story × version, synthesizes the audio and saves the complete result
(audio + every token's duration / word / alteration) as
    outputs/story<id>__<version>.html
That file is the saved data; build_experiment.py reads it later (no Kokoro needed).

To add a story: add it to STORIES and run again. With SKIP_EXISTING = True,
versions that already exist in outputs/ are not generated again.
"""
import time
import torch
import soundfile as sf
from pathlib import Path
from functions import (
    text_to_ipa_chunks, manipulate, check_ipa,
    synth_aligned, audiovisualize_interactive,
)

VOICES_DIR = Path("custom_voices")
OUT_DIR = Path("output_online"); OUT_DIR.mkdir(exist_ok=True)
SKIP_EXISTING = True      # don't regenerate versions that are already in outputs/

voice     = torch.load(VOICES_DIR / "nicola_80.pt",  weights_only=True)   # the voice you hear
dur_voice = torch.load(VOICES_DIR / "nicola_100.pt", weights_only=True)   # reference timing

# ---------------------------------------------------------------------------
# stories — id: text. The id is what participants choose and what is stored.
# ---------------------------------------------------------------------------

with open("story1.txt", "r", encoding="utf-8") as f:
    story1 = f.read()
# story1 = story1[:200] 
STORIES = {
    "1": story1,
}

# ---------------------------------------------------------------------------
# error rule sets — add a dict here for a new kind of error
# ---------------------------------------------------------------------------
RULESETS = {
    "consistent": [
        {"kind": "substitute_prob", "old": "d", "choices": {"ʈ": 100}, "same_in_word": True},
        {"kind": "substitute_prob", "old": "k", "choices": {"χ": 100}, "same_in_word": True},
        {"kind": "substitute_prob", "old": "e", "choices": {"ø": 100}, "same_in_word": True},
    ],
    "less_consistent": [
        {"kind": "substitute_prob", "old": "d", "choices": {"ʈ": 33, "θ": 33, "dʲ": 33}, "same_in_word": True},
        {"kind": "substitute_prob", "old": "k", "choices": {"x": 33, "χ": 33, "kʲ": 33}, "same_in_word": True},
        {"kind": "substitute_prob", "old": "e", "choices": {"ø": 50, "e": 50}, "same_in_word": False},
        {"kind": "substitute_prob", "old": "o", "choices": {"ə": 50, "o": 50}, "same_in_word": False},
    ],
}

# timing variants; comment one out to skip it
TIMINGS = {
    "native":   lambda ipa, origins: dict(dur_source="original", original_chunks=ipa, origins=origins),
    # "accented": lambda ipa, origins: dict(dur_source="manipulated"),
}


def render(name, text, chunks, masks=None, rules=None, stats=None, **timing):
    if SKIP_EXISTING and (OUT_DIR / f"{name}.html").exists():
        print(f"  {name}: already in outputs/, skipped")
        return
    t = time.time()
    audio, segments, seg_words, seg_altered = synth_aligned(
        chunks, voice, masks=masks, dur_voice=dur_voice, **timing)
    audiovisualize_interactive(
        audio, segments, seg_words=seg_words, text=text, seg_altered=seg_altered,
        title=name, rules=rules, stats=stats, out_html=f"{name}.html")
    sf.write(OUT_DIR / f"{name}.wav", audio, 24000)
    print(f"  {name}: {len(audio)/24000:.1f}s audio in {time.time()-t:.1f}s")


for sid, text in STORIES.items():
    print(f"story {sid}")
    ipa_chunks = text_to_ipa_chunks(text)
    check_ipa(ipa_chunks)
    render(f"story{sid}__clean", text, ipa_chunks, dur_source="manipulated")
    for rname, rules in RULESETS.items():
        manip_chunks, masks, stats, origins = manipulate(
            ipa_chunks, rules, return_mask=True, return_stats=True, return_origin=True, seed=42)
        for tname, timing in TIMINGS.items():
            render(f"story{sid}__{rname}_dur_forced_{tname}", text, manip_chunks, masks, rules, stats,
                   **timing(ipa_chunks, origins))

print("done — saved pages are in ./outputs/. Next: python build_experiment.py")