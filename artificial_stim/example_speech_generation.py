"""
Example usage of speech_generation.py — baseline, manipulated, and manipulated
with the baseline's prosody forced onto it.

    python example_speech_generation.py

Writes three wav files to outputs/ and prints what changed. The first run
downloads the Kokoro weights (~330 MB) into kokoro_model/.
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
except AttributeError:
    pass

from pathlib import Path

from text_manipulation import text_to_ipa_chunks, manipulate, describe_rules
from speech_generation import (extract_prosody, synthesize, synthesize_forced,
                               synthesize_pair)

TEXT  = "Il gatto dorme sul tetto caldo. Che cosa hai detto?"
VOICE = "im_nicola"                            # any built-in voice, or custom_voices/<name>.pt
SEED  = 0                                      # the vocoder adds noise; a seed pins it
OUT   = Path("outputs"); OUT.mkdir(exist_ok=True)

RULES = [
    {"map": {"k": "χ"}, "same_in_word": True},  # 1:1 substitution — token count unchanged
    {"map": {"p": "pʰ", "d": "dʲ"}},            # these ADD a token (ʰ, ʲ)
]


def banner(title):
    print("\n" + "=" * 72); print(title); print("=" * 72)


# --- 1. the two phoneme strings ----------------------------------------------
banner("1. TEXT -> IPA -> MANIPULATED IPA")
ipa = text_to_ipa_chunks(TEXT)
manip, masks = manipulate(ipa, RULES, seed=42, return_mask=True)
print("rules:", describe_rules(RULES))
for a, b in zip(ipa, manip):
    print(f"  before: {a}")
    print(f"  after : {b}")


# --- 2. plain, fast generation ------------------------------------------------
banner("2. PLAIN GENERATION")
baseline = synthesize(ipa, VOICE, seed=SEED)
free = synthesize(manip, VOICE, seed=SEED)
print(f"  baseline            {baseline.seconds:.3f}s -> {baseline.save(OUT / 'baseline.wav')}")
print(f"  manipulated, free   {free.seconds:.3f}s -> {free.save(OUT / 'manipulated_free.wav')}")
print("  the free version re-times the whole utterance, so it no longer lines up")


# --- 3. the prosody itself ----------------------------------------------------
# Cheap: it stops before the text encoder and the vocoder, and makes no audio.
banner("3. PROSODY OF THE BASELINE")
donors = extract_prosody(ipa, VOICE)
for p in donors:
    print(" ", p)


# --- 4. the same manipulation, with that prosody forced ----------------------
banner("4. FORCED GENERATION")
forced = synthesize_forced(manip, VOICE, donor=donors, seed=SEED)
kept, total = forced.n_forced
print(f"  manipulated, forced {forced.seconds:.3f}s -> {forced.save(OUT / 'manipulated_forced.wav')}")
print(f"  {kept}/{total} tokens took the baseline's timing; the rest are the tokens")
print("  the rules inserted, which keep their own predicted duration:")
for chunk in forced.chunks:
    own = [c for c, f in zip(chunk.phonemes, chunk.forced[1:-1]) if not f]
    print("   ", chunk.phonemes)
    print("    own time:", own or "—")
print(f"\n  baseline {baseline.seconds:.3f}s vs forced {forced.seconds:.3f}s — the difference is"
      f" exactly the {total - kept} inserted token(s)")


# --- 5. one call for a stimulus pair ------------------------------------------
# Same thing in one line: the donor is extracted once and used for both tracks,
# so they share their prosody AND their vocoder noise.
banner("5. synthesize_pair")
base2, manip2 = synthesize_pair(ipa, manip, VOICE, seed=SEED)
print("  ", base2)
print("  ", manip2)
print("  identical to step 4:", abs(manip2.seconds - forced.seconds) < 1e-9)
print("\n  to see it, hand the forced audio to the player in functions.py:")
print("    audio, segs, words, alt = synth_aligned(manip, VOICE, masks=masks, donor=ipa, seed=0)")
print("    audiovisualize_interactive(audio, segs, seg_words=words, text=TEXT, seg_altered=alt)")
