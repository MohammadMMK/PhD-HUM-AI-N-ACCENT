"""
Example usage of text_manipulation.py — text -> IPA -> manipulated IPA -> statistics.

Run it from this folder:

    python example_text_manipulation.py

Nothing here loads the Kokoro acoustic model: only espeak-ng (through misaki) is
needed. To hear the result, pass the manipulated chunks to synth_chunks() or
synth_aligned() from functions.py, which does load the model.
"""

import sys

# Windows consoles default to cp1252, which cannot print IPA.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:                     # non-reconfigurable stream
    pass

from text_manipulation import (
    text_to_ipa_chunks, one_sentence_chunks, check_ipa,
    manipulate, describe_rules, substitution_stats, print_stats,
    aligned_pairs, phoneme_units,
)

SAMPLE_TEXT = (
    "Il gatto dorme sul tetto caldo. "
    "Domani andremo a Bologna con la zia Giulia! "
    "Che cosa hai detto?"
)


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --- 1. text -> sentences -> IPA chunks ---------------------------------------
banner("1. TEXT -> IPA CHUNKS")
for s in one_sentence_chunks(SAMPLE_TEXT):
    print("  sentence:", s)

ipa = text_to_ipa_chunks(SAMPLE_TEXT)          # one IPA string per sentence
for i, chunk in enumerate(ipa):
    print(f"  [{i}] {chunk}")
print("  units of chunk 0:", phoneme_units(ipa[0]))


# --- 2. every character must be a real Kokoro token --------------------------
banner("2. VOCAB CHECK (original)")
check_ipa(ipa)


# --- 3. deterministic substitutions ------------------------------------------
# A single output = probability 1. Everything the old rule kinds did is written
# this way: aspiration is t -> tʰ, spirantization d -> ð, and so on.
# scope="outside_cluster" restricts a rule to singletons, so clusters like "nd"
# are left alone.
banner("3. DETERMINISTIC RULES")
rules_es = [
    {"map": {"b": "β", "v": "β", "d": "ð", "ɡ": "ɣ"}, "scope": "outside_cluster"},
    {"map": {"p": "pʰ", "t": "tʰ", "k": "kʰ"}},
]
print("rules:", describe_rules(rules_es))

out_es = manipulate(ipa, rules_es)
for before, after in zip(ipa, out_es):
    print(f"  before: {before}")
    print(f"  after : {after}")

print()
print_stats(substitution_stats(ipa, out_es, rules_es))


# --- 4. weighted substitutions ------------------------------------------------
# Several outputs with weights (percentages or fractions): one draw per
# occurrence. Listing the phoneme among its own choices gives it a chance to
# stay put. same_in_word=True makes every occurrence inside one word reuse the
# first draw, so a speaker stays self-consistent within a word; seed= makes the
# whole run reproducible.
banner("4. WEIGHTED (PROBABILISTIC) RULES")
rules_var = [
    {"map": {"e": {"e": 60, "ɛ": 40},                 # 60% / 40%
             "o": {"o": 0.7, "ɔ": 0.3},               # fractions work too
             "r": {"r": 50, "ɾ": 30, "ʁ": 20}},
     "same_in_word": True},
    {"map": {"a": {"ã": 50, "a": 50}}},               # a + combining tilde
    {"map": {"tʃ": "ʨ", "ʧ": "ʨ"}},                   # multi-char key: the affricate itself
]
print("rules:", describe_rules(rules_var))

out_var, masks = manipulate(ipa, rules_var, seed=7, return_mask=True)
for before, after in zip(ipa, out_var):
    print(f"  before: {before}")
    print(f"  after : {after}")

# Statistics are computed separately, by comparing the two chunk lists — so they
# report the NET effect, whatever the rules did on the way there.
banner("5. STATISTICS (before vs after)")
stats = substitution_stats(ipa, out_var, rules_var)
print_stats(stats)
print("\n  changed phoneme pairs in chunk 0:",
      [(o, n) for o, n in aligned_pairs(ipa[0], out_var[0]) if o != n])


# --- 6. the mask marks what changed ------------------------------------------
# One bool per character of the manipulated chunk: True = altered or inserted.
# synth_aligned(..., masks=masks) forwards it so the HTML visualiser can
# highlight the manipulated phonemes.
banner("6. ALTERATION MASK (chunk 0)")
print("  ipa :", out_var[0])
print("  mask:", "".join("^" if m else " " for m in masks[0]))


# --- 7. the manipulated string must still be tokenizable ---------------------
banner("7. VOCAB CHECK (manipulated)")
if check_ipa(out_var):
    print("\nReady for synthesis, e.g.:")
    print("    from functions import synth_aligned, audiovisualize_interactive")
    print("    audio, segs, words, altered = synth_aligned(out_var, voice='im_nicola', masks=masks)")
    print("    audiovisualize_interactive(audio, segs, seg_words=words, text=SAMPLE_TEXT,")
    print("                               seg_altered=altered, rules=rules_var, stats=stats)")
