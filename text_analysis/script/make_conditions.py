"""
make_conditions.py - text -> native IPA, consistent IPA, inconsistent IPA (+ counts and Levenshtein).

    conda activate kokoro
    python make_conditions.py                              # every data/story_*_p*.txt
    python make_conditions.py ../data/story_1_p1.txt       # only these files

Each input .txt is one SECTION (story_1_p2 = story 1, section 2). For every section it writes

    output/ipa/<section>/text.txt                one sentence per line (spelling)
    output/ipa/<section>/native.txt              one IPA sentence per line  <- input for generation
    output/ipa/<section>/consistent_R1.txt       ... one file per consistent rule
    output/ipa/<section>/inconsistent.txt

and for all sections together

    output/conditions.json      everything above + every change, in one file
    output/changes.csv          one row per substituted phoneme
    output/words.csv            one row per word per version (Levenshtein, per-word distance)
    output/sections.csv         one row per section per version (counts, distances)
    output/inconsistent.csv     inconsistent condition: changes per phoneme and section (+ story target)
    output/stories.csv          whole-story totals (edits, Levenshtein) per condition; consistent = one
                                row per rule order over the sections

Give all parts of a story when passing files: the inconsistent condition is drawn per whole story.

Every section is written with EVERY consistent rule, because in the design the rule of a section is
counterbalanced across participants. Then run plot_conditions.py for the figures.
"""
import csv
import itertools
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import ipa_tools as T

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
OUT_DIR = HERE.parent / "output"

# ==================================================================================================
# SETTINGS
# ==================================================================================================
INPUT_FILES = sorted(DATA_DIR.glob("story_*_p*.txt"))

# Clean-up of the native IPA before anything else (espeak writes unstressed i as ɪ; without this the
# i-rule would miss them). {} to switch off.
NORMALIZE = {"ɪ": "i"}

# ---- CONSISTENT: one fixed substitution per phoneme; a section gets ONE rule -------------------
#      {rule name: {native phoneme: replacement}}. Every occurrence in the section is replaced.
CONSISTENT_RULES = {
    "R1": {"t": "tʲ", "k": "x",  "i": "y"},
    "R2": {"t": "θ",  "k": "ɡ",  "e": "ø"},     # ɡ is U+0261 (Kokoro's g), not the ASCII g
    "R3": {"t": "ɖ",  "k": "kʲ", "o": "ə"},
}

# ---- INCONSISTENT: random alternatives ----------------------------------------------------------
#      alts  : replacements drawn at random (balanced, see BALANCED_DRAWS)
#      share : 1.0  -> every occurrence is changed (like the consistent condition); each word draws one
#                      alternative, the same for every occurrence in it
#              1/3  -> decided per OCCURRENCE: every occurrence has the same 1/3 chance, whatever else is
#                      in the word; exactly 1/3 of the occurrences OF THE WHOLE STORY are changed. In the
#                      consistent condition each vowel is changed in 1 of the 3 sections, so 1/3 gives the
#                      same amount over a story, spread evenly instead of in one section.
#      Decided over all parts of a story together, then split back into sections.
INCONSISTENT = {
    "t": dict(alts=["tʲ", "θ", "ɖ"], share=1.0),
    "k": dict(alts=["x", "ɡ", "kʲ"], share=1.0),
    "i": dict(alts=["y"], share=1 / 3),
    "e": dict(alts=["ø"], share=1 / 3),
    "o": dict(alts=["ə"], share=1 / 3),
}
BALANCED_DRAWS = True                      # alternatives used equally often (True) or independent draws
SEED = 1                                   # same seed -> same inconsistent condition

# ---- Weighted Levenshtein (lev_w): cost of each substitution, native -> replacement ---------------
#      Shared diacritics are ignored when looking a pair up (tː -> tʲː uses t -> tʲ).
SUB_WEIGHTS = {
    ("t", "tʲ"): 0.5, ("t", "θ"): 2, ("t", "ɖ"): 1,
    ("k", "x"): 2,    ("k", "ɡ"): 1, ("k", "kʲ"): 0.5,
    ("e", "ø"): 1,    ("i", "y"): 1, ("o", "ə"): 1,
}
DEFAULT_WEIGHT = 1.0                       # any pair not listed above
# ==================================================================================================


def check_settings():
    new = [n for r in CONSISTENT_RULES.values() for n in r.values()] + \
          [a for s in INCONSISTENT.values() for a in s["alts"]]
    bad = T.not_in_vocab(new)
    assert not bad, f"not in Kokoro's vocabulary (would be dropped from the speech): {bad}"
    for old in [o for r in CONSISTENT_RULES.values() for o in r] + list(INCONSISTENT):
        assert len(old) == 1, f"target phoneme must be one symbol, got {old!r}"
    # warn when the two conditions do not use the same substitutions
    cons = {(o, n) for r in CONSISTENT_RULES.values() for o, n in r.items()}
    inc = {(o, a) for o, s in INCONSISTENT.items() for a in s["alts"]}
    if cons != inc:
        print("NOTE: consistent and inconsistent use different substitutions:"
              f"\n   only consistent  : {sorted(cons - inc)}\n   only inconsistent: {sorted(inc - cons)}")


def section_id(path):
    m = re.search(r"story_(\d+)_p(\d+)", path.stem)
    return (f"story{m[1]}_p{m[2]}", int(m[1]), int(m[2])) if m else (path.stem, path.stem, 1)


def build_section(path):
    """Native + every consistent rule for one section (the inconsistent version is added per story)."""
    sid, story, part = section_id(path)
    text = T.sentences(path.read_text(encoding="utf-8"))
    native = [T.normalize(T.to_ipa(s), NORMALIZE) for s in text]
    versions = {"native": dict(condition="native", rule="", ipa=native, changes=[])}
    for name, rule in CONSISTENT_RULES.items():
        ipa, ch = T.consistent(native, rule)
        versions[f"consistent_{name}"] = dict(condition="consistent", rule=name, ipa=ipa, changes=ch)
    return dict(section=sid, story=story, part=part, source=str(path), text=text, native=native,
                versions=versions)


def add_inconsistent(secs):
    """Inconsistent condition over all sections of ONE story pooled, then split back per section.
    Returns the story-level report {phoneme: dict(target, changed, words, occurrences, ...)}."""
    pooled = [s for sec in secs for s in sec["native"]]
    rng = random.Random(f"{SEED}|story{secs[0]['story']}")
    ipa, ch, report = T.inconsistent(pooled, INCONSISTENT, BALANCED_DRAWS, rng)
    start = 0
    for sec in secs:
        n = len(sec["native"])
        sec_ch = [dict(c, sentence=c["sentence"] - start) for c in ch if start <= c["sentence"] < start + n]
        sec["versions"]["inconsistent"] = dict(condition="inconsistent", rule="", ipa=ipa[start:start + n],
                                               changes=sec_ch)
        start += n
    return report


def story_totals(sec_rows, word_rows):
    """Whole-story totals per condition. Consistent: every section has a different rule, so one row per
    rule order (e.g. R2-R1-R3 = section 1 R2, section 2 R1, section 3 R3), as counterbalanced in the design.
    changes / lev / lev_w are raw sums; pw / pw_w are normalized like in sections.csv
    (per word lev / phonemes, mean over the words of a sentence, then over all sentences of the story)."""
    sum_cols = ["sentences", "words", "phonemes", "words_changed", "changes", "lev", "lev_w"] + \
               [k for k in sec_rows[0] if k.startswith("n_")]
    by = {(r["story"], r["part"], r["version"]): r for r in sec_rows}
    # per-sentence mean of d and d_w, per (story, part, version)
    sent = {}
    for r in word_rows:
        sent.setdefault((r["story"], r["part"], r["version"]), {}).setdefault(r["sentence"], []).append(
            (r["lev"] / r["phonemes"], r["lev_w"] / r["phonemes"]))
    sent_mean = {k: [np.mean(v, axis=0) for v in s.values()] for k, s in sent.items()}
    rows = []
    for story in sorted({r["story"] for r in sec_rows}, key=str):
        parts = sorted({r["part"] for r in sec_rows if r["story"] == story})
        combos = [("inconsistent", "", ["inconsistent"] * len(parts))]
        if len(parts) <= len(CONSISTENT_RULES):
            combos += [("consistent", "-".join(o), [f"consistent_{r}" for r in o])
                       for o in itertools.permutations(CONSISTENT_RULES, len(parts))]
        for cond, order, versions in combos:
            srcs = [by[(story, p, v)] for p, v in zip(parts, versions)]
            tot = {c: sum(s[c] for s in srcs) for c in sum_cols}
            sm = np.array([m for p, v in zip(parts, versions) for m in sent_mean[(story, p, v)]])
            rows.append(dict(story=story, condition=cond, rule_order=order, sections=len(parts), **tot,
                             pw=round(100 * float(sm[:, 0].mean()), 3), pw_w=round(100 * float(sm[:, 1].mean()), 3),
                             lev_100=round(100 * tot["lev"] / tot["phonemes"], 3),
                             lev_w_100=round(100 * tot["lev_w"] / tot["phonemes"], 3)))
    return rows


def measure(sec, cost_w):
    """Per-word Levenshtein of every version against native.
    d = distance / phonemes of the native word; pw = mean over words in a sentence, then over sentences."""
    word_rows, sec_rows = [], []
    targets = sorted({o for r in CONSISTENT_RULES.values() for o in r} | set(INCONSISTENT))
    for vname, v in sec["versions"].items():
        if vname == "native":
            continue
        tot = Counter()
        sent_d, sent_dw = [], []
        n_change = Counter((c["sentence"], c["word"]) for c in v["changes"])
        for si, (nat_s, man_s) in enumerate(zip(sec["native"], v["ipa"])):
            d_s, dw_s = [], []
            for wi, (x, y) in enumerate(zip(T.words(nat_s), T.words(man_s))):
                ux, uy = T.units(x), T.units(y)
                if not ux:
                    continue
                d, dw = T.lev(ux, uy), T.lev(ux, uy, cost_w)
                d_s.append(d / len(ux)); dw_s.append(dw / len(ux))
                tot.update(words=1, phonemes=len(ux), lev=d, lev_w=dw, words_changed=int(d > 0))
                word_rows.append(dict(section=sec["section"], story=sec["story"], part=sec["part"],
                                      version=vname, condition=v["condition"], rule=v["rule"],
                                      sentence=si, word=wi, native=x, manipulated=y, phonemes=len(ux),
                                      changes=n_change[(si, wi)], lev=d, lev_w=dw,
                                      d=round(d / len(ux), 4), d_w=round(dw / len(ux), 4)))
            if d_s:
                sent_d.append(np.mean(d_s)); sent_dw.append(np.mean(dw_s))
        per_old = Counter(T.base(c["old"]) for c in v["changes"])
        sec_rows.append(dict(section=sec["section"], story=sec["story"], part=sec["part"], version=vname,
                             condition=v["condition"], rule=v["rule"], sentences=len(sec["native"]),
                             words=tot["words"], phonemes=tot["phonemes"], words_changed=tot["words_changed"],
                             changes=len(v["changes"]), lev=tot["lev"], lev_w=tot["lev_w"],
                             pw=round(100 * float(np.mean(sent_d)), 3), pw_w=round(100 * float(np.mean(sent_dw)), 3),
                             lev_100=round(100 * tot["lev"] / tot["phonemes"], 3),
                             lev_w_100=round(100 * tot["lev_w"] / tot["phonemes"], 3),
                             **{f"n_{o}": per_old[o] for o in targets}))
    return word_rows, sec_rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main(files):
    check_settings()
    cost_w = T.weighted_cost(SUB_WEIGHTS, DEFAULT_WEIGHT)
    (OUT_DIR / "ipa").mkdir(parents=True, exist_ok=True)
    sections, change_rows, word_rows, sec_rows, inc_rows = [], [], [], [], []

    for path in files:
        sections.append(build_section(Path(path)))
    sections.sort(key=lambda s: (str(s["story"]), s["part"]))

    # inconsistent condition: one draw per story, all its parts pooled
    story_report = {}
    for story, secs in itertools.groupby(sections, key=lambda s: s["story"]):
        story_report[story] = add_inconsistent(list(secs))

    for sec in sections:
        d = OUT_DIR / "ipa" / sec["section"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "text.txt").write_text("\n".join(sec["text"]) + "\n", encoding="utf-8")
        for vname, v in sec["versions"].items():
            (d / f"{vname}.txt").write_text("\n".join(v["ipa"]) + "\n", encoding="utf-8")
            for c in v["changes"]:
                old, new = T.pair_key(c["old"], c["new"])
                change_rows.append(dict(section=sec["section"], story=sec["story"], part=sec["part"],
                                        version=vname, condition=v["condition"], rule=v["rule"],
                                        sentence=c["sentence"], word=c["word"], native_word=c["native_word"],
                                        manipulated_word=c["word_ipa"], old=c["old"], new=c["new"],
                                        phoneme=old, replacement=new, pair=f"{old}→{new}"))
        occ = Counter(T.base(u) for s in sec["native"] for w in T.words(s) for u in T.units(w))
        chg = Counter(T.base(c["old"]) for c in sec["versions"]["inconsistent"]["changes"])
        for old, r in story_report[sec["story"]].items():
            inc_rows.append(dict(section=sec["section"], story=sec["story"], part=sec["part"], phoneme=old,
                                 share=round(INCONSISTENT[old]["share"], 4), occurrences=occ[old],
                                 changed=chg[old], pct=round(100 * chg[old] / occ[old], 1) if occ[old] else 0,
                                 story_occurrences=r["occurrences"], story_target=r["target"],
                                 story_changed=r["changed"]))
        w, s = measure(sec, cost_w)
        word_rows += w; sec_rows += s

        print(f"{sec['section']}: {len(sec['text'])} sentences | "
              + " | ".join(f"{vn.replace('consistent_', '')} {len(v['changes'])}"
                           for vn, v in sec["versions"].items() if vn != "native") + " changes")

    for story, rep in story_report.items():
        print(f"story {story} inconsistent (whole story, changed/target): " + ", ".join(
            f"{o} {r['changed']}/{r['target']}" + ("" if r["changed"] == r["target"] else " (!)")
            for o, r in rep.items()))
    story_rows = story_totals(sec_rows, word_rows)

    settings = dict(normalize=NORMALIZE, consistent_rules=CONSISTENT_RULES, inconsistent=INCONSISTENT,
                    balanced_draws=BALANCED_DRAWS, seed=SEED,
                    sub_weights={f"{a}→{b}": w for (a, b), w in SUB_WEIGHTS.items()}, default_weight=DEFAULT_WEIGHT)
    (OUT_DIR / "conditions.json").write_text(
        json.dumps(dict(settings=settings, sections=sections), ensure_ascii=False, indent=1), encoding="utf-8")
    write_csv(OUT_DIR / "changes.csv", change_rows)
    write_csv(OUT_DIR / "words.csv", word_rows)
    write_csv(OUT_DIR / "sections.csv", sec_rows)
    write_csv(OUT_DIR / "inconsistent.csv", inc_rows)
    write_csv(OUT_DIR / "stories.csv", story_rows)
    print(f"\nWritten to {OUT_DIR}:  ipa/<section>/*.txt, conditions.json, changes.csv, words.csv, "
          "sections.csv, inconsistent.csv, stories.csv")


if __name__ == "__main__":
    main(sys.argv[1:] or INPUT_FILES)
