"""voicing_neighbors.py - how many words in each story turn into ANOTHER existing Italian word
when /t/ -> /d/ and /k/ -> /g/ (voicing).

    conda activate koko
    python script/voicing_neighbors.py                 # default: candidate must have Zipf >= 3.0
    python script/voicing_neighbors.py --min-zipf 2.5  # looser definition of "existing word"

Spelling rules (Italian orthography, so only letters that really sound /t/ or /k/ are changed):
    t -> d          tt -> dd
    c -> g          only when it sounds /k/: before a, o, u, h, a consonant or at word end
    cc -> gg        same condition  (bocca -> bogga)
    ch -> gh        (chiave -> ghiave)
    q -> g          (quando -> guando),   cq -> gg  (acqua -> aggua)
    k -> g
Every non-empty SUBSET of the substitutable positions is tried (tanto -> danto, tando, dando),
so a word counts as "at risk" if at least one of these variants is an existing word.

"Existing word" = Zipf frequency in wordfreq's Italian list >= --min-zipf (wordfreq also contains
noise such as "gane" 1.66 or "tado" 1.79, so a threshold is needed; 3.0 = roughly once per million words).

Outputs (in output/):
    voicing_words.csv     one row per word token: story, part, word, n. variants tried, matches
    voicing_stories.csv   per story: tokens, types, at-risk tokens/types and percentages
"""
import argparse
import itertools
import re
from pathlib import Path

import pandas as pd
from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parent.parent  # text_analysis/
DATA = ROOT / "data"
OUT = ROOT / "output"

# Proper names of the stories (from data/proper_names.txt), lower case: never counted
PROPER_NAMES = {
    "sara", "adam", "lusana", "gubbana", "fanura", "malvara", "brua",
    "raul", "jamal", "ada", "parma", "val", "durna",
    "laura", "saul", "mara", "valmassa", "larga", "marsala",
}

# Words followed by an apostrophe are elided forms (c'è, l'isola, trent'anni): skipped
WORD_RE = re.compile(r"[a-zàáèéìíòóùú]+(?![a-zàáèéìíòóùú]*['’])")

# Variants that wordfreq lists but that are not Italian common words (names, dialect): never a match.
# Edit freely.
NOT_WORDS = {"ghe", "gui", "aldo", "alda", "garda"}
FRONT_VOWELS = set("eèéiìí")


def segments(word):
    """Split a word into (spelling, voiced_spelling_or_None) units.
    Units with a voiced alternative are the positions that can be substituted."""
    units, i, n = [], 0, len(word)
    while i < n:
        nxt = word[i + 1] if i + 1 < n else ""
        nxt2 = word[i + 2] if i + 2 < n else ""
        if word.startswith("tt", i):
            units.append(("tt", "dd")); i += 2
        elif word[i] == "t":
            units.append(("t", "d")); i += 1
        elif word.startswith("cch", i):
            units.append(("cch", "ggh")); i += 3
        elif word.startswith("ch", i):
            units.append(("ch", "gh")); i += 2
        elif word.startswith("cq", i):
            units.append(("cq", "gg")); i += 2
        elif word.startswith("cc", i):
            # /kk/ unless followed by e/i (then /ttʃ/, e.g. "accento")
            units.append(("cc", None if nxt2 in FRONT_VOWELS else "gg")); i += 2
        elif word[i] == "c":
            # /k/ unless followed by e/i (then /tʃ/, e.g. "cena", "ciao")
            units.append(("c", None if nxt in FRONT_VOWELS else "g")); i += 1
        elif word[i] == "q":
            units.append(("q", "g")); i += 1
        elif word[i] == "k":
            units.append(("k", "g")); i += 1
        else:
            units.append((word[i], None)); i += 1
    return units


def variants(word):
    """All spellings obtained by voicing any non-empty subset of the /t/ and /k/ positions."""
    units = segments(word)
    slots = [j for j, (_, v) in enumerate(units) if v is not None]
    out = set()
    for r in range(1, len(slots) + 1):
        for subset in itertools.combinations(slots, r):
            out.add("".join(v if j in subset else s for j, (s, v) in enumerate(units)))
    out.discard(word)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-zipf", type=float, default=3.0,
                    help="minimum Zipf frequency for a variant to count as an existing word")
    args = ap.parse_args()

    zipf_cache = {}
    def zipf(w):
        if w not in zipf_cache:
            zipf_cache[w] = zipf_frequency(w, "it")
        return zipf_cache[w]

    rows = []
    for path in sorted(DATA.glob("story_*_p*.txt")):
        story, part = map(int, re.findall(r"\d+", path.stem))
        text = path.read_text(encoding="utf-8").lower()
        for word in WORD_RE.findall(text):
            if word in PROPER_NAMES:
                continue
            vs = variants(word)
            matches = [v for v in vs if v not in NOT_WORDS and zipf(v) >= args.min_zipf]
            rows.append({
                "story": story,
                "part": part,
                "word": word,
                "zipf": zipf(word),
                "n_variants": len(vs),
                "at_risk": bool(matches),
                "matches": "; ".join(f"{m} ({zipf(m):.2f})" for m in matches),
            })

    words = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    words.to_csv(OUT / "voicing_words.csv", index=False, encoding="utf-8-sig")

    def summarise(g):
        types = g.drop_duplicates("word")
        return pd.Series({
            "tokens": len(g),
            "tokens_with_t_or_k": int((g.n_variants > 0).sum()),
            "at_risk_tokens": int(g.at_risk.sum()),
            "at_risk_tokens_%": round(100 * g.at_risk.mean(), 1),
            "types": len(types),
            "at_risk_types": int(types.at_risk.sum()),
            "at_risk_types_%": round(100 * types.at_risk.mean(), 1),
        })

    stories = words.groupby("story").apply(summarise, include_groups=False).reset_index()
    stories.to_csv(OUT / "voicing_stories.csv", index=False, encoding="utf-8-sig")

    print(f"Existing word = Zipf >= {args.min_zipf}\n")
    print(stories.to_string(index=False))
    for s, g in words[words.at_risk].groupby("story"):
        uniq = g.drop_duplicates("word")[["word", "matches"]]
        print(f"\nStory {s}: {len(uniq)} at-risk word types")
        for w, m in uniq.itertuples(index=False):
            print(f"  {w:<16} -> {m}")


if __name__ == "__main__":
    main()
