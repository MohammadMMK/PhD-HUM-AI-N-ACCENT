"""
plot_conditions.py - figures from the CSVs written by make_conditions.py.

    python plot_conditions.py          (or run cell by cell in VS Code: the # %% markers)

Saves to output/figures/:
    1_substitution_pairs.png     how many times each substitution (t→tʲ ...) happens, per version
    2_changes_per_section.png    changes per section, split by native phoneme
    3_levenshtein.png            per-word normalized Levenshtein distance per section (plain and weighted)
    4_changes_per_word.png       how many phonemes are changed inside one word
    5_inconsistent_spread.png    inconsistent condition: changes per sentence along each section
    6_story_totals.png           whole story: total edits and Levenshtein distance per condition
"""
# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "output"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

changes = pd.read_csv(OUT / "changes.csv")
words = pd.read_csv(OUT / "words.csv")
sections = pd.read_csv(OUT / "sections.csv")

# categorical palette, fixed order (validated reference palette, light mode)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e7e6e2", "#ffffff"

VERSIONS = [v for v in sections["version"].unique() if v.startswith("consistent")] + ["inconsistent"]
V_LABEL = {v: v.replace("consistent_", "consistent ") for v in VERSIONS}
V_COL = dict(zip(VERSIONS, SLOTS))
SETTINGS = json.loads((OUT / "conditions.json").read_text(encoding="utf-8"))["settings"]
PHONEMES = list(SETTINGS["inconsistent"]) + sorted(set(changes["phoneme"]) - set(SETTINGS["inconsistent"]))
P_COL = dict(zip(PHONEMES, SLOTS))
STORIES = sorted(sections["story"].unique(), key=str)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def style(ax, grid="y"):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)
    getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    print("saved", FIG / name)
    plt.show()


# %% 1. substitution pairs per version ------------------------------------------------------------
pairs = (changes.groupby(["version", "phoneme", "pair"]).size().rename("n").reset_index())
order = (pairs[["phoneme", "pair"]].drop_duplicates()
         .assign(k=lambda d: d["phoneme"].map(PHONEMES.index)).sort_values(["k", "pair"])["pair"].tolist())
fig, axes = plt.subplots(1, len(VERSIONS), figsize=(3 * len(VERSIONS), 0.32 * len(order) + 1.2), sharey=True)
for ax, v in zip(axes, VERSIONS):
    d = pairs[pairs["version"] == v].set_index("pair").reindex(order)
    y = np.arange(len(order))
    bars = ax.barh(y, d["n"].fillna(0), 0.7, color=[P_COL.get(p, MUTED) for p in d["phoneme"].fillna("")],
                   edgecolor=SURFACE, linewidth=2)
    ax.bar_label(bars, labels=[f"{int(n)}" if n > 0 else "" for n in d["n"].fillna(0)],
                 padding=3, fontsize=8, color=MUTED)
    ax.set_yticks(y, order, color=INK)
    ax.invert_yaxis()
    ax.set_title(V_LABEL[v], loc="left", color=INK, fontsize=10)
    style(ax, "x")
    ax.margins(x=0.18)
axes[0].set_ylabel("native → replacement")
fig.suptitle("Substitutions, all sections together", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
save(fig, "1_substitution_pairs.png")

# %% 2. changes per section, split by native phoneme ----------------------------------------------
fig, axes = plt.subplots(1, len(STORIES), figsize=(5 * len(STORIES), 3.8), sharey=True, squeeze=False,
                         layout="constrained")
for ax, story in zip(axes[0], STORIES):
    s = sections[sections["story"] == story].copy()
    s["vk"] = s["version"].map(VERSIONS.index)
    s = s.sort_values(["part", "vk"])
    x = np.array([p * (len(VERSIONS) + 1) + k for p, k in zip(s["part"], s["vk"])], float)
    bottom = np.zeros(len(s))
    for p in PHONEMES:
        v = s[f"n_{p}"].to_numpy()
        ax.bar(x, v, 0.85, bottom=bottom, color=P_COL[p], edgecolor=SURFACE, linewidth=2, label=p)
        bottom += v
    ax.set_xticks(x, [V_LABEL[v].replace("consistent ", "").replace("inconsistent", "Inc") for v in s["version"]],
                  fontsize=7, color=INK)
    for part in s["part"].unique():
        xs = x[s["part"].to_numpy() == part]
        ax.text(xs.mean(), -0.13, f"section {part}", transform=ax.get_xaxis_transform(),
                ha="center", color=MUTED)
    ax.set_title(f"story {story}", loc="left", color=INK, fontsize=10)
    style(ax)
axes[0][0].set_ylabel("changed phonemes")
fig.legend(*axes[0][0].get_legend_handles_labels(), title="native phoneme", frameon=False, ncol=len(PHONEMES),
           loc="outside upper right", fontsize=8, title_fontsize=8)
fig.suptitle("Changes per section (R = consistent rule, Inc = inconsistent)", x=0.01, ha="left",
             color=INK, fontsize=11)
save(fig, "2_changes_per_section.png")

# %% 3. Levenshtein: per-word normalized distance per section --------------------------------------
metrics = [("pw", "every change = 1"), ("pw_w", "weighted (SUB_WEIGHTS)")]
fig, axes = plt.subplots(len(metrics), len(STORIES), figsize=(5 * len(STORIES), 3.1 * len(metrics)),
                         sharey=True, squeeze=False, layout="constrained")
w = 0.8 / len(VERSIONS)
for row, (m, mlabel) in enumerate(metrics):
    for ax, story in zip(axes[row], STORIES):
        s = sections[sections["story"] == story]
        parts = sorted(s["part"].unique())
        for k, v in enumerate(VERSIONS):
            vals = [s[(s["part"] == p) & (s["version"] == v)][m].mean() for p in parts]
            bars = ax.bar(np.arange(len(parts)) + (k - (len(VERSIONS) - 1) / 2) * w, vals, w * 0.92,
                          color=V_COL[v], label=V_LABEL[v])
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7, color=MUTED)
        ax.set_xticks(range(len(parts)), [f"section {p}" for p in parts], color=INK)
        ax.set_title(f"story {story} · {mlabel}", loc="left", color=INK, fontsize=10)
        style(ax)
    axes[row][0].set_ylabel("mean per-word distance, %")
fig.legend(*axes[0][0].get_legend_handles_labels(), frameon=False, ncol=len(VERSIONS),
           loc="outside upper right", fontsize=8)
fig.suptitle("Levenshtein distance to native: distance / phonemes of the word, mean over words and sentences",
             x=0.01, ha="left", color=INK, fontsize=11)
save(fig, "3_levenshtein.png")

# %% 4. how many phonemes are changed inside one word ---------------------------------------------
cap = 4
words["n"] = words["changes"].clip(upper=cap)
share = (words.groupby("version")["n"].value_counts(normalize=True).mul(100)
         .rename("pct").reset_index())
fig, ax = plt.subplots(figsize=(7.5, 3.4))
cats = list(range(cap + 1))
for k, v in enumerate(VERSIONS):
    d = share[share["version"] == v].set_index("n").reindex(cats)["pct"].fillna(0)
    bars = ax.bar(np.arange(len(cats)) + (k - (len(VERSIONS) - 1) / 2) * w, d, w * 0.92,
                  color=V_COL[v], label=V_LABEL[v])
    ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=7, color=MUTED)
ax.set_xticks(range(len(cats)), [str(c) if c < cap else f"{cap}+" for c in cats], color=INK)
ax.set_xlabel("changed phonemes in the word")
ax.set_ylabel("% of words")
ax.legend(frameon=False, ncol=len(VERSIONS), loc="lower right", bbox_to_anchor=(1, 1.02), fontsize=8)
ax.set_title("Changes per word, all sections", loc="left", color=INK, fontsize=11, pad=24)
style(ax)
fig.tight_layout()
save(fig, "4_changes_per_word.png")

# %% 5. inconsistent condition: changes per sentence along each section ---------------------------
inc = changes[changes["condition"] == "inconsistent"]
secs = sorted(sections["section"].unique())
fig, axes = plt.subplots(len(secs), 1, figsize=(10, 1.25 * len(secs) + 0.6), sharex=True, squeeze=False)
for ax, sec in zip(axes[:, 0], secs):
    d = inc[inc["section"] == sec].groupby(["sentence", "phoneme"]).size().unstack(fill_value=0)
    n_sent = words[words["section"] == sec]["sentence"].max() + 1
    d = d.reindex(range(n_sent), fill_value=0)
    bottom = np.zeros(n_sent)
    for p in PHONEMES:
        if p in d:
            ax.bar(d.index, d[p], 0.8, bottom=bottom, color=P_COL[p], edgecolor=SURFACE, linewidth=1, label=p)
            bottom += d[p].to_numpy()
    ax.set_ylabel(sec, rotation=0, ha="right", va="center", color=INK)
    style(ax)
axes[-1, 0].set_xlabel("sentence")
axes[0, 0].legend(title="native phoneme", frameon=False, ncol=len(PHONEMES), loc="lower right",
                  bbox_to_anchor=(1, 1.05), fontsize=8, title_fontsize=8)
fig.suptitle("Inconsistent condition: changes per sentence", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
save(fig, "5_inconsistent_spread.png")

# %% 6. whole story: total edits and Levenshtein distance per condition ---------------------------
#    consistent = one value per rule order over the sections (R2-R1-R3 ...): bar = mean, dots = each order
stories = pd.read_csv(OUT / "stories.csv")
metrics = [("changes", "total edits (= Levenshtein, raw sum)"), ("lev_w", "weighted Levenshtein, raw sum"),
           ("pw", "per-word normalized Levenshtein, %"), ("pw_w", "per-word normalized weighted, %")]
conds = ["consistent", "inconsistent"]
c_col = {"consistent": SLOTS[0], "inconsistent": V_COL["inconsistent"]}
fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6), squeeze=False, layout="constrained")
w = 0.8 / len(conds)
for ax, (m, mlabel) in zip(axes[0], metrics):
    for k, c in enumerate(conds):
        xs = np.arange(len(STORIES)) + (k - (len(conds) - 1) / 2) * w
        vals = [stories[(stories["story"] == s) & (stories["condition"] == c)][m] for s in STORIES]
        bars = ax.bar(xs, [v.mean() for v in vals], w * 0.92, color=c_col[c], label=c)
        ax.bar_label(bars, fmt="%.1f" if m.startswith("pw") else "%.0f", padding=2, fontsize=7, color=MUTED)
        for x, v in zip(xs, vals):
            if len(v) > 1:
                ax.scatter(np.full(len(v), x), v, s=9, color=INK, zorder=3, linewidths=0)
    ax.set_xticks(range(len(STORIES)), [f"story {s}" for s in STORIES], color=INK)
    ax.set_title(mlabel, loc="left", color=INK, fontsize=10)
    style(ax)
fig.legend(*axes[0][0].get_legend_handles_labels(), frameon=False, ncol=len(conds),
           loc="outside upper right", fontsize=8)
fig.suptitle("Whole story, all sections summed (consistent: bar = mean over rule orders, dots = each order)",
             x=0.01, ha="left", color=INK, fontsize=11)
save(fig, "6_story_totals.png")
print(stories.groupby(["story", "condition"])[["changes", "lev_w", "pw", "pw_w"]].agg(["mean", "min", "max"])
      .round(1).to_string())

# %% summary table (mean over sections) ------------------------------------------------------------
cols = ["changes", "words_changed", "pw", "pw_w", "lev_100"] + [f"n_{p}" for p in PHONEMES]
print(sections.groupby("version")[cols].mean().reindex(VERSIONS).round(1).to_string())
