"""
rate_tools.py — speech rate from the exported player HTML files (no TTS model needed).

Every player file (accent_tools.export_condition / audiovisualize_interactive) stores each Kokoro token
with its kind, start and end time:
    phone      a phoneme                      diacritic  ʲ ʰ ː ...        stress  ˈ ˌ
    space      word gap                       punct      , . ? ...        bos/eos sentence start / end
A sentence is everything between a BOS and the next EOS.

    from rate_tools import *
    players = load_players("output_online/consistent/*_block*.html")
    rows    = sentence_table(players)
    flag_outliers(rows, measure="artic_rate")

MEASURES PER SENTENCE
    speech_rate  phonemes / s over the sentence (BOS / EOS silence excluded, pauses inside included)
    artic_rate   phonemes / s over the phonemes only (+ their diacritics and stress marks; pauses excluded)
    syll_rate    vowels (≈ syllables) / s over the sentence, like speech_rate
    wpm          words / minute over the sentence
    pause_s      time in spaces and punctuation inside the sentence
"""
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

__all__ = ["load_player", "load_players", "sentence_table", "flag_outliers", "rate_curve", "save_csv", "VOWELS"]

# vowels (one per syllable nucleus); capital letters are misaki's one-symbol diphthongs
VOWELS = set("aeiouy") | set("ɑɐɒæɔəɚɛɜɨɪɯøœʊʌɤ") | {"ᵻ", "ᵊ"} | set("AIOQWY")
ARTIC = {"phone", "diacritic", "stress"}          # tokens that count as articulation
PAUSE = {"space", "punct"}                        # tokens that count as pause inside a sentence


def load_player(path):
    """Read one player HTML -> {"path", "name", "title", "duration", "condition", "block", "segments"}.
    segments: list of {"l" label, "t" token, "k" kind, "s" start s, "e" end s, "ms", "fr", "w" word index,
              "wl" word label, "alt" altered by a rule}"""
    h = Path(path).read_text(encoding="utf-8")
    m = re.search(r"const segs=(\[.*?\]), env=", h, re.S)
    if not m:
        raise ValueError(f"{path}: no token data found (is this a player HTML?)")
    segs = json.loads(m.group(1))
    tm = re.search(r'const TITLE=(".*?"), RULES=', h, re.S)
    title = json.loads(tm.group(1)) if tm else ""
    dur = float(re.search(r"DUR=([0-9.]+)", h).group(1))
    name = Path(path).stem
    bm = re.search(r"_block(\d+)$", name)
    cond = title.split("·")[0].strip().lower() if "·" in title else name.split("_")[0]
    return dict(path=str(path), name=name, title=title, duration=dur, condition=cond,
                block=int(bm.group(1)) if bm else ("full" if name.endswith("_full") else None), segments=segs)


def load_players(pattern):
    """All player files matching a glob pattern (or a list of paths), sorted by name."""
    paths = sorted(glob.glob(pattern)) if isinstance(pattern, str) else list(pattern)
    if not paths:
        raise FileNotFoundError(f"no files match {pattern!r}")
    players = [load_player(p) for p in paths]
    print(f"{len(players)} file(s):")
    for p in players:
        n = sum(s["k"] == "bos" for s in p["segments"])
        print(f"   {p['name']:<40} {n:3d} sentences   {p['duration']:8.1f} s   {p['title']}")
    return players


def _sentences(segs):
    out, cur = [], None
    for s in segs:
        if s["k"] == "bos":
            cur = [s]
        elif cur is not None:
            cur.append(s)
            if s["k"] == "eos":
                out.append(cur); cur = None
    return out


def sentence_table(players):
    """One row per sentence of every player, with the rate measures (see the module docstring)."""
    rows = []
    for p in players:
        for i, toks in enumerate(_sentences(p["segments"])):
            inner = toks[1:-1]
            if not inner:
                continue
            start, end = inner[0]["s"], inner[-1]["e"]
            phones = [t for t in inner if t["k"] == "phone"]
            artic = sum(t["e"] - t["s"] for t in inner if t["k"] in ARTIC)
            pause = sum(t["e"] - t["s"] for t in inner if t["k"] in PAUSE)
            words, labels = [], []
            for t in inner:
                if t.get("w") is not None and (not words or words[-1] != t["w"]):
                    words.append(t["w"]); labels.append(t.get("wl") or "?")
            dur = end - start
            rows.append(dict(
                file=p["name"], condition=p["condition"], block=p["block"], sentence=i + 1,
                start=round(toks[0]["s"], 3), end=round(toks[-1]["e"], 3),
                n_phones=len(phones), n_syll=sum(t["t"] in VOWELS for t in phones), n_words=len(words),
                n_altered=sum(bool(t.get("alt")) for t in phones),
                dur_s=round(dur, 3), artic_s=round(artic, 3), pause_s=round(pause, 3),
                lead_s=round(toks[0]["e"] - toks[0]["s"], 3), trail_s=round(toks[-1]["e"] - toks[-1]["s"], 3),
                speech_rate=len(phones) / dur if dur else np.nan,
                artic_rate=len(phones) / artic if artic else np.nan,
                syll_rate=sum(t["t"] in VOWELS for t in phones) / dur if dur else np.nan,
                wpm=60 * len(words) / dur if dur else np.nan,
                mean_phone_ms=1000 * np.mean([t["e"] - t["s"] for t in phones]) if phones else np.nan,
                text=" ".join(labels)))
    return rows


def flag_outliers(rows, measure="artic_rate", against="n_phones", z=2.5, group_by="file", log_length=True):
    """Flag sentences whose rate is unusual FOR THEIR LENGTH.

    For every group (default: every file) the rate is regressed on sentence length
    (log(n_phones) by default), and the residuals are turned into robust z-scores
    (median / MAD, so the outliers themselves do not hide each other).
    Adds to every row:  fit (expected rate for that length), resid, z, outlier ("fast" / "slow" / "").
    Also adds z_raw: the robust z of the rate itself, ignoring length."""
    groups = {}
    for r in rows:
        groups.setdefault(r[group_by] if group_by else "all", []).append(r)
    for g, rs in groups.items():
        y = np.array([r[measure] for r in rs], float)
        x = np.array([r[against] for r in rs], float)
        x = np.log(np.maximum(x, 1)) if log_length else x
        ok = np.isfinite(y) & np.isfinite(x)
        if ok.sum() >= 3 and np.ptp(x[ok]) > 0:
            b, a = np.polyfit(x[ok], y[ok], 1)
        else:
            b, a = 0.0, np.nanmedian(y)
        fit = a + b * x
        res = y - fit
        mad = 1.4826 * np.nanmedian(np.abs(res - np.nanmedian(res))) or np.nanstd(res) or 1.0
        zz = (res - np.nanmedian(res)) / mad
        mad_raw = 1.4826 * np.nanmedian(np.abs(y - np.nanmedian(y))) or np.nanstd(y) or 1.0
        for r, f, e, zi, yi in zip(rs, fit, res, zz, y):
            r.update(fit=float(f), resid=float(e), z=float(zi), z_raw=float((yi - np.nanmedian(y)) / mad_raw),
                     outlier="fast" if zi > z else "slow" if zi < -z else "",
                     slope_per_log_length=float(b))
    n = sum(bool(r["outlier"]) for r in rows)
    print(f"{n} outlier sentence(s) out of {len(rows)}  ({measure} vs {'log ' if log_length else ''}{against}, |z| > {z}, per {group_by or 'all'})")
    return [r for r in rows if r["outlier"]]


def rate_curve(player, sigma=0.5, step=0.02, unit="phones"):
    """Continuous rate over time: every phoneme (or vowel, unit="syllables") is a point at its centre,
    smoothed with a Gaussian of width `sigma` seconds -> events per second at every `step`.
    Pauses show up as dips. Returns (t, rate)."""
    segs = player["segments"]
    if unit == "syllables":
        pts = np.array([(s["s"] + s["e"]) / 2 for s in segs if s["k"] == "phone" and s["t"] in VOWELS])
    else:
        pts = np.array([(s["s"] + s["e"]) / 2 for s in segs if s["k"] == "phone"])
    t = np.arange(0, player["duration"] + step, step)
    rate = np.zeros_like(t)
    for chunk in np.array_split(pts, max(1, len(pts) // 2000)):
        rate += np.exp(-0.5 * ((t[:, None] - chunk[None, :]) / sigma) ** 2).sum(1)
    return t, rate / (sigma * np.sqrt(2 * np.pi))


def save_csv(rows, path):
    """Write the sentence table (with outlier columns, if flagged) to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("written:", path)
