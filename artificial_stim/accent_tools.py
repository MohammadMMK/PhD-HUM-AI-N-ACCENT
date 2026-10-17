"""
accent_tools.py — everything the accent notebooks need, in one place.

Put this file next to functions.py (model, G2P, manipulate(), player template). Then in a notebook:

    from accent_tools import *

SECTIONS
  1. Stories and IPA        load_story(), normalize_ipa(), spell_words()
  2. Manipulation           manipulate_with_quota()  (= functions.manipulate() + "word_quota" rules)
  3. Conditions             make_inconsistent(), make_consistent(), make_native()
  4. Synthesis              render_condition()  (one voice per block, optional reference voice for timing)
                            synth_accent()      (the engine underneath: duration / pitch / energy transfer)
  5. Export                 export_condition()  (player HTML + wav per block and for the whole condition;
                                                 written to disk, never displayed in the notebook)
  6. Measurement            measure_distance(), print_summary()  (per-word normalized Levenshtein distance)

TYPICAL USE
    story = load_story("story1.txt")                              # blocks separated by a line with ###
    cond  = make_inconsistent(story, consonants={...}, vowels={...}, seed=1)
    print_summary(cond, weights=WEIGHTS)
    res   = render_condition(cond, voices=[...], ref_voice="if_sara")
    export_condition(cond, res, "output_online/inconsistent")

All objects are plain dicts, so you can look inside them:
    story = {"name", "path", "blocks": [ {"index", "text", "sentences", "ipa", "spelling"} ]}
    cond  = {"name", "story", "blocks": [ {"index", "ipa", "manip", "masks", "origins", "rules", "stats"} ], ...}
    res   = [ {"block", "voice", "ref_voice", "audio", "segments", "seg_words", "seg_altered", "pros"} ]
"""
import csv
import hashlib
import json
import os
import random
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import functions as F
from functions import (model, text_to_ipa_chunks, one_sentence_chunks, check_ipa, print_stats, describe_rules,
                       align_phonemes, audiovisualize_interactive, transfer_durations, VOCAB, VOWELS, BOUNDARY,
                       SR, FRAME_SAMPLES, PRESETS)
from functions import _apply_rule, _scan, _build_stats, _norm, _norm_choices, _encode, _voice_pack, _durations_cached

__all__ = [
    # 1. stories / IPA
    "load_story", "normalize_ipa", "spell_words",
    # 2. manipulation
    "manipulate_with_quota", "select_quota_words", "adjust_phonemes",
    # 3. conditions
    "make_inconsistent", "make_consistent", "make_native",
    # 4. synthesis
    "render_condition", "synth_accent",
    # 5. export
    "export_condition", "export_player", "merge_results",
    # 6. measurement
    "measure_distance", "distance_table", "print_summary", "block_changes", "DEFAULT_WEIGHTS",
    # handy re-exports from functions.py
    "SR", "VOCAB", "check_ipa", "print_stats", "describe_rules", "text_to_ipa_chunks",
]


# =====================================================================================================
# 1. STORIES AND IPA
# =====================================================================================================
NORMALIZE_DEFAULT = {"ɪ": "i"}


def normalize_ipa(ipa, mapping=None, verbose=True):
    """Replace symbols in the original IPA (default: every ɪ -> i). Returns the same type (str or list).
    Do this right after text_to_ipa_chunks(), so every version of the text starts from the same IPA."""
    mapping = NORMALIZE_DEFAULT if mapping is None else mapping
    single = isinstance(ipa, str)
    chunks = [ipa] if single else list(ipa)
    keys = sorted(mapping, key=len, reverse=True)
    counts, out = Counter(), []
    for s in chunks:
        for k in keys:
            counts[k] += s.count(k)
            s = s.replace(k, mapping[k])
        out.append(s)
    if verbose:
        print("normalize_ipa: " + (", ".join(f"{k}→{mapping[k]} ×{counts[k]}" for k in keys) or "nothing to do"))
    return out[0] if single else out


def _word_spans(s):
    """[(start, end)] of every word in s (maximal runs without boundary chars)."""
    spans, start = [], None
    for i, c in enumerate(s):
        if c in BOUNDARY:
            if start is not None:
                spans.append((start, i)); start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(s)))
    return spans


def spell_words(sentences, ipa_chunks, g2p_fn=None):
    """Spelling for every IPA word: returns (spell, unmatched) with spell[ci][wi] = spelling of word wi
    of chunk ci ("?" if it could not be matched) and unmatched = {ci: number of "?" words}.

    1) fast path: if the sentence has as many letter-only words as IPA words, pair them by position
    2) otherwise every whitespace token is converted to IPA on its own (g2p_fn, default the Italian G2P
       from functions.py) and the resulting word sequence is aligned to the sentence's IPA words
       (stress marks ignored). Tokens like dell'acqua / c'è that become one or two IPA words are handled;
       only words the alignment cannot place get "?"."""
    import re, difflib
    if g2p_fn is None:
        g2p_fn = lambda s: F.g2p.g2p(s)[0]
    strip = lambda w: w.replace("ˈ", "").replace("ˌ", "").replace("ɪ", "i")    # ignore stress and the ɪ clean-up
    spell, unmatched = [], {}
    for ci, (sent, ps) in enumerate(zip(sentences, ipa_chunks)):
        ipa_words = [ps[a:b] for a, b in _word_spans(ps)]
        letters = re.findall(r"[^\W\d_]+", sent)
        if len(letters) == len(ipa_words):
            spell.append(letters); continue
        pred, labels = [], []                          # predicted IPA words + their spelling
        for tok in sent.split():
            ip = g2p_fn(tok) or ""
            ws = [ip[a:b] for a, b in _word_spans(ip)]
            parts = re.findall(r"[^\W\d_]+", tok)
            clean = re.sub(r"^\W+|\W+$", "", tok) or tok
            for k, w in enumerate(ws):
                pred.append(strip(w))
                labels.append(parts[k] if len(parts) == len(ws) else clean)
        out = ["?"] * len(ipa_words)
        sm = difflib.SequenceMatcher(None, pred, [strip(w) for w in ipa_words], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):   # same count -> pair 1:1
                for d in range(j2 - j1):
                    out[j1 + d] = labels[i1 + d]
        spell.append(out)
        if "?" in out:
            unmatched[ci] = out.count("?")
    return spell, unmatched


def load_story(source, name=None, block_sep="###", normalize=None, n_blocks=None, verbose=True):
    """Read a story and convert every block to IPA.

    source    : a file path ("story1.txt") or the text itself
    name      : a short name used in titles / file names (default: the file name without extension)
    block_sep : every line that STARTS with this ends a block (default ###; "### block 2" works too)
    normalize : clean-up of the IPA before anything else (default {"ɪ": "i"}; {} to switch off)
    n_blocks  : only used when the text has no block_sep: cut it into n_blocks blocks of similar word
                count at sentence boundaries (for quick tests; for the experiment, mark the blocks yourself)
    Returns {"name", "path", "blocks": [{"index", "text", "sentences", "ipa", "spelling"}]}.
    ipa[s] is the IPA of sentence s; spelling[s][w] the spelled form of IPA word w ("?" if unknown)."""
    normalize = NORMALIZE_DEFAULT if normalize is None else normalize
    path = Path(source) if len(str(source)) < 260 and Path(str(source)).exists() else None
    text = path.read_text(encoding="utf-8") if path else str(source)
    name = name or (path.stem if path else "story")
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    # a block ends at every line that STARTS with block_sep ("###", "### block 2", ...)
    parts = [p.strip() for p in re.split(rf"^[ \t]*{re.escape(block_sep)}.*$", text, flags=re.M) if p.strip()]
    if len(parts) == 1 and block_sep in text and verbose:
        print(f"⚠ {name}: '{block_sep}' appears in the text but never at the START of a line, so it was not used "
              f"as a block separator. Put it on its own line between the blocks.")
    if len(parts) == 1 and n_blocks and n_blocks > 1:
        sents = one_sentence_chunks(text)
        nw = np.cumsum([len(s.split()) for s in sents])
        cuts = [int(np.argmin(np.abs(nw - nw[-1] * b / n_blocks))) + 1 for b in range(1, n_blocks)]
        bounds = [0] + cuts + [len(sents)]
        parts = [" ".join(sents[a:b]) for a, b in zip(bounds, bounds[1:])]
        if verbose:
            print(f"{name}: no '{block_sep}' found, cut automatically into {n_blocks} blocks")
    blocks, n_norm = [], 0
    for b, part in enumerate(parts, start=1):
        sents = one_sentence_chunks(part)
        raw = text_to_ipa_chunks(part)
        n_norm += sum(s.count(k) for s in raw for k in normalize)
        ipa = normalize_ipa(raw, normalize, verbose=False)
        spelling, _ = spell_words(sents, ipa)
        blocks.append(dict(index=b, text=part, sentences=sents, ipa=ipa, spelling=spelling))
    if verbose and len(blocks) == 1 and not n_blocks:
        print(f"⚠ {name}: only ONE block found (no line starting with '{block_sep}'). "
              f"If the story has several blocks, put a line '{block_sep}' between them.")
    if verbose:
        print(f"{name}: {len(blocks)} block(s), " + ", ".join(
            f"block {b['index']}: {len(b['sentences'])} sentences / {sum(len(_word_spans(s)) for s in b['ipa'])} words"
            for b in blocks) + (f"   (IPA clean-up: {n_norm} × {', '.join(f'{k}→{v}' for k, v in normalize.items())})"
                                if normalize else ""))
    return dict(name=name, path=str(path) if path else None, blocks=blocks)


# =====================================================================================================
# 2. MANIPULATION: functions.manipulate() + "word_quota" rules
# =====================================================================================================
# A "word_quota" rule changes EXACTLY N words per vowel over the whole text, at most ONE vowel per word:
#   {"kind": "word_quota",
#    "targets": [{"old": "e", "new": "ø", "words": 40, "phonemes": 50}, ...],
#    "batch": 10, "order": "rotate", "occurrences": "all", "skip_altered_words": False, "seed": 7}
# words      : exact number of words (a float < 1 = fraction of the candidates)
# phonemes   : optional; "all" -> reached by swapping words (every e of a chosen word changes);
#              "partial" -> exactly that many changes spread over the chosen words; "one" -> = words
# Selection  : random rounds (batch words per vowel per round, starting vowel rotates) + a repair step
#              that moves words between vowels when one runs out of candidates.

def select_quota_words(cand, quota, batch=10, order="rotate", rng=None):
    """cand  : {target: list of word keys that can take this target}
    quota : {target: number of words wanted}
    Returns ({word: target}, {target: shortfall}).

    1) Rounds: in each round every target that still needs words draws up to `batch` words at random
       from its candidates that nobody owns yet. order="rotate": the target order of the first round is
       random, every next round starts one target later; order="shuffle": new random order per round.
    2) Repair: if a target ran out of free candidates, augmenting paths move words between targets
       (o takes a word from e if e can take another free word instead, possibly via a chain).
       This reaches every quota whenever ANY one-change-per-word assignment exists; otherwise the
       shortfall is returned."""
    rng = rng or random.Random()
    targets = list(quota)
    cand = {t: sorted(set(cand[t])) for t in targets}
    owner, need = {}, dict(quota)

    # ---- 1) random rounds ----
    seq = targets[:]
    rng.shuffle(seq)
    while True:
        active = [t for t in seq if need[t] > 0 and any(w not in owner for w in cand[t])]
        if not active:
            break
        for t in active:
            free = [w for w in cand[t] if w not in owner]
            for w in rng.sample(free, min(batch, need[t], len(free))):
                owner[w] = t
                need[t] -= 1
        if order == "rotate":
            seq = seq[1:] + seq[:1]
        elif order == "shuffle":
            rng.shuffle(seq)
        else:
            raise ValueError(f"order must be 'rotate' or 'shuffle', not {order!r}")

    # ---- 2) repair with augmenting paths (BFS over targets) ----
    def augment(r):
        parent, queue, seen = {}, [r], {r}
        while queue:
            u = queue.pop(0)
            ws = cand[u][:]
            rng.shuffle(ws)
            for w in ws:
                if w not in owner:                     # free word: u takes it, walk the chain back to r
                    owner[w] = u
                    while u != r:
                        pu, pw = parent[u]             # pu takes pw away from u (u just got w instead)
                        owner[pw] = pu
                        u = pu
                    return True
                s = owner[w]
                if s not in seen:
                    seen.add(s); parent[s] = (u, w); queue.append(s)
        return False

    for t in targets:
        while need[t] > 0 and augment(t):
            need[t] -= 1
    return owner, {t: n for t, n in need.items() if n > 0}


def adjust_phonemes(owner, cand, count, ptarget, rng=None, protect=None):
    """Bring the number of phonemes of each target close to ptarget[t] WITHOUT changing its word count:
    repeatedly swap one of its words for a free candidate word (owned by nobody) with one more / one
    fewer occurrence (the smallest step that helps, random words within the step). If no free word helps,
    take a word from another target and give that target a replacement word (with the same number of
    its own phonemes if it is in `protect` (default: every target in ptarget), so its counts do not change).
    count[t][w] = occurrences of target t in word w. Changes `owner` in place, returns {t: phonemes}."""
    rng = rng or random.Random()
    protect = set(ptarget) if protect is None else set(protect) | set(ptarget)
    got = {}
    for t, P in ptarget.items():
        cur = sum(count[t][w] for w, o in owner.items() if o == t)
        while cur != P:
            sign = 1 if P > cur else -1
            mine, free = defaultdict(list), defaultdict(list)
            for w, o in owner.items():
                if o == t:
                    mine[count[t][w]].append(w)
            for w in cand[t]:
                if w not in owner:
                    free[count[t][w]].append(w)
            helps = lambda d: d * sign > 0 and abs(P - cur - d) < abs(P - cur)
            steps = sorted({cn - co for co in mine for cn in free if helps(cn - co)}, key=abs)
            if steps:                                   # 1) swap with a free word
                d = steps[0]
                pairs = [(co, co + d) for co in sorted(mine) if co + d in free]
                co, cn = rng.choice(pairs)
                w_out, w_in = rng.choice(sorted(mine[co])), rng.choice(sorted(free[cn]))
                del owner[w_out]; owner[w_in] = t
                cur += d
                continue
            # 2) take a word from another target s, and give s a replacement word (possibly the word t
            #    gives up) with the same number of s's phonemes when s has a phoneme target
            opts = [(abs(count[t][w] - co), w, co) for w in cand[t] if owner.get(w, t) != t
                    for co in mine if helps(count[t][w] - co)]
            rng.shuffle(opts); opts.sort(key=lambda x: x[0])
            for _, w_in, co in opts:
                s_ = owner[w_in]
                w_out = rng.choice(sorted(mine[co]))
                pool = [w for w in cand[s_] if (w not in owner or w == w_out) and w != w_in
                        and (s_ not in protect or count[s_][w] == count[s_][w_in])]
                if pool:
                    w_rep = rng.choice(sorted(pool))
                    del owner[w_out]; owner[w_in] = t; owner[w_rep] = s_
                    cur += count[t][w_in] - co
                    break
            else:
                break                                   # closest reachable
        got[t] = cur
    return got


def _targets(rule):
    out = []
    for k, t in enumerate(rule["targets"]):
        olds = t["old"] if isinstance(t["old"], (list, tuple)) else [t["old"]]
        olds = [_norm(o) for o in olds]
        if "choices" in t:
            outs, ws = _norm_choices(t["choices"])
        else:
            outs, ws = [_norm(t["new"])], [1.0]
        n = t["n"] if "n" in t else t["words"]
        out.append(dict(i=k, olds=olds, outs=outs, ws=ws, n=n, phonemes=t.get("phonemes")))
    keys = [o for t in out for o in t["olds"]]
    assert len(keys) == len(set(keys)), f"word_quota: a phoneme appears in two targets: {keys}"
    return out


def _apply_quota(chunks, masks, origins, rule, protect_affricates, rng, counter, verbose=True):
    tg = _targets(rule)
    key2t = {o: t["i"] for t in tg for o in t["olds"]}
    occ_mode = rule.get("occurrences", "all")
    assert occ_mode in ("all", "one", "partial"), occ_mode
    skip_alt = rule.get("skip_altered_words", False)

    # ---- find every matchable occurrence (same longest-match / affricate logic as the other rules) ----
    occ = {}                                            # (chunk, word start) -> {target: [(pos, key)]}
    words = {}                                          # (chunk, word start) -> IPA word string
    for ci, s in enumerate(chunks):
        spans = _word_spans(s)
        starts = [a for a, _ in spans]
        found = []
        _scan(s, key2t.keys(), lambda k, st, i: (found.append((i, k)), k)[1],
              "all", protect_affricates, masks[ci], pass_pos=True, in_origin=origins[ci])
        for wi, (a, b) in enumerate(spans):
            words[(ci, a)] = (s[a:b], wi)
        for i, k in found:
            if any(masks[ci][i:i + len(k)]):            # already changed by an earlier rule
                continue
            a = max(x for x in starts if x <= i)
            occ.setdefault((ci, a), {}).setdefault(key2t[k], []).append((i, k))
    if skip_alt:
        altered = {(ci, a) for ci, s in enumerate(chunks) for a, b in _word_spans(s) if any(masks[ci][a:b])}
        occ = {w: v for w, v in occ.items() if w not in altered}

    cand = {t["i"]: [w for w, v in occ.items() if t["i"] in v] for t in tg}
    quota = {}
    for t in tg:
        n = t["n"]
        quota[t["i"]] = int(round(n * len(cand[t["i"]]))) if isinstance(n, float) and n < 1 else int(n)
    owner, short = select_quota_words(cand, quota, rule.get("batch", 10), rule.get("order", "rotate"), rng)

    # ---- optional phoneme targets ----
    count = {t["i"]: {w: len(occ[w][t["i"]]) for w in cand[t["i"]]} for t in tg}
    ptarget, notes = {}, {}
    for t in tg:
        P, ti = t["phonemes"], t["i"]
        if P is None:
            continue
        total = sum(count[ti].values())
        P = int(round(P * total)) if isinstance(P, float) and P < 1 else int(P)
        nw = sum(o == ti for o in owner.values())
        if occ_mode == "one":
            notes[ti] = "occurrences='one': phonemes = words, phoneme target ignored"; continue
        if P < nw:
            notes[ti] = f"phoneme target {P} < {nw} words (every chosen word needs >= 1 change): using {nw}"
            P = nw
        ptarget[ti] = P
    if ptarget:
        if occ_mode == "all":
            adjust_phonemes(owner, cand, count, ptarget, rng)
        else:                                           # partial: only make sure there are ENOUGH occurrences
            cap = {ti: sum(count[ti][w] for w, o in owner.items() if o == ti) for ti in ptarget}
            adjust_phonemes(owner, cand, count, {ti: P for ti, P in ptarget.items() if cap[ti] < P}, rng,
                            protect=ptarget)
    n_extra = {}                                        # partial: extra changes beyond 1 per word, per word
    if occ_mode == "partial":
        for ti, P in ptarget.items():
            mine = sorted(w for w, o in owner.items() if o == ti)
            slots = [w for w in mine for _ in range(count[ti][w] - 1)]
            extra = min(max(P - len(mine), 0), len(slots))
            n_extra.update(Counter(rng.sample(slots, extra)))

    # ---- what to write where ----
    plan = [dict() for _ in chunks]                     # chunk -> {pos: piece}
    chosen = {t["i"]: [] for t in tg}
    for w in sorted(owner):
        ti = owner[w]; t = tg[ti]
        piece = rng.choices(t["outs"], weights=t["ws"], k=1)[0]
        hits = occ[w][ti]
        if occ_mode == "one" or (occ_mode == "partial" and ti in ptarget):
            hits = sorted(rng.sample(hits, 1 + n_extra.get(w, 0)))
        for i, k in hits:
            plan[w[0]][i] = piece
        chosen[ti].append(dict(chunk=w[0], index=words[w][1], word=words[w][0], new=piece,
                               n_phonemes=len(hits)))

    out_c, out_m, out_o = [], [], []
    for ci, s in enumerate(chunks):
        p = plan[ci]
        def act(k, st, i, p=p):
            piece = p[i]
            counter[(k, piece)] += 1
            return piece
        s2, m2, o2 = _scan(s, key2t.keys(), act, "all", protect_affricates, masks[ci],
                           guard=lambda st, a, b, p=p: a in p, pass_pos=True, in_origin=origins[ci])
        out_c.append(s2); out_m.append(m2); out_o.append(o2)

    # ---- report + stats entry for the player's legend ----
    shared = sum(len(v) > 1 for v in occ.values())
    if verbose:
        print(f"word_quota: {len(occ)} candidate words, {shared} of them can take more than one target")
        for t in tg:
            ti = t["i"]
            nph = sum(c["n_phonemes"] for c in chosen[ti])
            msg = f"   {'/'.join(t['olds'])} → {'|'.join(t['outs'])}: {len(cand[ti])} candidate words " \
                  f"({sum(count[ti].values())} phonemes) -> {len(chosen[ti])} of {quota[ti]} words, {nph} phonemes"
            if ti in ptarget:
                msg += f" (target {ptarget[ti]}{'' if nph == ptarget[ti] else ', closest reachable'})"
            if ti in short:
                msg += f"   ⚠ {short[ti]} short: no assignment with one change per word can reach the quota"
            print(msg)
            if ti in notes:
                print("      note:", notes[ti])
    entry = dict(label=_describe_quota(rule), kind="word_quota",
                 changed=sum(len(v) for v in chosen.values()),
                 groups=[dict(old="/".join(t["olds"]), total=len(cand[t["i"]]), draws=None,
                              rows=[dict(new=o, n=sum(c["new"] == o for c in chosen[t["i"]]), p=None, changed=True)
                                    for o in t["outs"]])
                         for t in tg],
                 words={"/".join(t["olds"]): chosen[t["i"]] for t in tg},     # the chosen words, per target
                 quota={"/".join(t["olds"]): quota[t["i"]] for t in tg},
                 phonemes={"/".join(t["olds"]): sum(c["n_phonemes"] for c in chosen[t["i"]]) for t in tg},
                 phoneme_target={"/".join(t["olds"]): ptarget.get(t["i"]) for t in tg},
                 shortfall={"/".join(tg[i]["olds"]): n for i, n in short.items()})
    return out_c, out_m, out_o, entry


def _describe_quota(rule):
    parts = []
    for t in _targets(rule):
        new = t["outs"][0] if len(t["outs"]) == 1 else "{" + " | ".join(
            f"{o} {w * 100:.0f}%" for o, w in zip(t["outs"], t["ws"])) + "}"
        n = t["n"]
        n = f"{n * 100:.0f}% of words" if isinstance(n, float) and n < 1 else f"{n} words"
        P = t["phonemes"]
        if P is not None:
            n += f", {P * 100:.0f}% of phonemes" if isinstance(P, float) and P < 1 else f", ~{P} phonemes"
        parts.append(f"{'/'.join(t['olds'])}→{new} ({n})")
    flags = [f"{rule.get('batch', 10)} per round, {rule.get('order', 'rotate')}",
             f"{rule.get('occurrences', 'all')} occurrences", "one change per word"]
    if rule.get("skip_altered_words"):
        flags.append("skips words changed earlier")
    return "word_quota: " + ", ".join(parts) + f" [{'; '.join(flags)}]"


# the player calls functions.describe_rules(rules): teach it the new kind
_orig_describe = F.describe_rules
def describe_rules(rules):
    return "; ".join(_describe_quota(r) if r.get("kind") == "word_quota" else _orig_describe([r]) for r in rules)
F.describe_rules = describe_rules


def manipulate_with_quota(ipa, rules, protect_affricates=True, return_mask=False, seed=None,
                          return_stats=False, return_origin=False, verbose=True):
    """functions.manipulate() + rules of kind "word_quota" (see the module docstring).
    Rules are applied in order. Runs of ordinary rules are applied sentence by sentence exactly like
    manipulate() (same random draws for the same seed); a word_quota rule works on the whole text."""
    single = isinstance(ipa, str)
    chunks = [ipa] if single else list(ipa)
    base_rng = random.Random(seed)
    rngs = [random.Random(r["seed"]) if "seed" in r else base_rng for r in rules]
    counters = [Counter() for _ in rules]
    masks = [[False] * len(s) for s in chunks]
    origins = [[(j, j + 1) for j in range(len(s))] for s in chunks]
    stats = [None] * len(rules)

    k = 0
    while k < len(rules):
        if rules[k]["kind"] == "word_quota":
            chunks, masks, origins, stats[k] = _apply_quota(chunks, masks, origins, rules[k],
                                                            protect_affricates, rngs[k], counters[k], verbose)
            k += 1
            continue
        j = k                                           # run of ordinary rules: chunk-major, like manipulate()
        while j < len(rules) and rules[j]["kind"] != "word_quota":
            j += 1
        for ci in range(len(chunks)):
            s, m, o = chunks[ci], masks[ci], origins[ci]
            for r in range(k, j):
                s, m, o = _apply_rule(s, rules[r], protect_affricates, m, rngs[r], counters[r], o)
            chunks[ci], masks[ci], origins[ci] = s, m, o
        for r in range(k, j):
            stats[r] = _build_stats([rules[r]], [counters[r]])[0]
        k = j

    result = [chunks[0] if single else chunks]
    if return_mask:
        result.append(masks[0] if single else masks)
    if return_stats:
        result.append(stats)
    if return_origin:
        result.append(origins[0] if single else origins)
    return result[0] if len(result) == 1 else tuple(result)


# =====================================================================================================
# 3. CONDITIONS
# =====================================================================================================
def _check_tokens(symbols, where):
    bad = sorted({s for s in symbols if any(c not in VOCAB for c in s)})
    assert not bad, f"{where}: not in Kokoro's vocab (would be dropped from the speech): {bad}" \
                    f"   (for g use 'ɡ' U+0261)"


def _split_back(story, flat_lists):
    """Cut lists that run over all sentences of the story back into blocks."""
    out, k = [], 0
    for b in story["blocks"]:
        n = len(b["ipa"])
        out.append([lst[k:k + n] for lst in flat_lists]); k += n
    return out


def make_native(story):
    """The story without manipulation (same structure as the other conditions)."""
    blocks = [dict(index=b["index"], ipa=b["ipa"], manip=b["ipa"], masks=None, origins=None, rules=[], stats=[])
              for b in story["blocks"]]
    return dict(name="native", story=story, blocks=blocks, rules=[], stats=[])


def make_inconsistent(story, consonants=None, vowels=None, same_in_word=True, occurrences="all", batch=10,
                      order="rotate", skip_altered_words=False, seed=1, verbose=True):
    """Inconsistent condition, manipulated over the WHOLE story, then cut back into blocks.

    consonants : {old: alternatives}, every occurrence of `old` gets one alternative at random.
                 alternatives = a list (equal chance) or a dict {alternative: weight}
                 e.g. {"t": ["tʲ", "θ", "ɖ"], "k": ["x", "ɡ", "kʲ"]}
    vowels     : {old: dict(new=..., words=N, phonemes=P)}  exact number of words (and phonemes) of each
                 vowel to change over the whole story, at most one vowel change per word
                 e.g. {"e": dict(new="ø", words=290, phonemes=320), ...}   (phonemes is optional)
    same_in_word : every t (k ...) of one word gets the same alternative
    occurrences  : "all" every e of a chosen word | "partial" exact phonemes | "one" one e per word
    seed       : the same seed gives the same manipulation
    Consonants are changed first, vowels second (a word can have a consonant AND a vowel change,
    unless skip_altered_words=True)."""
    consonants, vowels = consonants or {}, vowels or {}
    rules = []
    if consonants:
        cmap = {old: ({a: 1 for a in alts} if isinstance(alts, (list, tuple)) else dict(alts))
                for old, alts in consonants.items()}
        _check_tokens([a for m in cmap.values() for a in m], "consonants")
        rules.append({"kind": "substitute_prob", "map": cmap, "same_in_word": same_in_word})
    if vowels:
        _check_tokens([v["new"] for v in vowels.values()], "vowels")
        rules.append({"kind": "word_quota", "occurrences": occurrences, "batch": batch, "order": order,
                      "skip_altered_words": skip_altered_words,
                      "targets": [{"old": v, "new": spec["new"], "words": spec["words"],
                                   "phonemes": spec.get("phonemes")} for v, spec in vowels.items()]})
    flat = [s for b in story["blocks"] for s in b["ipa"]]
    if verbose:
        print(f"inconsistent ({story['name']}, seed {seed}):")
    manip, masks, stats, origins = manipulate_with_quota(flat, rules, return_mask=True, return_stats=True,
                                                         return_origin=True, seed=seed, verbose=verbose)
    blocks = []
    for b, (m, mk, og) in zip(story["blocks"], _split_back(story, [manip, masks, origins])):
        blk = dict(index=b["index"], ipa=b["ipa"], manip=m, masks=mk, origins=og, rules=rules)
        blk["stats"] = block_changes(blk)                    # what actually changed in THIS block
        blocks.append(blk)
    return dict(name="inconsistent", story=story, blocks=blocks, rules=rules, stats=stats, seed=seed)


def make_consistent(story, errors, verbose=True):
    """Consistent condition: ONE fixed substitution per block, applied to EVERY occurrence.

    errors : one dict per block, {old: new}, e.g.
             [{"t": "tʲ", "k": "x",  "e": "ø"},     # block 1
              {"t": "θ",  "k": "ɡ",  "i": "y"},     # block 2
              {"t": "ɖ",  "k": "kʲ", "o": "ə"}]     # block 3
             An empty dict {} leaves that block native."""
    assert len(errors) == len(story["blocks"]), \
        f"errors has {len(errors)} entries, the story has {len(story['blocks'])} blocks"
    _check_tokens([n for e in errors for n in e.values()], "errors")
    blocks = []
    for b, err in zip(story["blocks"], errors):
        rules = [{"kind": "substitute", "map": dict(err)}] if err else []
        m, mk, st, og = manipulate_with_quota(b["ipa"], rules, return_mask=True, return_stats=True,
                                              return_origin=True, verbose=False)
        blk = dict(index=b["index"], ipa=b["ipa"], manip=m, masks=mk, origins=og, rules=rules)
        blk["stats"] = block_changes(blk)
        blocks.append(blk)
        if verbose:
            n = {(g["old"], r["new"]): r["n"] for e in st for g in e["groups"] for r in g["rows"]}
            print(f"consistent ({story['name']}), block {b['index']}: "
                  + (", ".join(f"{o}→{v} ×{n.get((o, v), 0)}" for o, v in err.items()) or "no errors"))
    return dict(name="consistent", story=story, blocks=blocks, rules=[r for b in blocks for r in b["rules"]])


# =====================================================================================================
# 4. SYNTHESIS
# =====================================================================================================
# ---- 4a. model helpers --------------------------------------------------------------------------------
@torch.no_grad()
def make_alignment(ps, dur):
    """Token-to-frame alignment matrix (1, n_tokens, n_frames) from durations."""
    dev = model.device
    n = _encode(ps)["n"]
    dur = torch.as_tensor(dur, device=dev).long().clamp(min=0).reshape(-1)
    assert dur.shape[0] == n, f"durations ({dur.shape[0]}) != tokens incl. BOS/EOS ({n})"
    idx = torch.repeat_interleave(torch.arange(n, device=dev), dur)
    aln = torch.zeros((n, idx.shape[0]), device=dev)
    aln[idx, torch.arange(idx.shape[0], device=dev)] = 1
    return aln.unsqueeze(0)


@torch.no_grad()
def predict_f0n(ps, pack, aln):
    """Predicted F0 (Hz) and energy N of voice `pack` on the given alignment (numpy, 2*n_frames)."""
    enc = _encode(ps)
    s = pack[len(ps) - 1][:, 128:]
    d = model.predictor.text_encoder(enc["d_en"], s, enc["L"], enc["tm"])
    F0, N = model.predictor.F0Ntrain(d.transpose(-1, -2) @ aln, s)
    return F0.squeeze().cpu().numpy(), N.squeeze().cpu().numpy()


@torch.no_grad()
def decode(ps, pack, aln, f0, n):
    """Decoder with voice `pack`'s timbre and the GIVEN F0 / energy curves."""
    enc = _encode(ps)
    ref_s = pack[len(ps) - 1]
    to_t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=model.device).unsqueeze(0)
    return model.decoder(enc["t_en"] @ aln, to_t(f0), to_t(n), ref_s[:, :128]).squeeze().cpu().numpy()


# ---- 4b. pitch / energy normalization ----------------------------------------------------------------
VOICED_HZ = 50.0

def voiced_mask(f0, thr=VOICED_HZ, rel=0.6):
    """Voiced = above thr AND above rel * the median voiced pitch (drops on/offset frames)."""
    v = f0 > thr
    return v & (f0 > rel * np.median(f0[v])) if v.any() else v


def f0_stats(f0_list):
    """(mean, std) of log-F0 over the voiced frames of several chunks."""
    lf = np.log(np.concatenate([f[voiced_mask(f)] for f in f0_list]))
    return lf.mean(), lf.std()


def en_stats(n_list):
    x = np.concatenate(n_list)
    return x.mean(), x.std()


def apply_f0(f0_d, st_d, st_t, mode):
    """Donor contour into the target's register. Unvoiced frames keep the donor's values."""
    if mode == "raw":
        return f0_d.copy()
    out, v = f0_d.copy(), voiced_mask(f0_d)
    z = np.log(f0_d[v]) - st_d[0]
    if mode == "meanstd":
        z = z / st_d[1] * st_t[1]
    out[v] = np.exp(z + st_t[0])
    return out


def apply_energy(n_d, st_d, st_t, mode):
    if mode == "raw":
        return n_d.copy()
    z = n_d - st_d[0]
    if mode == "meanstd":
        z = z / (st_d[1] + 1e-8) * st_t[1]
    return z + st_t[0]


def scale_durations(dur, factor, total=None):
    """Multiply durations by `factor` and round so the sum is EXACTLY `total`
    (default: round(sum * factor)), with every token >= 1 frame.
    Largest-remainder rounding: frames go to / come from the tokens whose
    rounding error is largest, so each token stays as close as possible to dur*factor."""
    x = np.asarray(dur, float) * factor
    total = int(round(x.sum())) if total is None else int(total)
    assert total >= len(x), f"total {total} < number of tokens {len(x)}"
    out = np.maximum(np.floor(x), 1).astype(int)
    while out.sum() < total:                       # add to the most under-rounded tokens
        k = min(total - out.sum(), len(x))
        out[np.argsort(-(x - out))[:k]] += 1
    while out.sum() > total:                       # remove from the most over-rounded tokens (never below 1)
        cand = np.where(out > 1)[0]
        k = min(out.sum() - total, len(cand))
        out[cand[np.argsort(x[cand] - out[cand])[:k]]] -= 1
    return out.tolist()


# ---- 4c. durations between the original and the accented string --------------------------------------
# map_durations      : reference durations of the ORIGINAL text -> accented tokens; a "ʲ" added by a rule
#                      keeps the TARGET voice's own frames, the consonant gets all the reference frames
# back_map_durations : accented durations -> original tokens (to predict the reference's pitch on the
#                      original text with exactly the same number of frames)
# rescale_free       : rate normalization that keeps the fixed (ʲ) frames
def _identity(ps):
    return [(j, j + 1) for j in range(len(ps))]


def own_flags(ps, mask, keep_own=("ʲ",)):
    '''One bool per KEPT token of `ps` (no BOS/EOS): True for a token in `keep_own`
    that a rule added (mask True). These keep the target voice's own duration.'''
    if mask is None or not keep_own:
        return [False] * sum(c in VOCAB for c in ps)
    return [bool(mask[j]) and ps[j] in keep_own for j, c in enumerate(ps) if c in VOCAB]


def map_durations(orig_ps, manip_ps, origin, src_dur, own_dur, own, diacritic_weight=0.3):
    '''src_dur : durations of ORIGINAL string (a donor, or the target), len kept(orig)+2
    own_dur : the target's durations of the ACCENTED string, len kept(manip)+2
    own     : own_flags(manip_ps, ...), one bool per kept accented token
    -> durations of the accented string. own tokens get own_dur, the others share the frames
       of what they replaced (transfer_durations); sum = sum(src_dur) + frames of the own tokens.'''
    origin = _identity(manip_ps) if origin is None else origin
    kept = [j for j, c in enumerate(manip_ps) if c in VOCAB]
    assert len(own) == len(kept) and len(own_dur) == len(kept) + 2
    drop = {j for j, f in zip(kept, own) if f}
    red_ps = "".join(c for j, c in enumerate(manip_ps) if j not in drop)
    red_or = [o for j, o in enumerate(origin) if j not in drop]
    red = transfer_durations(orig_ps, red_ps, red_or, src_dur, diacritic_weight)
    rest = iter(red[1:-1])
    out = [red[0]] + [own_dur[k] if f else next(rest) for k, f in enumerate(own, start=1)] + [red[-1]]
    assert next(rest, None) is None
    return out


def _groups(orig_ps, origin):
    '''Same grouping as transfer_durations: consecutive accented chars with overlapping source
    spans form one group; the groups' source ranges [a, b) partition the original string.'''
    groups = []
    for oi, (a, b) in enumerate(origin):
        if groups and a < groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], b); groups[-1][2].append(oi)
        else:
            groups.append([a, b, [oi]])
    if groups:
        groups[0][0] = 0
        for g, nxt in zip(groups, groups[1:]):
            g[1] = nxt[0]
        groups[-1][1] = len(orig_ps)
    return groups


def _split(total, weights):
    '''Largest-remainder split of `total` frames by `weights`.'''
    ideal = [total * w / sum(weights) for w in weights]
    alloc = [int(v) for v in ideal]
    for j in sorted(range(len(ideal)), key=lambda j: ideal[j] - alloc[j], reverse=True)[:total - sum(alloc)]:
        alloc[j] += 1
    return alloc


def back_map_durations(orig_ps, manip_ps, origin, manip_dur, ref_orig_dur):
    '''Accented durations -> durations of the ORIGINAL string with the SAME total, group by group.
    Inside a group with several original tokens (e.g. tʃ -> ʨ) the frames are split in proportion
    to `ref_orig_dur` (the donor's own durations of the original string).'''
    origin = _identity(manip_ps) if origin is None else origin
    m, k = [0] * len(manip_ps), 1
    for j, c in enumerate(manip_ps):
        if c in VOCAB:
            m[j] = manip_dur[k]; k += 1
    assert k == len(manip_dur) - 1, "manip_dur does not match manip_ps"
    r, k = [0] * len(orig_ps), 1
    for j, c in enumerate(orig_ps):
        if c in VOCAB:
            r[j] = ref_orig_dur[k]; k += 1
    assert k == len(ref_orig_dur) - 1, "ref_orig_dur does not match orig_ps"
    o, carry, last = [0] * len(orig_ps), 0, None
    for a, b, outs in _groups(orig_ps, origin):
        total = sum(m[oi] for oi in outs) + carry
        kept = [j for j in range(a, b) if orig_ps[j] in VOCAB]
        if not kept:                                   # nothing audible in the original here
            carry = total; continue
        carry = 0
        for j, v in zip(kept, _split(total, [max(r[j], 1) for j in kept])):
            o[j] = v
        last = kept[-1]
    if carry and last is not None:
        o[last] += carry
    out = [manip_dur[0]] + [o[j] for j, c in enumerate(orig_ps) if c in VOCAB] + [manip_dur[-1]]
    assert sum(out) == sum(manip_dur), (sum(out), sum(manip_dur))
    return out


def rescale_free(dur, fixed, total):
    '''Scale the non-fixed durations so the whole sentence has `total` frames; fixed ones unchanged.'''
    free = [i for i, f in enumerate(fixed) if not f]
    fsum = sum(d for d, f in zip(dur, fixed) if f)
    x = [dur[i] for i in free]
    new = scale_durations(x, (total - fsum) / sum(x), total=total - fsum)
    out = list(dur)
    for i, v in zip(free, new):
        out[i] = v
    return out


# ---- 4d. the engine -----------------------------------------------------------------------------------
def _voice_name(v, k):
    return v if isinstance(v, str) else f"voice{k}"


def assign_donors(n, donors, assign="balanced", seed=0):
    '''Donor index for each of n sentences.'''
    k = len(donors)
    names = [_voice_name(v, i) for i, v in enumerate(donors)]
    if isinstance(assign, (list, tuple)):
        assert len(assign) == n, f"assign has {len(assign)} entries for {n} sentences"
        return [a if isinstance(a, (int, np.integer)) else names.index(a) for a in assign]
    rng = np.random.default_rng(seed)
    if assign == "random":
        return rng.integers(0, k, n).tolist()
    if assign == "balanced":
        idx = np.concatenate([np.tile(np.arange(k), n // k), rng.choice(k, n % k, replace=False)])
        rng.shuffle(idx)
        return idx.astype(int).tolist()
    raise ValueError(f"unknown assign: {assign!r}")


def synth_accent(ipa_chunks, target_voice, donor_voices=None,
                 manip_chunks=None, masks=None, origins=None,
                 pitch=False, energy=False, duration=False,
                 donor_text="original", target_timing="manipulated",
                 keep_own=("ʲ",), diacritic_weight=0.3,
                 f0_mode="meanstd", energy_mode="meanstd", dur_mode="rate",
                 norm_scope="text", dur_scope="sentence",
                 assign="balanced", seed=0, speed=1.0, sr=SR):
    assert donor_text in ("original", "manipulated"), donor_text
    assert target_timing in ("original", "manipulated"), target_timing
    assert norm_scope in ("text", "sentence") and dur_scope in ("text", "sentence") and dur_mode in ("raw", "rate")
    if manip_chunks is None:                           # native: the accented text IS the original
        manip_chunks, masks, origins = ipa_chunks, None, None
    assert len(manip_chunks) == len(ipa_chunks), "manip_chunks must be parallel to ipa_chunks"
    if donor_voices is None:
        donor_voices = []
    if isinstance(donor_voices, str) or torch.is_tensor(donor_voices):
        donor_voices = [donor_voices]
    transfer = pitch or energy or duration
    assert donor_voices or not transfer, "pitch / energy / duration transfer needs donor_voices"
    keep_own = tuple(keep_own or ())
    names = [_voice_name(v, i) for i, v in enumerate(donor_voices)]
    tpack, tkey = _voice_pack(target_voice)
    dpacks = [_voice_pack(v) for v in donor_voices]                 # [(pack, key)]
    need_origin = (transfer and donor_text == "original") or target_timing == "original"

    # ---- pass 0: sentences, their original / accented strings, donor of each ----
    items = []
    for ci, (orig, ps) in enumerate(zip(ipa_chunks, manip_chunks)):
        if not ps.strip():
            continue
        origin = origins[ci] if origins is not None else (_identity(ps) if ps == orig else None)
        assert origin is not None or not need_origin, \
            f"sentence {ci} was manipulated: pass origins=... from manipulate(..., return_origin=True)"
        mask = masks[ci] if masks is not None else None
        own = own_flags(ps, mask, keep_own)
        items.append(dict(ci=ci, orig=orig, ps=ps, origin=origin, mask=mask, own=own,
                          fixed=[False] + own + [False]))
    dis = assign_donors(len(items), donor_voices, assign, seed) if donor_voices else [None] * len(items)
    for it, di in zip(items, dis):
        it["di"], it["donor"] = di, (names[di] if di is not None else None)

    def own_sum(it):                                   # frames of the tokens that keep the target's duration
        return sum(d for d, f in zip(it["dur_own"], it["fixed"]) if f)

    def donor_dur(it, d):                              # donor d's durations on the accented tokens
        dp, dk = dpacks[d]
        if donor_text == "manipulated":
            src_ = _durations_cached(it["ps"], dp, dk, speed)
            return [o if f else s_ for s_, o, f in zip(src_, it["dur_own"], it["fixed"])]
        return map_durations(it["orig"], it["ps"], it["origin"], _durations_cached(it["orig"], dp, dk, speed),
                             it["dur_own"], it["own"], diacritic_weight)

    # ---- pass 1: durations ----
    for it in items:
        it["dur_own"] = _durations_cached(it["ps"], tpack, tkey, speed)          # target on the accented text
        if target_timing == "manipulated":
            it["dur_t"] = list(it["dur_own"])
        else:
            it["dur_t"] = map_durations(it["orig"], it["ps"], it["origin"],
                                        _durations_cached(it["orig"], tpack, tkey, speed),
                                        it["dur_own"], it["own"], diacritic_weight)
        it["dur_d"] = donor_dur(it, it["di"]) if (duration and it["di"] is not None) else None
    if duration and dur_mode == "rate" and dur_scope == "text":
        # each donor slowed/sped to the target's overall rate (fixed ʲ frames excluded), then the text total is made exact
        free_t = sum(sum(it["dur_t"]) - own_sum(it) for it in items)
        rate = {d: free_t / sum(sum(donor_dur(it, d)) - own_sum(it) for it in items)
                for d in sorted({it["di"] for it in items})}
        L = np.array([(sum(it["dur_d"]) - own_sum(it)) * rate[it["di"]] for it in items])
        L *= free_t / L.sum()
        budgets = np.diff(np.round(np.concatenate([[0], np.cumsum(L)])).astype(int))
    for i, it in enumerate(items):
        if not duration:
            it["dur"] = list(it["dur_t"])
        elif dur_mode == "raw":
            it["dur"] = list(it["dur_d"])
        elif dur_scope == "sentence":
            it["dur"] = rescale_free(it["dur_d"], it["fixed"], sum(it["dur_t"]))
        else:
            it["dur"] = rescale_free(it["dur_d"], it["fixed"], int(budgets[i]) + own_sum(it))

    # ---- pass 2: F0 / energy of the target (accented text) and of EVERY donor (the text it reads) ----
    for it in items:
        it["aln"] = make_alignment(it["ps"], it["dur"])
        it["f0_t"], it["n_t"] = predict_f0n(it["ps"], tpack, it["aln"])
        it["f0_d"] = it["n_d"] = it["d_dur"] = None
        if pitch or energy:
            if donor_text == "manipulated":
                it["d_ps"], it["d_dur"], d_aln = it["ps"], it["dur"], it["aln"]
            else:                                      # original text, on a timeline that matches the accented one
                dp, dk = dpacks[it["di"]]
                it["d_ps"] = it["orig"]
                it["d_dur"] = back_map_durations(it["orig"], it["ps"], it["origin"], it["dur"],
                                                 _durations_cached(it["orig"], dp, dk, speed))
                d_aln = make_alignment(it["orig"], it["d_dur"])
            it["all_d"] = [predict_f0n(it["d_ps"], dp, d_aln) for dp, _ in dpacks]
            it["f0_d"], it["n_d"] = it["all_d"][it["di"]]
            assert len(it["f0_d"]) == len(it["f0_t"]), "donor and target frames differ"

    # ---- pass 3: normalize each sentence's donor into the target's register ----
    if (pitch or energy) and norm_scope == "text":
        sf_t = f0_stats([it["f0_t"] for it in items]); se_t = en_stats([it["n_t"] for it in items])
        sf_d = [f0_stats([it["all_d"][d][0] for it in items]) for d in range(len(dpacks))]
        se_d = [en_stats([it["all_d"][d][1] for it in items]) for d in range(len(dpacks))]
    for it in items:
        if pitch or energy:
            if norm_scope == "text":
                st_f, st_e, tt_f, tt_e = sf_d[it["di"]], se_d[it["di"]], sf_t, se_t
            else:
                st_f, st_e = f0_stats([it["f0_d"]]), en_stats([it["n_d"]])
                tt_f, tt_e = f0_stats([it["f0_t"]]), en_stats([it["n_t"]])
        it["f0"] = apply_f0(it["f0_d"], st_f, tt_f, f0_mode)       if pitch  else it["f0_t"]
        it["n"]  = apply_energy(it["n_d"], st_e, tt_e, energy_mode) if energy else it["n_t"]

    # ---- pass 4: decode + segments, concatenated like synth_aligned ----
    what = [w for w, on in [("pitch", pitch), ("energy", energy), ("duration", duration)] if on]
    audios, segments, seg_words, seg_altered = [], [], [], []
    T, F0o, F0t, Ne, Nt, chunks, sentences = [], [], [], [], [], [], []
    t0, word_offset = 0.0, 0
    for it in items:
        a = decode(it["ps"], tpack, it["aln"], it["f0"], it["n"])
        segs, widx, altd = align_phonemes(it["ps"], it["dur"], n_samples=len(a), sr=sr, altered_mask=it["mask"])
        segments += [(lab, s + t0, e + t0, info) for lab, s, e, info in segs]
        seg_words += [None if w is None else w + word_offset for w in widx]
        seg_altered += altd
        real = [w for w in widx if w is not None]
        word_offset += (max(real) + 1) if real else 0

        k = len(it["f0"])                                     # prosody curves on the global timeline
        T.append(t0 + np.arange(k) * (len(a) / k) / sr)
        F0o.append(np.where(voiced_mask(it["f0"]), it["f0"], np.nan))
        F0t.append(np.where(voiced_mask(it["f0_t"]), it["f0_t"], np.nan))
        Ne.append(it["n"]); Nt.append(it["n_t"])
        for arr, fill in [(T, t0 + len(a) / sr), (F0o, np.nan), (F0t, np.nan), (Ne, np.nan), (Nt, np.nan)]:
            arr.append(np.array([fill]))                      # NaN gap -> curves break between sentences

        info = {k_: it[k_] for k_ in ("ci", "orig", "ps", "origin", "mask", "fixed", "dur", "dur_own", "dur_t",
                                      "dur_d", "d_dur", "f0", "f0_t", "f0_d", "n", "n_t", "n_d")}
        info.update(chunk=it["ci"], target=_voice_name(target_voice, 0), donor=it["donor"], what=what,
                    donor_text=donor_text)
        chunks.append((a, info))
        sentences.append(dict(chunk=it["ci"], start=t0, end=t0 + len(a) / sr,
                              donor=it["donor"], donor_idx=it["di"]))
        audios.append(a); t0 += len(a) / sr

    pros = dict(t=np.concatenate(T), f0=np.concatenate(F0o), f0_t=np.concatenate(F0t),
                n=np.concatenate(Ne), n_t=np.concatenate(Nt), chunks=chunks, sentences=sentences,
                target=_voice_name(target_voice, 0), donors=names, what=what,
                modes=dict(f0=f0_mode, energy=energy_mode, dur=dur_mode, scope=norm_scope,
                           dur_scope=dur_scope, assign=assign if isinstance(assign, str) else "fixed", seed=seed,
                           donor_text=donor_text, target_timing=target_timing, keep_own=keep_own))
    return np.concatenate(audios), segments, seg_words, seg_altered, pros


# ---- 4e. one call per condition -----------------------------------------------------------------------
def _is_one_voice(x):
    # a voice = a name, a tensor, or a (name, tensor) pair
    return (x is None or isinstance(x, str) or torch.is_tensor(x)
            or (isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) and torch.is_tensor(x[1])))


def _per_block(x, n, what):
    if _is_one_voice(x):
        return [x] * n
    x = list(x)
    assert len(x) == n, (f"{what}: {len(x)} entries but the story has {n} block(s). "
                         + ("Check that the story file has a line starting with ### between the blocks "
                            "(see the load_story printout)." if n == 1 else ""))
    return x


def _voice(v, b):
    """-> (label, voice). v = name | tensor | (name, tensor); an unnamed tensor is called 'voice_block<b>'."""
    if isinstance(v, tuple):
        return v[0], v[1]
    if torch.is_tensor(v):
        return f"voice_block{b}", v
    return v, v


def render_condition(cond, voices, ref_voice=None, duration=True, pitch=False, energy=False,
                     dur_mode="raw", dur_scope="sentence", target_timing="manipulated", keep_own=("ʲ",),
                     f0_mode="meanstd", energy_mode="meanstd", norm_scope="text", speed=1.0, verbose=True):
    """Synthesize every block of a condition.

    voices     : one voice for all blocks, or a list with one voice per block. A voice is a name
                 ("im_nicola"), a loaded tensor (torch.load(...)), or a (name, tensor) pair such as
                 ("nicola_20", torch.load("voices/nicola_20.pt")) so titles and file summaries show the name
    ref_voice  : reference voice that reads the ORIGINAL (unmanipulated) text and gives its timing
                 (and its pitch / energy if pitch=True / energy=True). One voice, a list per block,
                 or None = no transfer (each voice times the accented text itself).
    duration / pitch / energy : what is taken from the reference voice
    dur_mode   : "raw"  -> exactly the reference voice's frames (same timing for every voice / condition)
                 "rate" -> the reference rhythm, stretched to the target voice's own length
    dur_scope  : (rate only) "sentence" every sentence as long as the target's own | "text" only the total
    target_timing : the target's own timing when nothing is transferred (and the length for "rate"):
                 "manipulated" = it times the accented text itself; "original" = its timing of the original
                 text mapped onto the accented one
    keep_own   : tokens added by a rule that keep the TARGET voice's duration (default ʲ: in kʲ the k gets the
                 reference frames of k, the ʲ the target's own frames)
    Returns one result dict per block."""
    n = len(cond["blocks"])
    voices = _per_block(voices, n, "voices")
    refs = _per_block(ref_voice, n, "ref_voice")
    out = []
    for blk, v, ref in zip(cond["blocks"], voices, refs):
        v_label, v = _voice(v, blk["index"])
        ref_label, ref = _voice(ref, blk["index"]) if ref is not None else (None, None)
        transfer = ref is not None and (duration or pitch or energy)
        native = blk["manip"] is blk["ipa"] or blk["masks"] is None
        if verbose:
            what = [w for w, on in (("duration", duration), ("pitch", pitch), ("energy", energy)) if on]
            print(f"  {cond['name']} block {blk['index']}: voice {v_label}"
                  + (f", {' + '.join(what)} from {ref_label} reading the original text" if transfer else
                     ", own timing and prosody"))
        a, segs, words, alt, pros = synth_accent(
            blk["ipa"], v, [ref] if transfer else None,
            manip_chunks=None if native else blk["manip"], masks=None if native else blk["masks"],
            origins=None if native else blk["origins"],
            pitch=transfer and pitch, energy=transfer and energy, duration=transfer and duration,
            donor_text="original", target_timing=target_timing, keep_own=keep_own,
            f0_mode=f0_mode, energy_mode=energy_mode, dur_mode=dur_mode, norm_scope=norm_scope,
            dur_scope=dur_scope, speed=speed)
        out.append(dict(block=blk["index"], voice=v_label, ref_voice=ref_label if transfer else None,
                        audio=a, segments=segs, seg_words=words, seg_altered=alt, pros=pros))
    return out


# =====================================================================================================
# 5. EXPORT (written to disk, never displayed)
# =====================================================================================================
_PROS_PANEL = r"""
<div id="prosLegend" style="font-size:11.5px;color:#555;margin:8px 0 2px;display:flex;gap:14px;flex-wrap:wrap;align-items:center;">
  <span id="prosDonors"></span>
  <span><span style="display:inline-block;width:22px;border-top:2px dashed #94a3b8;vertical-align:middle"></span> target's own prosody</span>
  <span id="prosInfo" style="color:#777"></span>
</div>
<canvas id="pc" width="1040" height="170" style="width:100%;height:170px;background:#fff;border:1px solid #e5e7eb;border-radius:6px;cursor:pointer"></canvas>
"""

_PROS_JS = r"""
const PROS=__PROS__;
const pc=document.getElementById('pc'), pctx=pc.getContext('2d');
document.getElementById('prosInfo').textContent = PROS.info;
const PAL=['#f59e0b','#10b981','#8b5cf6','#ef4444','#0ea5e9','#ec4899'];
const dcol=i=>PAL[i % PAL.length];
(function(){ const box=document.getElementById('prosDonors');
    PROS.donors.forEach((d,i)=>{ const sp=document.createElement('span'); sp.style.marginRight='12px';
        sp.innerHTML='<span style="display:inline-block;width:22px;border-top:2.5px solid '+dcol(i)
            +';vertical-align:middle"></span> '+d+' <span style="color:#999">×'+PROS.counts[i]+'</span>';
        box.appendChild(sp); }); })();
function drawPros(){
    const W=pc.width, H=pc.height, a=view[0], b=view[1];
    pctx.clearRect(0,0,W,H);
    const TOP=16, H1=Math.round(H*0.66);
    const lo=Math.log(PROS.f0lim[0]), hi=Math.log(PROS.f0lim[1]);
    const X=t=>(t-a)/(b-a)*W;
    const yF=f=>H1-6-(Math.log(f)-lo)/(hi-lo)*(H1-TOP-14);
    const yE=e=>H-4-(e-PROS.nlim[0])/(PROS.nlim[1]-PROS.nlim[0])*(H-H1-14);
    pctx.font='10px system-ui,sans-serif'; pctx.lineWidth=1;
    PROS.sents.forEach(s=>{ if(s.e<a||s.s>b) return;              // donor strip, one block per sentence
        const x0=Math.max(0,X(s.s)), x1=Math.min(W,X(s.e));
        pctx.fillStyle=dcol(s.d)+'33'; pctx.fillRect(x0,0,x1-x0-1,TOP-3);
        pctx.fillStyle=dcol(s.d); pctx.fillRect(x0,TOP-4,x1-x0-1,2);
        pctx.save(); pctx.beginPath(); pctx.rect(x0,0,x1-x0-2,TOP); pctx.clip();
        pctx.fillStyle='#333'; pctx.fillText(PROS.donors[s.d],x0+3,TOP-6); pctx.restore(); });
    PROS.ticks.forEach(f=>{ const y=yF(f); pctx.strokeStyle='#f1f1f1';
        pctx.beginPath(); pctx.moveTo(0,y); pctx.lineTo(W,y); pctx.stroke();
        pctx.fillStyle='#999'; pctx.fillText(f+' Hz',3,y-2); });
    pctx.strokeStyle='#e5e7eb'; pctx.beginPath(); pctx.moveTo(0,H1+2); pctx.lineTo(W,H1+2); pctx.stroke();
    pctx.fillStyle='#999'; pctx.fillText('energy',3,H1+13);
    pctx.strokeStyle='#00000012';                                   // word starts, like the waveform
    groups.forEach(g=>{ if(g.w===null||g.s<a||g.s>b) return; const x=X(g.s);
        pctx.beginPath(); pctx.moveTo(x,TOP); pctx.lineTo(x,H); pctx.stroke(); });
    const t=PROS.t;
    function line(arr, yf, col, dash, lw, t0, t1){
        pctx.strokeStyle=col; pctx.setLineDash(dash); pctx.lineWidth=lw; pctx.beginPath();
        let pen=false;
        for(let i=0;i<t.length;i++){
            const v=arr[i];
            if(v===null || t[i]<Math.max(a,t0) || t[i]>Math.min(b,t1)){ pen=false; continue; }
            const x=X(t[i]), y=yf(v);
            if(pen) pctx.lineTo(x,y); else { pctx.moveTo(x,y); pen=true; }
        }
        pctx.stroke(); pctx.setLineDash([]); pctx.lineWidth=1;
    }
    line(PROS.f0_t, yF, '#94a3b8', [4,3], 1.5, -1, 1e9);
    line(PROS.n_t,  yE, '#94a3b8', [4,3], 1.2, -1, 1e9);
    PROS.sents.forEach(s=>{                                         // synthesized curves in the donor's color
        line(PROS.f0, yF, dcol(s.d), [], 2.2, s.s, s.e);
        line(PROS.n,  yE, dcol(s.d), [], 1.8, s.s, s.e); });
    const cx=X(au.currentTime); pctx.strokeStyle='#f43f5e'; pctx.lineWidth=2;
    pctx.beginPath(); pctx.moveTo(cx,0); pctx.lineTo(cx,H); pctx.stroke(); pctx.lineWidth=1;
}
"""

_IDS = ["pv", "pageTitle", "rulesBox", "statsBox", "statsTotal", "statsBody", "au", "pp", "sp", "spl",
        "tt", "legend", "durNote", "wf", "ph", "pc", "prosLegend", "prosInfo", "prosDonors"]


def _prosody_template():
    """The player template from functions.py with the prosody panel spliced in."""
    t = F._TEMPLATE
    anchors = ['<canvas id="wf"', "let view=[0,DUR];", "function drawWave(){", "\ndrawWave();\n})();"]
    missing = [a for a in anchors if a not in t]
    assert not missing, f"player template changed, anchors not found: {missing}"
    i = t.index('<canvas id="wf"'); j = t.index("</canvas>", i) + len("</canvas>")
    t = t[:j] + _PROS_PANEL + t[j:]
    t = t.replace("let view=[0,DUR];", "let view=[0,DUR];\n" + _PROS_JS, 1)
    t = t.replace("function drawWave(){", "function drawWave(){ drawPros();", 1)
    t = t.replace("\ndrawWave();\n})();", "\npc.onclick=wf.onclick; pc.onwheel=wf.onwheel;\ndrawWave();\n})();", 1)
    return t


def _uniquify_ids(html):
    uid = uuid.uuid4().hex[:8]
    for i in _IDS:
        html = html.replace(f'id="{i}"', f'id="{i}_{uid}"').replace(f"getElementById('{i}')", f"getElementById('{i}_{uid}')")
    return html


def _pros_payload(pros):
    from collections import Counter
    tolist = lambda x, r: [None if not np.isfinite(v) else round(float(v), r) for v in x]
    f0_all = np.concatenate([pros["f0"], pros["f0_t"]]); f0_all = f0_all[np.isfinite(f0_all)]
    lo, hi = np.percentile(f0_all, [1, 99]); lo, hi = lo / 1.15, hi * 1.15
    n_all = np.concatenate([pros["n"], pros["n_t"]]); n_all = n_all[np.isfinite(n_all)]
    ticks = [f for f in (50, 75, 100, 150, 200, 300, 400, 500) if lo < f < hi]
    m = lambda f: np.exp(np.nanmean(np.log(f)))
    md = pros["modes"]
    if pros["what"]:
        info = (" + ".join(pros["what"]) + f" · F0 {md['f0']}, energy {md['energy']}, duration {md['dur']}"
                f" ({md['dur_scope']}), stats per {md['scope']}, assign {md['assign']} (seed {md['seed']})"
                f" · donor reads the {md['donor_text']} text")
    else:
        info = "nothing transferred (native)"
    info += f" · mean F0 {m(pros['f0']):.0f} Hz (target's own {m(pros['f0_t']):.0f} Hz)"
    cnt = Counter(s_["donor_idx"] for s_ in pros["sentences"])
    sents = [dict(s=round(s_["start"], 4), e=round(s_["end"], 4), d=s_["donor_idx"]) for s_ in pros["sentences"]]
    if not pros["what"]:          # native: no donor strip colours
        sents, donors, counts = [], [], []
    else:
        donors, counts = pros["donors"], [cnt.get(i, 0) for i in range(len(pros["donors"]))]
    return dict(t=tolist(pros["t"], 4), f0=tolist(pros["f0"], 1), f0_t=tolist(pros["f0_t"], 1),
                n=tolist(pros["n"], 3), n_t=tolist(pros["n_t"], 3),
                f0lim=[float(lo), float(hi)], nlim=[float(n_all.min()), float(n_all.max())],
                ticks=ticks, info=info, sents=sents, donors=donors, counts=counts)


def export_player(result, path, words=None, title=None, rules=None, stats=None):
    """Write ONE interactive player (audio, word boxes, token durations, pitch / energy panel) to `path`.
    words : one spelled word per IPA word (for the word labels); None = no labels."""
    original = F._TEMPLATE
    try:
        F._TEMPLATE = _prosody_template()
        text = " ".join(w.replace(" ", "_") or "?" for w in words) if words else None
        html = audiovisualize_interactive(result["audio"], result["segments"], sr=SR, out_html=None,
                                          seg_words=result["seg_words"], text=text,
                                          seg_altered=result["seg_altered"], title=title, rules=rules, stats=stats)
    finally:
        F._TEMPLATE = original
    html = _uniquify_ids(html.replace("__PROS__", json.dumps(_pros_payload(result["pros"]))))
    from html import escape
    page = ('<!DOCTYPE html>\n<html><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{escape(title or "Kokoro")}</title></head>\n<body>\n' + html + '\n</body></html>\n')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(page, encoding="utf-8")
    return str(path)


def merge_results(results):
    """Join block results into one (audio, segments, words and pitch curves shifted in time)."""
    audio, segs, words, alt = [], [], [], []
    T, F0o, F0t, Ne, Nt, sents, chunks = [], [], [], [], [], [], []
    donors, t0, w0 = [], 0.0, 0
    for r in results:
        p = r["pros"]
        audio.append(r["audio"])
        segs += [(lab, s + t0, e + t0, info) for lab, s, e, info in r["segments"]]
        words += [None if w is None else w + w0 for w in r["seg_words"]]
        alt += r["seg_altered"]
        real = [w for w in r["seg_words"] if w is not None]
        w0 += (max(real) + 1) if real else 0
        T.append(p["t"] + t0); F0o.append(p["f0"]); F0t.append(p["f0_t"]); Ne.append(p["n"]); Nt.append(p["n_t"])
        for d in p["donors"]:
            if d not in donors:
                donors.append(d)
        for s_ in p["sentences"]:
            di = donors.index(p["donors"][s_["donor_idx"]]) if s_["donor_idx"] is not None else None
            sents.append(dict(s_, start=s_["start"] + t0, end=s_["end"] + t0, donor_idx=di))
        chunks += p["chunks"]
        t0 += len(r["audio"]) / SR
    p0 = results[0]["pros"]
    pros = dict(t=np.concatenate(T), f0=np.concatenate(F0o), f0_t=np.concatenate(F0t), n=np.concatenate(Ne),
                n_t=np.concatenate(Nt), chunks=chunks, sentences=sents, donors=donors,
                target=" / ".join(dict.fromkeys(r["voice"] for r in results)),
                what=sorted({w for r in results for w in r["pros"]["what"]}), modes=p0["modes"])
    return dict(block="all", voice=pros["target"], ref_voice=" / ".join(dict.fromkeys(
        r["ref_voice"] for r in results if r["ref_voice"])) or None,
                audio=np.concatenate(audio), segments=segs, seg_words=words, seg_altered=alt, pros=pros)


def _block_words(cond, b):
    story_blk = cond["story"]["blocks"][b]
    return [w for s in story_blk["spelling"] for w in s]


def export_condition(cond, results, out_dir, weights=None, prefix=None):
    """Write, for every block and for the whole condition:
         <prefix>_block<N>.html / .wav   and   <prefix>_full.html / .wav
       plus <prefix>_ipa.txt (native and accented IPA of every sentence) and <prefix>_summary.csv.
    prefix : default "<condition>_<story>". Nothing is displayed; the written paths are returned."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or f"{cond['name']}_{cond['story']['name']}"
    story = cond["story"]["name"]
    written = []
    for blk, r in zip(cond["blocks"], results):
        b = blk["index"]
        title = (f"{cond['name'].capitalize()} · {story} · block {b} · voice {r['voice']}"
                 + (f" · timing from {r['ref_voice']}" if r["ref_voice"] else ""))
        written.append(export_player(r, out_dir / f"{prefix}_block{b}.html", words=_block_words(cond, b - 1),
                                     title=title, rules=blk["rules"] or None, stats=blk["stats"] or None))
        sf.write(out_dir / f"{prefix}_block{b}.wav", r["audio"], SR); written.append(str(out_dir / f"{prefix}_block{b}.wav"))
    full = merge_results(results)
    all_words = [w for i in range(len(cond["blocks"])) for w in _block_words(cond, i)]
    full_stats = block_changes(dict(ipa=[s for b in cond["blocks"] for s in b["ipa"]],
                                    manip=[s for b in cond["blocks"] for s in b["manip"]]), label="Changes in the whole story")
    rules = cond["rules"]
    written.append(export_player(full, out_dir / f"{prefix}_full.html", words=all_words,
                                 title=f"{cond['name'].capitalize()} · {story} · whole story · voices {full['voice']}",
                                 rules=rules or None, stats=full_stats or None))
    sf.write(out_dir / f"{prefix}_full.wav", full["audio"], SR); written.append(str(out_dir / f"{prefix}_full.wav"))
    with open(out_dir / f"{prefix}_ipa.txt", "w", encoding="utf-8") as f:
        for blk in cond["blocks"]:
            f.write(f"### block {blk['index']}\n")
            for nat, man in zip(blk["ipa"], blk["manip"]):
                f.write(f"native  : {nat}\naccented: {man}\n\n")
    written.append(str(out_dir / f"{prefix}_ipa.txt"))
    rows = distance_table(cond, weights)
    with open(out_dir / f"{prefix}_summary.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["block", "voice", "ref_voice", "duration_s", "sentences", "words", "phonemes", "words_changed",
                  "pw", "pw_w", "lev", "lev_w", "changes"]
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader()
        for row, r in zip(rows, results + [full]):
            wr.writerow({**{k: row[k] for k in fields if k in row}, "voice": r["voice"], "ref_voice": r["ref_voice"],
                         "duration_s": round(len(r["audio"]) / SR, 2),
                         "changes": "; ".join(f"{a}→{b} ×{n}" for (a, b), n in row["pairs"].most_common())})
    written.append(str(out_dir / f"{prefix}_summary.csv"))
    print(f"written to {out_dir}/ :")
    for p in written:
        print("   ", Path(p).name)
    return written


# =====================================================================================================
# 6. MEASUREMENT: per-word normalized Levenshtein distance
# =====================================================================================================
# A word is split into phoneme units (base phoneme + its diacritics: tʲ, tː = one unit; stress ignored).
# d = Levenshtein distance / units in the native word; then mean over the words of a sentence, mean of
# the sentence means in a block, mean of the block means for the story.
#   pw   : every change costs 1           (in %)
#   pw_w : every change costs its weight  (in %; weights from DEFAULT_WEIGHTS or your own dict,
#          pairs you did not list fall back to the number of features that change)

DEFAULT_WEIGHTS = {            # native -> replacement : weight (prefilled = number of features that change)
    ("t", "tʲ"): 1, ("t", "θ"): 2, ("t", "ɖ"): 2,
    ("k", "x"): 1, ("k", "ɡ"): 1, ("k", "kʲ"): 1,
    ("e", "ø"): 1, ("i", "y"): 1, ("o", "ə"): 3,
}

CONS = {   # voice, place, manner
 'p':('-','bilabial','stop'), 'b':('+','bilabial','stop'), 'm':('+','bilabial','nasal'),
 'ɸ':('-','bilabial','fricative'), 'β':('+','bilabial','fricative'),
 'f':('-','labiodental','fricative'), 'v':('+','labiodental','fricative'), 'ʋ':('+','labiodental','approximant'),
 'θ':('-','dental','fricative'), 'ð':('+','dental','fricative'),
 't':('-','alveolar','stop'), 'd':('+','alveolar','stop'), 'n':('+','alveolar','nasal'),
 's':('-','alveolar','fricative'), 'z':('+','alveolar','fricative'), 'S':('-','alveolar','fricative'),
 'ʦ':('-','alveolar','affricate'), 'ʣ':('+','alveolar','affricate'),
 'r':('+','alveolar','trill'), 'ɾ':('+','alveolar','tap'), 'ɹ':('+','alveolar','approximant'), 'l':('+','alveolar','lateral'),
 'ʃ':('-','postalveolar','fricative'), 'ʒ':('+','postalveolar','fricative'),
 'ʧ':('-','postalveolar','affricate'), 'ʤ':('+','postalveolar','affricate'),
 'ɕ':('-','alveolopalatal','fricative'), 'ʨ':('-','alveolopalatal','affricate'), 'ʥ':('+','alveolopalatal','affricate'),
 'ʈ':('-','retroflex','stop'), 'ɖ':('+','retroflex','stop'), 'ɳ':('+','retroflex','nasal'),
 'ʂ':('-','retroflex','fricative'), 'ɽ':('+','retroflex','tap'), 'ɻ':('+','retroflex','approximant'),
 'c':('-','palatal','stop'), 'ɟ':('+','palatal','stop'), 'ɲ':('+','palatal','nasal'), 'ç':('-','palatal','fricative'),
 'ʝ':('+','palatal','fricative'), 'j':('+','palatal','approximant'), 'ʎ':('+','palatal','lateral'),
 'k':('-','velar','stop'), 'ɡ':('+','velar','stop'), 'g':('+','velar','stop'), 'ŋ':('+','velar','nasal'),
 'x':('-','velar','fricative'), 'ɣ':('+','velar','fricative'), 'ɰ':('+','velar','approximant'),
 'w':('+','labiovelar','approximant'),
 'q':('-','uvular','stop'), 'ɴ':('+','uvular','nasal'), 'χ':('-','uvular','fricative'), 'ʁ':('+','uvular','fricative'),
 'h':('-','glottal','fricative'), 'ʔ':('-','glottal','stop'),
}
VOWS = {   # height, backness, rounding
 'i':('close','front','-'), 'y':('close','front','+'), 'ɨ':('close','central','-'), 'ɯ':('close','back','-'),
 'u':('close','back','+'), 'ɪ':('near-close','front','-'), 'ʊ':('near-close','back','+'),
 'e':('close-mid','front','-'), 'ø':('close-mid','front','+'), 'ɤ':('close-mid','back','-'), 'o':('close-mid','back','+'),
 'ə':('mid','central','-'),
 'ɛ':('open-mid','front','-'), 'œ':('open-mid','front','+'), 'ɜ':('open-mid','central','-'),
 'ʌ':('open-mid','back','-'), 'ɔ':('open-mid','back','+'),
 'æ':('near-open','front','-'), 'ɐ':('near-open','central','-'),
 'a':('open','front','-'), 'ɑ':('open','back','-'), 'ɒ':('open','back','+'),
}
DIACRITICS = {'ʲ': 'palatalized', 'ʰ': 'aspirated', 'ː': 'long', '̃': 'nasalized'}
INDEL_FEAT = 3          # cost of inserting / deleting a whole unit in lev_feat
DIPHTHONGS = {'A': 'e', 'I': 'a', 'W': 'a', 'O': 'o', 'Q': 'ə', 'Y': 'ɔ'}     # eɪ aɪ aʊ oʊ əʊ ɔɪ
STRESS = {"ˈ", "ˌ"}


def units(word):
    '''A word -> list of (base, frozenset(diacritic features)); stress marks dropped.'''
    out = []
    for c in word:
        if c in STRESS:
            continue
        if c in DIACRITICS and out:
            b, d = out[-1]; out[-1] = (b, d | {DIACRITICS[c]})
        elif c in DIACRITICS or c in BOUNDARY:
            continue
        else:
            out.append((c, frozenset({"long"}) if c == "S" else frozenset()))
    return out


_unknown = set()
def feat_cost(a, b):
    '''Number of features that differ between two units.'''
    if a == b:
        return 0
    (ba, da), (bb, db) = a, b
    dia = len(da ^ db) + ((ba in DIPHTHONGS) != (bb in DIPHTHONGS))
    ba_, bb_ = DIPHTHONGS.get(ba, ba), DIPHTHONGS.get(bb, bb)
    if ba != bb and ba_ == bb_:                      # e.g. A (eɪ) vs I... same first vowel, different glide
        return max(dia, 1)
    ba, bb = ba_, bb_
    fa = CONS.get(ba) or VOWS.get(ba); fb = CONS.get(bb) or VOWS.get(bb)
    if fa is None or fb is None:
        _unknown.update(x for x, f in ((ba, fa), (bb, fb)) if f is None)
        return (ba != bb) + dia
    if (ba in CONS) != (bb in CONS):
        return 3 + dia
    return sum(x != y for x, y in zip(fa, fb)) + dia


def lev(a, b, cost=None, indel=1):
    '''Levenshtein distance between two unit lists + the list of edit operations (native, manipulated).'''
    cost = cost or (lambda x, y: 0 if x == y else 1)
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1)); D[:, 0] = np.arange(n + 1) * indel; D[0, :] = np.arange(m + 1) * indel
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i-1, j] + indel, D[i, j-1] + indel, D[i-1, j-1] + cost(a[i-1], b[j-1]))
    ops, i, j = [], n, m                                     # backtrace
    while i or j:
        if i and j and D[i, j] == D[i-1, j-1] + cost(a[i-1], b[j-1]):
            if a[i-1] != b[j-1]: ops.append((a[i-1], b[j-1]))
            i, j = i - 1, j - 1
        elif i and D[i, j] == D[i-1, j] + indel:
            ops.append((a[i-1], None)); i -= 1
        else:
            ops.append((None, b[j-1])); j -= 1
    return D[n, m], ops[::-1]


def show(u):
    if u is None: return "∅"
    b, d = u
    return b + "".join(k for k, v in DIACRITICS.items() if v in d)


def _key(a, b):
    # (native, replacement) as strings, diacritics shared by both sides removed
    (ba, da), (bb, db) = a, b
    common = da & db
    return show((ba, da - common)), show((bb, db - common))


def _weight_fn(weights, missing="features"):
    weights = DEFAULT_WEIGHTS if weights is None else weights
    def w(a, b):
        if a == b:
            return 0
        key = _key(a, b)
        if key in weights:
            return weights[key]
        return feat_cost(a, b) if missing == "features" else missing
    return w


def measure_distance(native_chunks, manip_chunks, weights=None, missing="features", indel=3):
    """Per-word normalized distance between native and accented IPA (lists of sentences).
    Returns dict(pw, pw_w, lev, lev_w, words, phonemes, words_changed, pairs=Counter((native, new)))."""
    wfn = _weight_fn(weights, missing)
    r = dict(words=0, phonemes=0, words_changed=0, lev=0.0, lev_w=0.0, pairs=Counter())
    sent_pw, sent_pw_w = [], []
    for nat_s, man_s in zip(native_chunks, manip_chunks):
        wn = [nat_s[a:c] for a, c in _word_spans(nat_s)]
        wm = [man_s[a:c] for a, c in _word_spans(man_s)]
        pairs = list(zip(wn, wm)) if len(wn) == len(wm) else [(" ".join(wn), " ".join(wm))]
        d_, dw_ = [], []
        for x, y in pairs:
            ux, uy = units(x), units(y)
            if not ux:
                continue
            d, ops = lev(ux, uy)
            dw, _ = lev(ux, uy, wfn, indel)
            d_.append(d / len(ux)); dw_.append(dw / len(ux))
            r["lev"] += d; r["lev_w"] += dw; r["words_changed"] += d > 0
            r["words"] += 1; r["phonemes"] += len(ux)
            for a, b in ops:                         # tː→tʲː is counted as t→tʲ
                r["pairs"][_key(a, b) if a is not None and b is not None else (show(a), show(b))] += 1
        if d_:
            sent_pw.append(np.mean(d_)); sent_pw_w.append(np.mean(dw_))
    r["pw"] = 100 * float(np.mean(sent_pw)) if sent_pw else 0.0
    r["pw_w"] = 100 * float(np.mean(sent_pw_w)) if sent_pw_w else 0.0
    return r


def distance_table(cond, weights=None, missing="features"):
    """measure_distance() for every block, plus a row "all" (pw / pw_w = mean of the block means)."""
    rows = []
    for blk in cond["blocks"]:
        r = measure_distance(blk["ipa"], blk["manip"], weights, missing)
        rows.append(dict(block=blk["index"], sentences=len(blk["ipa"]), **r))
    tot = dict(block="all", sentences=sum(r["sentences"] for r in rows),
               **{k: sum(r[k] for r in rows) for k in ("words", "phonemes", "words_changed", "lev", "lev_w")},
               pw=float(np.mean([r["pw"] for r in rows])), pw_w=float(np.mean([r["pw_w"] for r in rows])),
               pairs=sum((r["pairs"] for r in rows), Counter()))
    return rows + [tot]


def block_changes(blk, label="Changes in this block"):
    """What actually changed (native -> accented, counted per phoneme unit), in the format of
    manipulate(..., return_stats=True), for the player's legend."""
    groups = defaultdict(Counter)
    totals = Counter()
    for nat_s, man_s in zip(blk["ipa"], blk["manip"]):
        for (a0, b0), (a1, b1) in zip(_word_spans(nat_s), _word_spans(man_s)):
            un, um = units(nat_s[a0:b0]), units(man_s[a1:b1])
            for u in un:
                totals[show(u)] += 1
            if len(un) == len(um):
                for x, y in zip(un, um):
                    if x != y:
                        groups[show(x)][show(y)] += 1
    if not groups:
        return []
    gl = [dict(old=o, total=totals[o], draws=None,
               rows=[dict(new=n, n=c, p=None, changed=True) for n, c in groups[o].most_common()])
          for o in sorted(groups)]
    return [dict(label=label, kind="substitute", changed=sum(c for g in groups.values() for c in g.values()),
                 groups=gl)]


def print_summary(cond, weights=None, missing="features"):
    """One line per block: what changed, how many words, and the per-word normalized distance."""
    rows = distance_table(cond, weights, missing)
    print(f"\n{cond['name']} · {cond['story']['name']}")
    print(f"  {'block':<6}{'sent':>5}{'words':>7}{'changed':>9}{'pw %':>8}{'pw_w %':>8}   changes")
    for r in rows:
        ch = ", ".join(f"{a}→{b} ×{n}" for (a, b), n in sorted(r["pairs"].items(), key=lambda kv: (kv[0][0], -kv[1])))
        print(f"  {str(r['block']):<6}{r['sentences']:>5}{r['words']:>7}{r['words_changed']:>9}"
              f"{r['pw']:>8.1f}{r['pw_w']:>8.1f}   {ch if r['block'] != 'all' else ''}")
    unknown = {k for r in rows for k in r["pairs"] if k not in (weights or DEFAULT_WEIGHTS)}
    if unknown:
        print("  (weights not given, feature count used for: " + ", ".join(f"{a}→{b}" for a, b in sorted(unknown)) + ")")
    return rows
