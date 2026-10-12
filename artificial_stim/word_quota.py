"""
word_quota.py — "manipulate exactly N words" rules for the IPA manipulation engine in functions.py.

    from word_quota import manipulate_with_quota
    out, masks, stats, origins = manipulate_with_quota(ipa_chunks, rules, return_mask=True,
                                                        return_stats=True, return_origin=True, seed=1)

Same signature and return values as functions.manipulate(). Ordinary rules work exactly as before
(same results for the same seed); you can additionally use rules of kind "word_quota":

    {"kind": "word_quota",
     "targets": [{"old": "e", "new": "ɛ", "n": 40},
                 {"old": "o", "new": "ɔ", "n": 45},
                 {"old": "i", "choices": {"ɪ": 50, "y": 50}, "n": 50}],
     "batch": 10,               # words picked per target per round
     "order": "rotate",         # "rotate": random first round, then every round starts at the next target
                                # "shuffle": a fresh random order every round
     "occurrences": "all",      # "all": every matching phoneme in a chosen word; "one": one random occurrence
     "skip_altered_words": False,   # True: words already changed by an EARLIER rule are not candidates
     "seed": 7}                 # optional; otherwise the seed= of manipulate_with_quota is used

  targets[*]["old"] : a phoneme (or token id), or a list of them treated as one class, e.g. ["e", "ɛ"]
  targets[*]["new"] : the replacement, or "choices": {out: weight} for a random pick per word
                      (all occurrences in one word get the same pick)
  targets[*]["n"]   : number of words; a float < 1 means a fraction of that target's candidates (0.2 = 20 %)

A word = a stretch of the chunk between boundary characters (space / punctuation), counted as a TOKEN:
a word that occurs three times in the text is three candidates.
Every word is changed by AT MOST ONE target of the rule. The selection is made over the WHOLE text
(all chunks together), not per sentence.
"""
import random
from collections import Counter

import functions as F
from functions import _apply_rule, _scan, _build_stats, _norm, _norm_choices, BOUNDARY

__all__ = ["manipulate_with_quota", "select_quota_words"]


# ---------------------------------------------------------------------------------------------------
# selection: random round-robin + repair
# ---------------------------------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------------------------------
# one word_quota rule over all chunks
# ---------------------------------------------------------------------------------------------------
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


def _targets(rule):
    out = []
    for k, t in enumerate(rule["targets"]):
        olds = t["old"] if isinstance(t["old"], (list, tuple)) else [t["old"]]
        olds = [_norm(o) for o in olds]
        if "choices" in t:
            outs, ws = _norm_choices(t["choices"])
        else:
            outs, ws = [_norm(t["new"])], [1.0]
        out.append(dict(i=k, olds=olds, outs=outs, ws=ws, n=t["n"]))
    keys = [o for t in out for o in t["olds"]]
    assert len(keys) == len(set(keys)), f"word_quota: a phoneme appears in two targets: {keys}"
    return out


def _apply_quota(chunks, masks, origins, rule, protect_affricates, rng, counter, verbose=True):
    tg = _targets(rule)
    key2t = {o: t["i"] for t in tg for o in t["olds"]}
    occ_mode = rule.get("occurrences", "all")
    assert occ_mode in ("all", "one"), occ_mode
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

    # ---- what to write where ----
    plan = [dict() for _ in chunks]                     # chunk -> {pos: piece}
    chosen = {t["i"]: [] for t in tg}
    for w in sorted(owner):
        ti = owner[w]; t = tg[ti]
        piece = rng.choices(t["outs"], weights=t["ws"], k=1)[0]
        hits = occ[w][ti]
        if occ_mode == "one":
            hits = [rng.choice(hits)]
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
            msg = f"   {'/'.join(t['olds'])} → {'|'.join(t['outs'])}: {len(cand[ti])} candidates, " \
                  f"{len(chosen[ti])} of {quota[ti]} words changed"
            if ti in short:
                msg += f"   ⚠ {short[ti]} short: no assignment with one change per word can reach the quota"
            print(msg)
    entry = dict(label=_describe_quota(rule), kind="word_quota",
                 changed=sum(len(v) for v in chosen.values()),
                 groups=[dict(old="/".join(t["olds"]), total=len(cand[t["i"]]), draws=None,
                              rows=[dict(new=o, n=sum(c["new"] == o for c in chosen[t["i"]]), p=None, changed=True)
                                    for o in t["outs"]])
                         for t in tg],
                 words={"/".join(t["olds"]): chosen[t["i"]] for t in tg},     # the chosen words, per target
                 quota={"/".join(t["olds"]): quota[t["i"]] for t in tg},
                 shortfall={"/".join(tg[i]["olds"]): n for i, n in short.items()})
    return out_c, out_m, out_o, entry


def _describe_quota(rule):
    parts = []
    for t in _targets(rule):
        new = t["outs"][0] if len(t["outs"]) == 1 else "{" + " | ".join(
            f"{o} {w * 100:.0f}%" for o, w in zip(t["outs"], t["ws"])) + "}"
        n = t["n"]
        n = f"{n * 100:.0f}% of words" if isinstance(n, float) and n < 1 else f"{n} words"
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


# ---------------------------------------------------------------------------------------------------
# public entry point: drop-in replacement for functions.manipulate
# ---------------------------------------------------------------------------------------------------
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
