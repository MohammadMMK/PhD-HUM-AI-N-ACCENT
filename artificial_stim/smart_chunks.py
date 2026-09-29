import math, re

# Break cost AFTER a word: lower = better place to cut.
# Cutting at sentence end is free; mid-clause is expensive (the model adds
# phrase-final lengthening + falling intonation at every chunk end).
BREAK_COST = [
    (re.compile(r'[.!?…]+["”)]*$'), 0),     # sentence end
    (re.compile(r'[;:—]["”)]*$'),   1),     # semicolon / colon / dash
    (re.compile(r'[,)]["”]*$'),     3),     # comma
]
MID_CLAUSE = 10                              # plain space between two words


def _break_cost(word):
    for rx, c in BREAK_COST:
        if rx.search(word):
            return c
    return MID_CLAUSE


def smart_chunks(text, g2p_fn, vocab, lo=100, hi=200, target=150, hard_max=500,
                 sentence_split=None):
    """Cut `text` into chunks of ~lo..hi Kokoro tokens, only at word boundaries,
    preferring sentence ends > ; : — > commas > plain spaces.

    g2p_fn : str -> IPA string   (in your code: lambda s: g2p.g2p(s)[0])
    vocab  : Kokoro VOCAB dict   (tokens are counted exactly like _encode does)
    sentence_split : your one_sentence_chunks (so G2P always sees whole sentences)

    Returns (text_chunks, ipa_chunks), parallel lists.
    Use ipa_chunks for synthesis and " ".join(text_chunks) as the word labels.
    """
    ntok = lambda s: sum(c in vocab for c in s)
    sents = sentence_split(text) if sentence_split else [text]

    # 1) G2P each WHOLE sentence (keeps sentence-level pronunciation/context),
    #    then split text and IPA into parallel words.
    words = []                                     # (text_word, ipa_word, break_cost)
    for s in sents:
        ipa = g2p_fn(s).strip()
        tw, iw = s.split(), ipa.split(' ')
        if len(tw) == len(iw):                     # 1:1 -> we may cut inside the sentence
            for k, (t, p) in enumerate(zip(tw, iw)):
                words.append((t, p, 0 if k == len(tw) - 1 else _break_cost(t)))
        else:                                      # G2P merged/split words -> keep sentence whole
            print(f"⚠ word mismatch ({len(tw)} text vs {len(iw)} IPA), sentence kept unsplit: {s[:50]}…")
            words.append((s, ipa, 0))

    # 2) Dynamic programming: choose cut points minimizing
    #    length penalty (outside lo..hi, or far from target) + break penalty.
    def len_cost(n):
        if n > hard_max:  return math.inf
        if n < lo:        return ((lo - n) / 10) ** 2
        if n > hi:        return ((n - hi) / 10) ** 2
        return ((n - target) / (hi - lo)) ** 2      # small pull toward target

    N = len(words)
    best, prev = [0.0] + [math.inf] * N, [0] * (N + 1)
    for j in range(1, N + 1):                      # chunk = words[i:j]
        n = -1
        for i in range(j - 1, -1, -1):
            n += ntok(words[i][1]) + 1             # +1 for the joining space
            if n > hard_max: break
            c = best[i] + len_cost(n) + words[j - 1][2] * 2
            if c < best[j]:
                best[j], prev[j] = c, i

    cuts, j = [], N
    while j > 0:
        cuts.append((prev[j], j)); j = prev[j]
    cuts.reverse()

    text_chunks = [" ".join(w[0] for w in words[i:j]) for i, j in cuts]
    ipa_chunks  = [" ".join(w[1] for w in words[i:j]) for i, j in cuts]
    return text_chunks, ipa_chunks


def report(text_chunks, ipa_chunks, vocab):
    for k, (t, p) in enumerate(zip(text_chunks, ipa_chunks)):
        n = sum(c in vocab for c in p)
        flag = "" if 100 <= n <= 200 else "  <-- outside 100-200"
        print(f"{k:>3} {n:>4} tok | …{t[-45:]}{flag}")
