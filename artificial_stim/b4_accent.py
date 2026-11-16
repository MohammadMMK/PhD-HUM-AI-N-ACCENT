"""
b4_accent.py — B4: an in-between phoneme for EVERY occurrence of a phoneme in a text.

Put this file next to functions.py and accent_tools.py. Edit the SETTINGS below and run
    python b4_accent.py
or, in a notebook:
    from b4_accent import run
    run(weight_b=[0, 0.25, 0.5, 0.75, 1.0])        # any setting can be overridden as an argument

WHAT IT DOES (per sentence)
  A = the original IPA                          (… litˈalja …)
  B = the same IPA with every SUB_FROM -> SUB_TO (… lidˈalja …)   one-to-one: same number of tokens
  Both are encoded. Then, with w = WEIGHT_B:
  * step 10 (acoustic encoder output): the vectors of the substituted tokens become (1-w)·A + w·B
      STEP10_NEIGHBORS=True  -> the same mix is also applied to STEP10_WINDOW phonemes on each side
                                (they were re-encoded in B knowing their neighbour is now SUB_TO)
      STEP10_NEIGHBORS=False -> neighbours keep A's vectors (only the target token is mixed)
  * step 4 (PL-BERT output = prosody features):
      MIX_PROSODY=True  -> the same mix on the target ± PROSODY_WINDOW phonemes; durations, pitch and
                           energy are then PREDICTED from the mix (steps 5-8), so timing and voicing
                           move with the weight too (needed for t→d, which is mostly a voicing contrast)
      MIX_PROSODY=False -> prosody of the original text A (timing / pitch / energy unchanged)
  Every mix keeps the vector length (norm) by default, so a 50/50 mix is not a "weaker" vector.
  w = 0 is exactly the original; w = 1 with both mixes on is exactly the B text (with the windows' limits).

OUTPUT (OUT_DIR)
  b4_<from>to<to>_weights.html             ONE page with a player per weight (+ the original first):
                                           your accent_tools player (audio, words, phonemes with the
                                           changed ones marked, pitch/energy panel; dashed curves =
                                           prosody with the original phoneme, same timing)
  b4_<from>to<to>_w<weight>.wav            audio of every weight
  b4_reference_original.wav                the unmanipulated text (MAKE_REFERENCE)
  b4_ipa_<from>to<to>.txt                  original and substituted IPA of every sentence
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import accent_tools as AT
from accent_tools import load_story, manipulate_with_quota, make_alignment, block_changes, voiced_mask, export_player
from functions import model, VOCAB, BOUNDARY, SR, align_phonemes, _encode, _voice_pack

# =====================================================================================================
# SETTINGS
# =====================================================================================================
TEXT = """Nell'estate del duemila, Tito ha visitato l'Italia con tutta la sua famiglia.
E ha trovato tanti posti stupendi vicino al mare."""     # the text itself, or a path to a .txt file

VOICE = "if_sara"           # a name, a loaded tensor, or a (name, tensor) pair (like render_condition)

SUB_FROM = "t"              # substitution SUB_FROM -> SUB_TO, applied to EVERY occurrence in the text
SUB_TO = "d"                # must be one-to-one (same number of symbols), e.g. t->d, t->θ, k->ɡ, e->ø

WEIGHT_B = [0, 0.25, 0.5, 0.75, 1.0]   # weight(s) of SUB_TO (0 = original, 1 = full substitution);
                                       # one number or a list -> ALL weights go into ONE html page

STEP10_NEIGHBORS = True     # also mix the neighbours' step-10 vectors
STEP10_WINDOW = 2           # phonemes on each side (stress marks / spaces are crossed, not counted), or "all"

MIX_PROSODY = True          # also mix the prosody features (step 4) -> durations, pitch, energy follow
PROSODY_WINDOW = 2          # phonemes on each side, or "all"

KEEP_NORM = True            # keep the vector length when mixing
MAKE_REFERENCE = True       # also include the original text (no manipulation) as the first player
SEPARATE_HTML = False       # also write one html per weight (the combined page is always written)
NORMALIZE = None            # IPA clean-up of load_story (None = its default ɪ->i; {} = off)
SPEED = 1.0
SEED = 0                    # decoder noise; the same seed is used for every weight -> comparable outputs
OUT_DIR = "b4_output"

STRESS = {"ˈ", "ˌ"}


# =====================================================================================================
# B4 CORE
# =====================================================================================================
def _tokens(ps):
    """char index -> token index (0 = start pad), and the token labels (dropped chars have no token)."""
    kept = [j for j, c in enumerate(ps) if c in VOCAB]
    return {j: k + 1 for k, j in enumerate(kept)}, ["<s>"] + [ps[j] for j in kept] + ["</s>"]


def _window(labels, centers, k):
    """Token positions of the centers plus k PHONEMES on each side (stress marks, spaces and punctuation
    on the way are included but not counted). k="all" -> every token, including the start/end pads."""
    if k == "all":
        return list(range(len(labels)))
    pos = set(centers)
    for c in centers:
        for step in (-1, 1):
            i, n = c, 0
            while n < k:
                i += step
                if i <= 0 or i >= len(labels) - 1:
                    break
                pos.add(i)
                if labels[i] not in STRESS and labels[i] not in BOUNDARY:
                    n += 1
    return sorted(pos)


def blend(xA, xB, w, positions, keep_norm=True):
    """(1-w)·A + w·B on the given token positions (last axis); vectors along axis 1. Returns a copy."""
    out = xA.clone()
    if not positions or w == 0:
        return out
    p = torch.as_tensor(positions, device=xA.device)
    if w == 1:                                  # exact copy (avoids float noise from the norm rescaling)
        out[..., p] = xB[..., p]
        return out
    a, b = xA[..., p], xB[..., p]
    v = (1 - w) * a + w * b
    if keep_norm:
        target = (1 - w) * a.norm(dim=1, keepdim=True) + w * b.norm(dim=1, keepdim=True)
        v = v * target / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
    out[..., p] = v
    return out


@torch.no_grad()
def render_sentence(psA, psB, targets_tok, pack, w, step10_pos, pros_pos, mix_prosody=True,
                    keep_norm=True, speed=1.0, seed=0):
    """One sentence. targets_tok / step10_pos / pros_pos are token positions.
    Returns audio (numpy), durations (list incl. pads), F0, N, and F0/N of the ORIGINAL prosody
    on the same timing (for the dashed reference curves)."""
    eA, eB = _encode(psA), _encode(psB)
    assert eA["n"] == eB["n"], "B4 needs the same number of tokens in A and B"
    ref_s = pack[len(psA) - 1].to(model.device)          # same length -> same voice row for A and B
    s = ref_s[:, 128:]
    L, tm = eA["L"], eA["tm"]

    # steps 4-6: prosody features (mixed or original) -> durations
    d_en = blend(eA["d_en"], eB["d_en"], w, pros_pos, keep_norm) if mix_prosody else eA["d_en"].clone()
    d = model.predictor.text_encoder(d_en, s, L, tm)
    x, _ = model.predictor.lstm(d)
    dur = torch.round(torch.sigmoid(model.predictor.duration_proj(x)).sum(-1) / speed).clamp(min=1).long()
    dur = dur.reshape(-1).tolist()

    # steps 7-8: alignment, pitch and energy
    aln = make_alignment(psB, dur)
    F0, N = model.predictor.F0Ntrain(d.transpose(-1, -2) @ aln, s)
    dA = model.predictor.text_encoder(eA["d_en"], s, L, tm)          # original prosody, same timing
    F0a, Na = model.predictor.F0Ntrain(dA.transpose(-1, -2) @ aln, s)

    # step 10: mixed acoustic vectors; steps 11-12: decode
    t_en = blend(eA["t_en"], eB["t_en"], w, step10_pos, keep_norm)
    torch.manual_seed(seed)
    audio = model.decoder(t_en @ aln, F0, N, ref_s[:, :128]).squeeze().cpu().numpy()
    np_ = lambda z: z.squeeze().cpu().numpy()
    return audio, dur, np_(F0), np_(N), np_(F0a), np_(Na)


# =====================================================================================================
# PIPELINE: text -> substitution -> render every sentence -> player result dict
# =====================================================================================================
def _prepare(text, sub_from, sub_to, normalize):
    story = load_story(text, name="b4", normalize=normalize, verbose=False)
    ipa = [s for b in story["blocks"] for s in b["ipa"]]
    words = [w for b in story["blocks"] for s in b["spelling"] for w in s]
    AT._check_tokens([sub_to], "SUB_TO")
    rule = {"kind": "substitute", "map": {sub_from: sub_to}}
    manip, masks, origins = manipulate_with_quota(ipa, [rule], return_mask=True, return_origin=True,
                                                  verbose=False)
    for i, (a, b) in enumerate(zip(ipa, manip)):
        na, nb = sum(c in VOCAB for c in a), sum(c in VOCAB for c in b)
        assert na == nb, (f"sentence {i}: {sub_from}->{sub_to} changes the number of tokens ({na} -> {nb}). "
                          f"B4 needs a one-to-one substitution (same number of symbols).")
    n_sub = sum(sum(m) for m in masks)
    print(f"{len(ipa)} sentences, {len(words)} words, {n_sub} × {sub_from}→{sub_to}")
    if n_sub == 0:
        print(f"⚠ '{sub_from}' does not occur in the text: nothing to substitute")
    return story, ipa, manip, masks, [rule], words


def _render_text(ipa, manip, masks, voice, w, settings, show_original=False):
    """Render all sentences with weight w. show_original=True labels the player with the original IPA
    (used for the reference)."""
    pack, _ = _voice_pack(voice)
    st = settings
    segments, seg_words, seg_altered = [], [], []
    T, F0o, F0r, Ne, Nr, sents, audios = [], [], [], [], [], [], []
    t0, word_off, n_targets = 0.0, 0, 0
    for ci, (psA, psB, mask) in enumerate(zip(ipa, manip, masks)):
        if not psA.strip():
            continue
        tok_of, labels = _tokens(psB)
        targets = [tok_of[j] for j, m in enumerate(mask) if m and j in tok_of]
        n_targets += len(targets)
        s10 = _window(labels, targets, st["step10_window"]) if st["step10_neighbors"] else targets
        pp = _window(labels, targets, st["prosody_window"])
        a, dur, f0, n, f0a, na = render_sentence(psA, psB, targets, pack, w, s10, pp, st["mix_prosody"],
                                                 st["keep_norm"], st["speed"], st["seed"] + ci)
        ps_show, mask_show = (psA, None) if show_original else (psB, mask)
        segs, widx, altd = align_phonemes(ps_show, dur, n_samples=len(a), sr=SR, altered_mask=mask_show)
        segments += [(lab, s + t0, e + t0, info) for lab, s, e, info in segs]
        seg_words += [None if x is None else x + word_off for x in widx]
        seg_altered += altd
        real = [x for x in widx if x is not None]
        word_off += (max(real) + 1) if real else 0
        k = len(f0)
        T += [t0 + np.arange(k) * (len(a) / k) / SR, np.array([t0 + len(a) / SR])]
        F0o += [np.where(voiced_mask(f0), f0, np.nan), np.array([np.nan])]
        F0r += [np.where(voiced_mask(f0a), f0a, np.nan), np.array([np.nan])]
        Ne += [n, np.array([np.nan])]
        Nr += [na, np.array([np.nan])]
        sents.append(dict(chunk=ci, start=t0, end=t0 + len(a) / SR, donor=None, donor_idx=0))
        audios.append(a)
        t0 += len(a) / SR
    label = "original" if show_original else f"{SUB_LABEL(st)} w={w:g}"
    pros = dict(t=np.concatenate(T), f0=np.concatenate(F0o), f0_t=np.concatenate(F0r),
                n=np.concatenate(Ne), n_t=np.concatenate(Nr), chunks=[], sentences=sents,
                target=_name(voice), donors=[label], what=["B4"],
                modes=dict(f0="-", energy="-", dur="-", dur_scope="-", scope="-", assign="-",
                           seed=st["seed"], donor_text="-"))
    result = dict(block="all", voice=_name(voice), ref_voice=None, audio=np.concatenate(audios),
                  segments=segments, seg_words=seg_words, seg_altered=seg_altered, pros=pros)
    return result, n_targets


def SUB_LABEL(st):
    return f"{st['sub_from']}→{st['sub_to']}"


def _name(voice):
    return voice if isinstance(voice, str) else (voice[0] if isinstance(voice, tuple) else "voice")


def _info(st, w):
    s10 = (f"target + {st['step10_window']} phonemes each side" if st["step10_neighbors"]
           and st["step10_window"] != "all" else "all tokens" if st["step10_neighbors"] else "target only")
    pr = (f"mixed, target + {st['prosody_window']} phonemes each side" if st["mix_prosody"]
          else "original (not mixed)")
    return (f"B4 {SUB_LABEL(st)}, weight of {st['sub_to']} = {w:g} · step 10: {s10} · prosody (step 4): {pr}"
            f" · norm kept: {st['keep_norm']} · seed {st['seed']} · dashed = prosody with the original "
            f"{st['sub_from']}, same timing")


def _player_fragment(result, words, title, rules, stats, info):
    """The accent_tools player as an HTML fragment (same steps as export_player, without writing a page),
    with our own description line in the pitch panel and unique element ids, so several players can
    live on one page."""
    F, orig_payload = AT.F, AT._pros_payload
    original = F._TEMPLATE
    try:
        F._TEMPLATE = AT._prosody_template()
        text = " ".join(w.replace(" ", "_") or "?" for w in words) if words else None
        html = AT.audiovisualize_interactive(result["audio"], result["segments"], sr=SR, out_html=None,
                                             seg_words=result["seg_words"], text=text,
                                             seg_altered=result["seg_altered"], title=title, rules=rules,
                                             stats=stats)
    finally:
        F._TEMPLATE = original
    payload = orig_payload(result["pros"])
    payload["info"] = info
    return AT._uniquify_ids(html.replace("__PROS__", json.dumps(payload)))


def _page(title, intro, sections):
    """sections: [(anchor, heading, fragment)] -> a full HTML page with a link bar on top."""
    from html import escape
    nav = " ".join(f'<a href="#{a}">{escape(h)}</a>' for a, h, _ in sections)
    body = "\n".join(f'<section id="{a}"><h2>{escape(h)}</h2>\n{frag}\n</section>' for a, h, frag in sections)
    css = ("body{font-family:system-ui,sans-serif;max-width:1120px;margin:24px auto;padding:0 16px;color:#222}"
           "h1{font-size:20px;margin:0 0 6px}h2{font-size:16px;margin:0 0 8px;color:#333}"
           ".intro{font-size:13px;color:#555;margin:0 0 12px;line-height:1.5}"
           "nav{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #e5e7eb;z-index:5}"
           "nav a{display:inline-block;margin:0 6px 4px 0;padding:4px 10px;border:1px solid #d1d5db;"
           "border-radius:6px;text-decoration:none;color:#1d4ed8;font-size:13px}"
           "section{margin:22px 0 34px;padding-top:6px;border-top:1px solid #f1f1f1}")
    return ('<!DOCTYPE html>\n<html><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{escape(title)}</title><style>{css}</style></head>\n<body>\n'
            f'<h1>{escape(title)}</h1>\n<p class="intro">{intro}</p>\n<nav>{nav}</nav>\n'
            f'{body}\n</body></html>\n')


def _export_single(path, title, frag):
    Path(path).write_text(_page(title, "", [("p", title, frag)]), encoding="utf-8")
    return Path(path)


def run(text=TEXT, voice=VOICE, sub_from=SUB_FROM, sub_to=SUB_TO, weight_b=WEIGHT_B,
        step10_neighbors=STEP10_NEIGHBORS, step10_window=STEP10_WINDOW, mix_prosody=MIX_PROSODY,
        prosody_window=PROSODY_WINDOW, keep_norm=KEEP_NORM, make_reference=MAKE_REFERENCE,
        separate_html=SEPARATE_HTML, normalize=NORMALIZE, speed=SPEED, seed=SEED, out_dir=OUT_DIR):
    st = dict(sub_from=sub_from, sub_to=sub_to, step10_neighbors=step10_neighbors,
              step10_window=step10_window, mix_prosody=mix_prosody, prosody_window=prosody_window,
              keep_norm=keep_norm, speed=speed, seed=seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    story, ipa, manip, masks, rules, words = _prepare(text, sub_from, sub_to, normalize)
    stats = block_changes(dict(ipa=ipa, manip=manip), label="Substituted (B4 mix)")
    tag = f"{sub_from}to{sub_to}"
    written = []

    with open(out / f"b4_ipa_{tag}.txt", "w", encoding="utf-8") as f:
        for a, b in zip(ipa, manip):
            f.write(f"original   : {a}\nsubstituted: {b}\n\n")
    written.append(out / f"b4_ipa_{tag}.txt")

    from html import escape
    weights = [float(x) for x in (weight_b if isinstance(weight_b, (list, tuple)) else [weight_b])]
    sections = []                                     # (anchor, heading, player fragment)
    if make_reference:
        res, _ = _render_text(ipa, manip, masks, voice, 0.0, st, show_original=True)
        frag = _player_fragment(res, words, f"Original · voice {_name(voice)}", None, None,
                                "original text, no manipulation (solid and dashed curves are the same)")
        sections.append(("original", "Original (no manipulation)", frag))
        sf.write(out / "b4_reference_original.wav", res["audio"], SR)
        written.append(out / "b4_reference_original.wav")

    for w in weights:
        res, n_t = _render_text(ipa, manip, masks, voice, w, st)
        name = f"b4_{tag}_w{w:.2f}"
        title = f"B4 {SUB_LABEL(st)} · weight of {sub_to} = {w:g} · voice {_name(voice)}"
        frag = _player_fragment(res, words, title, rules, stats, _info(st, w))
        sections.append((f"w{w:.2f}".replace(".", "_"), f"weight of {sub_to} = {w:g}", frag))
        sf.write(out / f"{name}.wav", res["audio"], SR)
        written.append(out / f"{name}.wav")
        if separate_html:
            written.append(_export_single(out / f"{name}.html", title, frag))
        print(f"w={w:g}: {n_t} {sub_from}→{sub_to} tokens mixed, {len(res['audio']) / SR:.1f} s")

    page_title = f"B4 {SUB_LABEL(st)} · voice {_name(voice)} · weights {', '.join(f'{w:g}' for w in weights)}"
    intro = escape(_info(st, 0).split(" · ", 1)[1]) + \
        f"<br>{sum(sum(m) for m in masks)} × {escape(SUB_LABEL(st))} in {len(ipa)} sentences. " \
        "Every player: click to play from there, the changed phonemes are marked; " \
        "pitch panel: solid = this weight, dashed = prosody with the original phoneme."
    page = out / f"b4_{tag}_weights.html"
    page.write_text(_page(page_title, intro, sections), encoding="utf-8")
    written.insert(0, page)

    print(f"written to {out}/ :")
    for p in written:
        print("   ", Path(p).name)
    return written


if __name__ == "__main__":
    run()
