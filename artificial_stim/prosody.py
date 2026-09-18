"""
prosody.py — prosody extraction and transfer for Kokoro (StyleTTS2).

WHAT THIS IS FOR
    Kokoro keeps three things separate inside one 256-d voice vector `ref_s`:

        ref_s[:, 128:]  -> "s", the prosody style: drives duration prediction,
                           the F0 contour and the energy envelope
        ref_s[:, :128]  -> the decoder style: timbre / speaker identity

    Between the two there are three transferable curves:

        durations  [T_tok]         how many 600-sample frames each phoneme holds
        F0         [1, 2*T_frames] pitch contour
        N          [1, 2*T_frames] energy / loudness envelope

    This module lets you pull those three out of one (text, voice) pair WITHOUT
    running the vocoder, then force any subset of them into another (text, voice)
    generation.

WHY EXTRACTION IS CHEAP
    A full Kokoro forward is:

        bert -> bert_encoder -> predictor.text_encoder -> durations
             -> alignment -> F0Ntrain          <- prosody ends here
             -> text_encoder -> asr
             -> decoder                        <- the vocoder, most of the cost

    `extract()` stops at the arrow, so it skips the decoder and the ASR text
    encoder entirely. Everything it returns is what the full pass would have
    produced, bit for bit.

TYPICAL USE
    donor = extract(model, ps, "im_nicola")             # cheap, no audio
    audio = generate(model, ps, "if_sara", prosody=donor,
                     duration="copy", f0="match", energy="copy")

GLOSSARY OF MODES (see `generate`)
    duration : "copy"    reuse the donor's per-phoneme durations (same text only)
               "scale"   predict this text's own durations, then stretch them so
                         the total length matches the donor (works on new text)
               "predict" ignore the donor's timing
    f0       : "match"   donor contour, shifted into this voice's own register
               "copy"    donor contour verbatim, in absolute Hz
               "predict" this voice's own pitch
    energy   : "copy" | "predict"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as nnf

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

SR = 24_000              # Kokoro output sample rate
FRAME_SAMPLES = 600      # samples per duration frame  -> 40 frames/second
F0_UPSAMPLE = 2          # F0Ntrain upsamples once, so len(F0) == 2 * n_frames
MAX_TOKENS = 510         # model limit, excluding the two BOS/EOS zeros

VoiceLike = Union[str, Path, torch.Tensor]


# --------------------------------------------------------------------------- #
# 1. voice packs
# --------------------------------------------------------------------------- #

def load_voice(voice: VoiceLike,
               pipeline=None,
               custom_dir: Union[str, Path] = "custom_voices") -> torch.Tensor:
    """Return a voice pack tensor, shape [N, 1, 256].

    Accepts, in order of preference:
      - a tensor (returned untouched, so you can pass an already-loaded pack)
      - "nicola_50" / "nicola_50.pt" / a path -> loaded from `custom_dir`
      - any other name -> delegated to `pipeline.load_voice` (the built-in voices)
    """
    if isinstance(voice, torch.Tensor):
        return voice

    name = str(voice)
    path = Path(name)
    if path.suffix != ".pt":
        path = Path(custom_dir) / f"{name}.pt"
    elif not path.is_absolute() and not path.exists():
        path = Path(custom_dir) / path.name

    if path.exists():
        return torch.load(path, weights_only=True)
    if pipeline is not None:
        return pipeline.load_voice(name)
    raise FileNotFoundError(
        f"voice {name!r}: no file at {path} and no pipeline given to fall back on"
    )


def voice_ref(pack: torch.Tensor, phonemes: str) -> torch.Tensor:
    """Pick the [1, 256] row of a pack for a phoneme string.

    Mirrors KPipeline, which indexes the pack by `len(ps) - 1`, but clamps so a
    short custom pack (fewer than 510 rows) never goes out of bounds.
    An already-sliced [1, 256] tensor is passed through unchanged.
    """
    if pack.dim() == 2:                       # already a single reference row
        return pack
    i = min(max(len(phonemes) - 1, 0), pack.shape[0] - 1)
    ref = pack[i]
    return ref if ref.dim() == 2 else ref.unsqueeze(0)


# --------------------------------------------------------------------------- #
# 2. tokenisation
# --------------------------------------------------------------------------- #

@dataclass
class Tokens:
    """The model's view of a phoneme string."""
    ids: torch.LongTensor     # [1, T_tok], includes the leading/trailing 0
    phonemes: str             # only the characters that survived the vocab
    dropped: List[tuple]      # [(position, char), ...] for debugging

    @property
    def n_tokens(self) -> int:
        return int(self.ids.shape[-1])        # == len(self.phonemes) + 2


def encode(model, phonemes: str, warn: bool = True) -> Tokens:
    """Phoneme string -> Tokens. Silently drops characters outside the vocab,
    exactly like KPipeline does, but tells you which ones."""
    kept, ids, dropped = [], [], []
    for pos, ch in enumerate(phonemes):
        tid = model.vocab.get(ch)
        if tid is None:
            dropped.append((pos, ch))
        else:
            kept.append(ch)
            ids.append(tid)

    if len(ids) > MAX_TOKENS:
        if warn:
            print(f"⚠ {len(ids)} tokens > {MAX_TOKENS}, truncating")
        ids, kept = ids[:MAX_TOKENS], kept[:MAX_TOKENS]
    if dropped and warn:
        preview = ", ".join(f"U+{ord(c):04X} {c!r}@{p}" for p, c in dropped[:6])
        print(f"⚠ {len(dropped)} phoneme(s) not in vocab, dropped: {preview}")

    return Tokens(
        ids=torch.LongTensor([[0, *ids, 0]]).to(model.device),
        phonemes="".join(kept),
        dropped=dropped,
    )


# --------------------------------------------------------------------------- #
# 3. the forward graph, in reusable pieces
#    (each mirrors the corresponding lines of KModel.forward_with_tokens)
# --------------------------------------------------------------------------- #

def _lengths_and_mask(input_ids: torch.Tensor):
    dev = input_ids.device
    lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1],
                         device=dev, dtype=torch.long)
    mask = (torch.arange(int(lengths.max()), device=dev)
            .unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths))
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return lengths, mask


def _prosody_hidden(model, input_ids, lengths, mask, s):
    """BERT -> prosody text encoder. Returns `d` [1, T_tok, H]."""
    bert_dur = model.bert(input_ids, attention_mask=(~mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    return model.predictor.text_encoder(d_en, s, lengths, mask)


def _predict_durations(model, d, speed: float = 1.0) -> torch.Tensor:
    """The duration head. Returns [T_tok] long, one frame count per token."""
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(dim=-1) / speed
    return torch.round(duration).clamp(min=1).long().reshape(-1)


def _alignment(durations: torch.Tensor, device) -> torch.Tensor:
    """Hard monotonic alignment [1, T_tok, T_frames].

    Column f holds a 1 in row t when frame f belongs to token t, so `X @ aln`
    upsamples any token-rate tensor to frame rate by holding each column.
    This single matrix is what carries the rhythm: swap the durations and every
    downstream tensor, including the final sample count, follows.
    """
    durations = durations.to(device).clamp(min=1).long().reshape(-1)
    n_tok = durations.shape[0]
    idx = torch.repeat_interleave(torch.arange(n_tok, device=device), durations)
    aln = torch.zeros((n_tok, idx.shape[0]), device=device)
    aln[idx, torch.arange(idx.shape[0], device=device)] = 1
    return aln.unsqueeze(0)


# --------------------------------------------------------------------------- #
# 4. the prosody container
# --------------------------------------------------------------------------- #

@dataclass
class Prosody:
    """Everything transferable about one (text, voice, speed) triple.

    Tensors are stored on CPU so the object is cheap to keep around and can be
    torch.save'd directly.
    """
    phonemes: str                 # the vocab-filtered string these belong to
    durations: torch.Tensor       # [T_tok] long, includes BOS/EOS
    f0: torch.Tensor              # [1, 2*T_frames]
    energy: torch.Tensor          # [1, 2*T_frames]
    speed: float = 1.0
    voice: Optional[str] = None

    # -- derived ---------------------------------------------------------- #
    @property
    def n_tokens(self) -> int:
        return int(self.durations.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.durations.sum())

    @property
    def seconds(self) -> float:
        return self.n_frames * FRAME_SAMPLES / SR

    def voiced_f0(self, threshold: float = 1.0) -> torch.Tensor:
        return self.f0[self.f0 > threshold]

    def median_f0(self, threshold: float = 1.0) -> float:
        v = self.voiced_f0(threshold)
        return float(v.median()) if v.numel() else 0.0

    # -- io --------------------------------------------------------------- #
    def save(self, path: Union[str, Path]) -> None:
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Prosody":
        return cls(**torch.load(path, weights_only=False))

    def __repr__(self) -> str:
        return (f"Prosody(voice={self.voice!r}, {self.n_tokens} tokens, "
                f"{self.n_frames} frames = {self.seconds:.2f}s, "
                f"median F0 {self.median_f0():.1f}, speed {self.speed})")


# --------------------------------------------------------------------------- #
# 5. extraction — no decoder, no audio
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract(model,
            phonemes: str,
            voice: VoiceLike,
            speed: float = 1.0,
            pipeline=None,
            custom_dir: Union[str, Path] = "custom_voices") -> Prosody:
    """Durations, F0 and energy for `phonemes` in `voice`, without synthesising.

    Skips `model.text_encoder` and `model.decoder`, which is where nearly all of
    the compute sits, so this is several times faster than a full generation and
    allocates no waveform.
    """
    pack = load_voice(voice, pipeline=pipeline, custom_dir=custom_dir)
    tok = encode(model, phonemes)
    ref_s = voice_ref(pack, phonemes).to(model.device)
    s = ref_s[:, 128:]

    lengths, mask = _lengths_and_mask(tok.ids)
    d = _prosody_hidden(model, tok.ids, lengths, mask, s)
    durations = _predict_durations(model, d, speed)

    aln = _alignment(durations, model.device)
    en = d.transpose(-1, -2) @ aln
    f0, energy = model.predictor.F0Ntrain(en, s)

    return Prosody(
        phonemes=tok.phonemes,
        durations=durations.cpu(),
        f0=f0.detach().cpu(),
        energy=energy.detach().cpu(),
        speed=speed,
        voice=str(voice) if not isinstance(voice, torch.Tensor) else None,
    )


def extract_chunks(model, phoneme_chunks: Sequence[str], voice: VoiceLike,
                   speed: float = 1.0, **kw) -> List[Prosody]:
    """`extract` over a list of chunks (e.g. text_to_ipa_chunks output)."""
    return [extract(model, ps, voice, speed=speed, **kw)
            for ps in phoneme_chunks if ps.strip()]


# --------------------------------------------------------------------------- #
# 6. curve surgery
# --------------------------------------------------------------------------- #

def resample_curve(curve: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linearly resample a [1, L] curve to [1, target_len]."""
    if curve.shape[-1] == target_len:
        return curve
    return nnf.interpolate(curve.unsqueeze(1).float(), size=target_len,
                           mode="linear", align_corners=True).squeeze(1)


def rescale_durations(durations: torch.Tensor, target_total: int,
                      min_dur: int = 1) -> torch.Tensor:
    """Stretch/squeeze per-token durations so they sum to exactly `target_total`.

    Uses largest-remainder rounding, so the total is exact rather than off by a
    few frames, and no token is ever driven below `min_dur`.
    """
    dur = durations.reshape(-1).long().clamp(min=min_dur)
    n = dur.numel()
    target_total = max(int(target_total), n * min_dur)

    scaled = dur.float() * (target_total / float(dur.sum()))
    out = scaled.floor().long().clamp(min=min_dur)
    remainder = scaled - scaled.floor()
    diff = target_total - int(out.sum())

    if diff > 0:                                   # hand out the spare frames
        order = torch.argsort(remainder, descending=True)
        for k in range(diff):
            out[order[k % n]] += 1
    elif diff < 0:                                 # take frames back
        order = torch.argsort(remainder)           # smallest remainder loses first
        max_steps = n * (abs(diff) + 1)
        k = 0
        while diff < 0 and k < max_steps:
            i = int(order[k % n])
            if out[i] > min_dur:
                out[i] -= 1
                diff += 1
            k += 1
    return out


def match_register(donor_f0: torch.Tensor,
                   target_f0: torch.Tensor,
                   mode: str = "median",
                   threshold: float = 1.0) -> torch.Tensor:
    """Move a donor pitch contour into the target voice's own register.

    "median"   : log-domain median shift. Keeps the contour shape exactly and
                 only moves it up/down. Safe default.
    "mean_std" : also rescales the log-F0 spread, so a monotone donor stops
                 flattening an expressive target (and vice versa).
    "none"     : passthrough — the contour stays in the donor's absolute Hz,
                 which is what makes a female voice sound male.
    """
    if mode == "none":
        return donor_f0

    dv, tv = donor_f0 > threshold, target_f0 > threshold
    if not bool(dv.any()) or not bool(tv.any()):
        return donor_f0

    d_log, t_log = donor_f0[dv].log(), target_f0[tv].log()
    out = donor_f0.clone()

    if mode == "median":
        out[dv] = (d_log + (t_log.median() - d_log.median())).exp()
    elif mode == "mean_std":
        d_std = d_log.std().clamp(min=1e-4)
        t_std = t_log.std().clamp(min=1e-4)
        out[dv] = (((d_log - d_log.mean()) * (t_std / d_std)) + t_log.mean()).exp()
    else:
        raise ValueError(f"unknown f0_match mode {mode!r}")
    return out


# --------------------------------------------------------------------------- #
# 7. generation with forced prosody
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate(model,
             phonemes: str,
             voice: VoiceLike,
             *,
             speed: float = 1.0,
             prosody: Optional[Prosody] = None,
             duration: str = "auto",       # copy | scale | predict | auto
             f0: str = "predict",          # match  | copy | predict
             energy: str = "predict",      # copy   | predict
             f0_match: str = "median",     # median | mean_std | none
             f0_weight: float = 1.0,
             energy_weight: float = 1.0,
             pipeline=None,
             custom_dir: Union[str, Path] = "custom_voices",
             return_prosody: bool = False):
    """Synthesise `phonemes` in `voice`, optionally forcing a donor's prosody.

    With `prosody=None` this is an ordinary generation and every mode is ignored.

    duration
        "copy"    inject the donor's per-token durations. Requires the same
                  token count, i.e. the same phoneme string.
        "scale"   predict this text's durations, then rescale them so the total
                  duration matches the donor. Use this for DIFFERENT text.
        "auto"    "copy" when the token counts match, "scale" when they don't.
        "predict" ignore donor timing.
        `speed` is applied only when durations are predicted — copied durations
        already have the donor's speed baked in.

    f0 / energy
        Frame-level curves, so they are resampled to whatever frame count the
        chosen durations produce. That makes them usable across different text,
        though the further the two texts diverge the less the contour means.
        `f0_weight` / `energy_weight` blend donor and predicted curves
        (1.0 = donor only, 0.5 = halfway, 0.0 = predicted only).

    Returns float32 numpy audio, or (audio, Prosody) when return_prosody=True.
    """
    pack = load_voice(voice, pipeline=pipeline, custom_dir=custom_dir)
    tok = encode(model, phonemes)
    ref_s = voice_ref(pack, phonemes).to(model.device)
    s = ref_s[:, 128:]
    dev = model.device

    lengths, mask = _lengths_and_mask(tok.ids)
    d = _prosody_hidden(model, tok.ids, lengths, mask, s)

    # ---- durations ------------------------------------------------------- #
    mode = duration
    if prosody is None:
        mode = "predict"
    elif mode == "auto":
        mode = "copy" if prosody.n_tokens == tok.n_tokens else "scale"

    if mode == "copy":
        if prosody.n_tokens != tok.n_tokens:
            raise ValueError(
                f'duration="copy" needs matching token counts: donor has '
                f'{prosody.n_tokens}, this text has {tok.n_tokens}. '
                f'Use duration="scale" for different text.')
        durations = prosody.durations.to(dev)
    elif mode == "scale":
        durations = rescale_durations(_predict_durations(model, d, 1.0),
                                      prosody.n_frames)
    elif mode == "predict":
        durations = _predict_durations(model, d, speed)
    else:
        raise ValueError(f"unknown duration mode {mode!r}")

    # ---- frame-level prosody --------------------------------------------- #
    aln = _alignment(durations, dev)
    en = d.transpose(-1, -2) @ aln
    f0_pred, n_pred = model.predictor.F0Ntrain(en, s)

    f0_out, n_out = f0_pred, n_pred
    if prosody is not None and f0 != "predict":
        donor = resample_curve(prosody.f0.to(dev), f0_pred.shape[-1])
        if f0 == "match":
            donor = match_register(donor, f0_pred, mode=f0_match)
        elif f0 != "copy":
            raise ValueError(f"unknown f0 mode {f0!r}")
        w = float(f0_weight)
        f0_out = donor if w >= 1.0 else (1 - w) * f0_pred + w * donor

    if prosody is not None and energy != "predict":
        if energy != "copy":
            raise ValueError(f"unknown energy mode {energy!r}")
        donor = resample_curve(prosody.energy.to(dev), n_pred.shape[-1])
        w = float(energy_weight)
        n_out = donor if w >= 1.0 else (1 - w) * n_pred + w * donor

    # ---- content + vocoder ------------------------------------------------ #
    t_en = model.text_encoder(tok.ids, lengths, mask)
    asr = t_en @ aln
    audio = model.decoder(asr, f0_out, n_out, ref_s[:, :128]).squeeze()
    audio = audio.detach().cpu().numpy().astype(np.float32)

    if not return_prosody:
        return audio
    used = Prosody(phonemes=tok.phonemes, durations=durations.cpu(),
                   f0=f0_out.detach().cpu(), energy=n_out.detach().cpu(),
                   speed=speed,
                   voice=str(voice) if not isinstance(voice, torch.Tensor) else None)
    return audio, used


def generate_chunks(model, phoneme_chunks: Sequence[str], voice: VoiceLike,
                    prosody: Optional[Sequence[Prosody]] = None, **kw) -> np.ndarray:
    """`generate` over a list of chunks, concatenated.

    `prosody` is a list parallel to `phoneme_chunks` (what extract_chunks gives
    you); pass None to generate normally.
    """
    chunks = [ps for ps in phoneme_chunks if ps.strip()]
    if prosody is not None and len(prosody) != len(chunks):
        raise ValueError(f"{len(prosody)} prosody entries for {len(chunks)} chunks")
    pieces = [generate(model, ps, voice,
                       prosody=None if prosody is None else prosody[i], **kw)
              for i, ps in enumerate(chunks)]
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 8. the multi-voice convenience wrapper
# --------------------------------------------------------------------------- #

def synth_matched(model,
                  phoneme_chunks: Sequence[str],
                  donor_voice: VoiceLike,
                  voices: Dict[str, VoiceLike],
                  speed: float = 1.0,
                  duration: str = "auto",
                  f0: str = "predict",
                  energy: str = "predict",
                  render_donor: bool = True,
                  **kw):
    """Donor track plus one prosody-matched track per voice in `voices`.

    Every returned track is sample-aligned with the donor, chunk by chunk, so
    they can be stacked, A/B'd or used as dubbing alternates.

    Returns (donor_audio_or_None, {name: audio}, [Prosody per chunk]).
    """
    chunks = [ps for ps in phoneme_chunks if ps.strip()]
    donor_prosody = extract_chunks(model, chunks, donor_voice, speed=speed, **kw)

    donor_audio = None
    if render_donor:
        # copy/copy/copy on the donor's own voice reproduces the normal
        # generation exactly — same durations, same F0, same N.
        donor_audio = np.concatenate([
            generate(model, ps, donor_voice, prosody=p, duration="copy",
                     f0="copy", energy="copy", **kw)
            for ps, p in zip(chunks, donor_prosody)])

    tracks = {}
    for name, v in voices.items():
        tracks[name] = np.concatenate([
            generate(model, ps, v, prosody=p, speed=speed, duration=duration,
                     f0=f0, energy=energy, **kw)
            for ps, p in zip(chunks, donor_prosody)])

    return donor_audio, tracks, donor_prosody
