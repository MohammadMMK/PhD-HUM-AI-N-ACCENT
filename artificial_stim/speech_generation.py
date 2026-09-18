"""
speech_generation.py — Kokoro synthesis, plain or with forced prosody.

Companion to text_manipulation.py: that module decides WHAT is said (phonemes),
this one decides HOW it is said (timing, pitch, loudness) and produces audio.

    synthesize(ipa, voice)                     plain, fast generation
    extract_prosody(ipa, voice)                durations + F0 + energy, no audio
    synthesize_forced(ipa, voice, donor)       generation with a donor's prosody
    synthesize_pair(original, manipulated, v)  both of the above, as one stimulus pair

WHY FORCING
    A manipulated stimulus should differ from its baseline in the manipulated
    segments and nowhere else. Left alone, Kokoro re-predicts the rhythm and the
    pitch contour for the new phoneme string, so /p/ -> /b/ also moves every
    following word. Forcing makes the baseline generation the reference: the
    manipulated version reuses its per-phoneme durations and its F0 / energy
    curves, so the two recordings stay aligned and only the segments differ.

WHAT IS TRANSFERABLE
    Kokoro (StyleTTS2) keeps timbre and prosody apart inside one 256-d voice
    vector: ref_s[:, :128] drives the decoder (who is speaking), ref_s[:, 128:]
    drives duration, F0 and energy (how it is said). Between two generations
    three curves can travel:

        durations  [T_tok]          frames (600 samples = 25 ms) per phoneme
        f0         [1, 2*T_frames]  pitch contour
        energy     [1, 2*T_frames]  loudness envelope

    extract_prosody() stops the forward pass right after these, before the text
    encoder and the vocoder, so it costs a fraction of a full generation.

PHONEME COUNTS THAT DO NOT MATCH
    Donor and target are the same utterance, but a manipulation can change the
    token count: p -> pʰ and d -> dʲ add a token, tʃ -> ʨ removes one. The
    transfer aligns the two phoneme strings and decides per token:

        unchanged, or substituted 1:1 (t -> ʈ)   the donor's duration and frames
        inserted by the rule (ʰ, ʲ, ̃ )           the model's OWN prediction, kept
                                                 as it is — the new token simply
                                                 adds its own time
        rewritten n:m (tʃ -> ʨ)                  predicted durations, fitted to
                                                 the donor block's total so the
                                                 rest of the utterance stays put
        dropped by the rule                      the donor's frames go with it

    So the timeline only ever moves where a token was added or removed.

MODES (synthesize_forced)
    duration : "copy"    reuse the donor's timing, per the rules above
               "predict" ignore it, let the model time the new string
    f0       : "copy"    the donor's contour, in absolute Hz
               "match"   the donor's contour shifted into the target voice's own
                         register (use when generating in a DIFFERENT voice)
               "predict" the target voice's own pitch
    energy   : "copy" | "predict"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as nnf

from text_manipulation import MAX_TOKENS, REPO, get_g2p

SR            = 24_000          # Kokoro output sample rate
FRAME_SAMPLES = 600             # samples per duration frame -> 40 frames/second
F0_UPSAMPLE   = 2               # F0Ntrain upsamples once: len(f0) == 2 * n_frames

MODEL_DIR = Path("kokoro_model")
VOICE_DIR = Path("custom_voices")

VoiceLike = Union[str, Path, torch.Tensor]


# =============================================================================
# 1. MODEL AND VOICES
# =============================================================================
_MODEL = None

def get_model(device=None):
    """The Kokoro acoustic model, downloaded on first use and cached.
    Importing this module does NOT load it."""
    global _MODEL
    if _MODEL is None:
        from huggingface_hub import hf_hub_download
        from kokoro import KModel
        MODEL_DIR.mkdir(exist_ok=True)
        config, weights = MODEL_DIR / "config.json", MODEL_DIR / "kokoro-v1_0.pth"
        if not config.exists() or not weights.exists():
            print("Downloading Kokoro model files …")
            hf_hub_download(REPO, "config.json",     local_dir=MODEL_DIR)
            hf_hub_download(REPO, "kokoro-v1_0.pth", local_dir=MODEL_DIR)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = KModel(repo_id=REPO, config=str(config), model=str(weights)).to(dev).eval()
    return _MODEL


def load_voice(voice: VoiceLike, custom_dir: Union[str, Path] = VOICE_DIR) -> torch.Tensor:
    """Return a voice pack tensor, shape [N, 1, 256].

    Accepts, in order: a tensor (passed through, so an already-loaded pack
    works), "nicola_50" / "nicola_50.pt" / a path -> loaded from `custom_dir`,
    any other name -> a built-in voice, downloaded through the G2P pipeline.
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
    return get_g2p().load_voice(name)                  # built-in voice, from the hub


def voice_ref(pack: torch.Tensor, phonemes: str) -> torch.Tensor:
    """The [1, 256] row of a pack to use for a phoneme string.

    Mirrors KPipeline, which indexes by len(phonemes) - 1, but clamps so a short
    custom pack never goes out of bounds. An already-sliced row passes through."""
    if pack.dim() == 2:
        return pack
    i = min(max(len(phonemes) - 1, 0), pack.shape[0] - 1)
    ref = pack[i]
    return ref if ref.dim() == 2 else ref.unsqueeze(0)


# =============================================================================
# 2. TOKENISATION
# =============================================================================
@dataclass
class Tokens:
    """The model's view of a phoneme string."""
    ids: torch.LongTensor         # [1, T_tok], with the leading/trailing 0
    phonemes: str                 # only the characters that survived the vocab
    dropped: List[tuple]          # [(position, char), …]

    @property
    def n_tokens(self) -> int:
        return int(self.ids.shape[-1])                 # len(phonemes) + 2


def encode(model, phonemes: str, warn: bool = True) -> Tokens:
    """Phoneme string -> Tokens. Drops characters outside the vocab exactly like
    KPipeline does, but says which ones (text_manipulation.check_ipa prevents it)."""
    kept, ids, dropped = [], [], []
    for pos, ch in enumerate(phonemes):
        tid = model.vocab.get(ch)
        if tid is None:
            dropped.append((pos, ch))
        else:
            kept.append(ch); ids.append(tid)
    if len(ids) > MAX_TOKENS:
        if warn:
            print(f"⚠ {len(ids)} tokens > {MAX_TOKENS}, truncating")
        ids, kept = ids[:MAX_TOKENS], kept[:MAX_TOKENS]
    if dropped and warn:
        preview = ", ".join(f"U+{ord(c):04X} {c!r}@{p}" for p, c in dropped[:6])
        print(f"⚠ {len(dropped)} phoneme(s) not in vocab, dropped: {preview}")
    return Tokens(ids=torch.LongTensor([[0, *ids, 0]]).to(model.device),
                  phonemes="".join(kept), dropped=dropped)


# =============================================================================
# 3. THE FORWARD PASS, IN REUSABLE PIECES
#    (each mirrors the matching lines of KModel.forward_with_tokens)
# =============================================================================
def _lengths_and_mask(input_ids):
    dev = input_ids.device
    lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], device=dev, dtype=torch.long)
    mask = (torch.arange(int(lengths.max()), device=dev)
            .unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths))
    return lengths, torch.gt(mask + 1, lengths.unsqueeze(1))


def _prosody_hidden(model, input_ids, lengths, mask, s):
    """BERT -> prosody text encoder. Returns d [1, T_tok, H]."""
    bert_dur = model.bert(input_ids, attention_mask=(~mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    return model.predictor.text_encoder(d_en, s, lengths, mask)


def _predict_durations(model, d, speed: float = 1.0) -> torch.Tensor:
    """The duration head. Returns [T_tok] long: frames per token."""
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(dim=-1) / speed
    return torch.round(duration).clamp(min=1).long().reshape(-1)


def _alignment(durations: torch.Tensor, device) -> torch.Tensor:
    """Hard monotonic alignment [1, T_tok, T_frames]: column f holds a 1 in row t
    when frame f belongs to token t, so `X @ aln` upsamples token-rate tensors to
    frame rate. This matrix is what carries the rhythm — swap the durations and
    every downstream tensor, the sample count included, follows."""
    durations = durations.to(device).clamp(min=1).long().reshape(-1)
    idx = torch.repeat_interleave(torch.arange(durations.shape[0], device=device), durations)
    aln = torch.zeros((durations.shape[0], idx.shape[0]), device=device)
    aln[idx, torch.arange(idx.shape[0], device=device)] = 1
    return aln.unsqueeze(0)


# =============================================================================
# 4. PROSODY
# =============================================================================
@dataclass
class Prosody:
    """Everything transferable about one (phonemes, voice, speed) triple.
    Tensors live on the CPU, so it is cheap to keep and torch.save-able."""
    phonemes: str                 # the vocab-filtered string these belong to
    durations: torch.Tensor       # [T_tok] long, BOS/EOS included
    f0: torch.Tensor              # [1, 2*T_frames]
    energy: torch.Tensor          # [1, 2*T_frames]
    speed: float = 1.0
    voice: Optional[str] = None

    @property
    def n_tokens(self) -> int:
        return int(self.durations.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.durations.sum())

    @property
    def seconds(self) -> float:
        return self.n_frames * FRAME_SAMPLES / SR

    def median_f0(self, threshold: float = 1.0) -> float:
        voiced = self.f0[self.f0 > threshold]
        return float(voiced.median()) if voiced.numel() else 0.0

    def save(self, path) -> None:
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path) -> "Prosody":
        return cls(**torch.load(path, weights_only=False))

    def __repr__(self) -> str:
        return (f"Prosody(voice={self.voice!r}, {self.n_tokens} tokens, "
                f"{self.n_frames} frames = {self.seconds:.2f}s, "
                f"median F0 {self.median_f0():.1f} Hz, speed {self.speed})")


@torch.no_grad()
def extract_prosody(ipa, voice: VoiceLike, speed: float = 1.0, custom_dir=VOICE_DIR):
    """Durations, F0 and energy for `ipa`, WITHOUT synthesising.

    Skips the text encoder and the decoder — nearly all of the compute — so this
    is several times faster than a generation and allocates no waveform.
    Returns one Prosody, or a list of them when `ipa` is a list of chunks.
    """
    if not isinstance(ipa, str):
        return [extract_prosody(chunk, voice, speed, custom_dir)
                for chunk in ipa if chunk.strip()]

    model = get_model()
    pack = load_voice(voice, custom_dir)
    tok = encode(model, ipa)
    ref_s = voice_ref(pack, tok.phonemes).to(model.device)
    s = ref_s[:, 128:]

    lengths, mask = _lengths_and_mask(tok.ids)
    d = _prosody_hidden(model, tok.ids, lengths, mask, s)
    durations = _predict_durations(model, d, speed)
    en = d.transpose(-1, -2) @ _alignment(durations, model.device)
    f0, energy = model.predictor.F0Ntrain(en, s)

    return Prosody(phonemes=tok.phonemes, durations=durations.cpu(),
                   f0=f0.detach().cpu(), energy=energy.detach().cpu(),
                   speed=speed, voice=None if isinstance(voice, torch.Tensor) else str(voice))


# =============================================================================
# 5. TRANSFER: DONOR TOKENS -> TARGET TOKENS
# =============================================================================
def _offsets(durations: torch.Tensor) -> List[int]:
    """Cumulative frame offsets, so token t owns frames [off[t], off[t+1])."""
    off, acc = [0], 0
    for d in durations.reshape(-1).tolist():
        acc += int(d); off.append(acc)
    return off


def _fit_total(durations: torch.Tensor, total: int, min_dur: int = 1) -> torch.Tensor:
    """Scale per-token durations so they sum to exactly `total`, by largest
    remainder, never pushing a token below `min_dur`. Used only inside a block
    the manipulation rewrote n:m, to keep the rest of the utterance in place."""
    dur = durations.reshape(-1).long().clamp(min=min_dur)
    n = dur.numel()
    total = max(int(total), n * min_dur)
    scaled = dur.float() * (total / float(dur.sum()))
    out = scaled.floor().long().clamp(min=min_dur)
    remainder = scaled - scaled.floor()
    diff = total - int(out.sum())
    if diff > 0:
        order = torch.argsort(remainder, descending=True)
        for k in range(diff):
            out[order[k % n]] += 1
    elif diff < 0:
        order, k = torch.argsort(remainder), 0
        while diff < 0 and k < n * (abs(diff) + 1):
            i = int(order[k % n])
            if out[i] > min_dur:
                out[i] -= 1; diff += 1
            k += 1
    return out


def transfer_plan(donor: Prosody, target_phonemes: str, predicted: torch.Tensor):
    """Align the donor's phonemes with the target's and decide, token by token,
    where its time comes from.

    Returns (durations [T_target], forced [T_target bools], plan), where `plan`
    is the list of ("donor"|"predicted", frame_start, frame_stop, n_target_frames)
    segments that rebuilds the F0 / energy curves in the target's own layout.
    See the module docstring for the rules; `forced[t]` is False exactly where
    the model's own prediction was kept (the tokens a rule inserted).
    """
    d_dur = donor.durations.reshape(-1).long()
    p_dur = predicted.reshape(-1).long().cpu()
    d_off, p_off = _offsets(d_dur), _offsets(p_dur)

    durations, forced, plan = [], [], []

    def take_donor(i1, i2, dur):                        # i1, i2 index d_dur
        durations.extend(int(x) for x in dur)
        plan.append(("donor", d_off[i1], d_off[i2], int(sum(int(x) for x in dur))))

    take_donor(0, 1, d_dur[:1])                        # BOS
    forced.append(True)

    for tag, i1, i2, j1, j2 in SequenceMatcher(a=donor.phonemes, b=target_phonemes,
                                               autojunk=False).get_opcodes():
        di1, di2, pj1, pj2 = i1 + 1, i2 + 1, j1 + 1, j2 + 1     # shift past BOS
        if tag == "delete":                            # the rule removed this token
            continue
        if tag == "equal" or (tag == "replace" and (i2 - i1) == (j2 - j1)):
            take_donor(di1, di2, d_dur[di1:di2])       # 1:1 — donor timing verbatim
            forced += [True] * (j2 - j1)
        elif tag == "insert":                          # ʰ, ʲ, ̃  … keep their own time
            durations.extend(int(x) for x in p_dur[pj1:pj2])
            plan.append(("predicted", p_off[pj1], p_off[pj2], int(p_dur[pj1:pj2].sum())))
            forced += [False] * (j2 - j1)
        else:                                          # n:m rewrite, e.g. tʃ -> ʨ
            fitted = _fit_total(p_dur[pj1:pj2], int(d_dur[di1:di2].sum()))
            take_donor(di1, di2, fitted)
            forced += [True] * (j2 - j1)

    take_donor(len(d_dur) - 1, len(d_dur), d_dur[-1:])  # EOS
    forced.append(True)
    return torch.tensor(durations, dtype=torch.long), forced, plan


def _resample(curve: torch.Tensor, length: int) -> torch.Tensor:
    """Linearly resample a [1, L] curve to [1, length] (a no-op when equal)."""
    if curve.shape[-1] == length:
        return curve
    return nnf.interpolate(curve.unsqueeze(1).float(), size=length,
                           mode="linear", align_corners=True).squeeze(1)


def _splice(donor_curve, predicted_curve, plan) -> torch.Tensor:
    """Rebuild a frame-level curve in the target's layout by following `plan`:
    donor frames wherever the donor's timing was kept, predicted frames for the
    tokens a rule inserted. No stretching happens in the normal case, because
    each segment already has the frame count its tokens ask for."""
    pieces = []
    for src, a, b, n in plan:
        curve = donor_curve if src == "donor" else predicted_curve
        pieces.append(_resample(curve[..., a * F0_UPSAMPLE:b * F0_UPSAMPLE], n * F0_UPSAMPLE))
    return torch.cat(pieces, dim=-1)


def match_register(donor_f0, target_f0, mode: str = "median", threshold: float = 1.0):
    """Move a donor pitch contour into the target voice's register.

    "median"   log-domain median shift: keeps the contour's shape exactly and
               only moves it up or down. The safe default across voices.
    "mean_std" also rescales the log-F0 spread, so a flat donor stops flattening
               an expressive target.
    "none"     passthrough, i.e. the donor's absolute Hz.
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
        d_std, t_std = d_log.std().clamp(min=1e-4), t_log.std().clamp(min=1e-4)
        out[dv] = (((d_log - d_log.mean()) * (t_std / d_std)) + t_log.mean()).exp()
    else:
        raise ValueError(f"unknown f0_match mode {mode!r}")
    return out


# =============================================================================
# 6. GENERATION
# =============================================================================
@dataclass
class Chunk:
    """One synthesised chunk (normally one sentence)."""
    phonemes: str                 # what the model actually read
    audio: np.ndarray             # float32, SR
    prosody: Prosody              # the curves it was built from
    forced: List[bool]            # per token: did it take the donor's timing?

    @property
    def durations(self) -> torch.Tensor:
        return self.prosody.durations

    @property
    def seconds(self) -> float:
        return len(self.audio) / SR


@dataclass
class Speech:
    """A whole utterance: the chunks and their concatenated audio."""
    chunks: List[Chunk]
    voice: Optional[str] = None
    sr: int = SR
    audio: np.ndarray = field(init=False)

    def __post_init__(self):
        self.audio = (np.concatenate([c.audio for c in self.chunks])
                      if self.chunks else np.zeros(0, dtype=np.float32))

    @property
    def phonemes(self) -> List[str]:
        return [c.phonemes for c in self.chunks]

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.sr

    @property
    def n_forced(self) -> tuple:
        """(tokens that took the donor's timing, tokens in total)."""
        forced = [f for c in self.chunks for f in c.forced]
        return sum(forced), len(forced)

    def save(self, path):
        import soundfile as sf
        sf.write(str(path), self.audio, self.sr)
        return path

    def play(self):
        """IPython audio widget, for notebooks."""
        from IPython.display import Audio
        return Audio(self.audio, rate=self.sr)

    def __repr__(self) -> str:
        return (f"Speech(voice={self.voice!r}, {len(self.chunks)} chunks, "
                f"{self.seconds:.2f}s)")


@torch.no_grad()
def _generate_chunk(ipa: str, pack, speed, donor, duration, f0, energy, f0_match,
                    seed=None) -> Chunk:
    model = get_model()
    dev = model.device
    if seed is not None:
        torch.manual_seed(seed)                        # the vocoder's noise excitation
    tok = encode(model, ipa)
    ref_s = voice_ref(pack, tok.phonemes).to(dev)
    s = ref_s[:, 128:]

    lengths, mask = _lengths_and_mask(tok.ids)
    d = _prosody_hidden(model, tok.ids, lengths, mask, s)
    predicted = _predict_durations(model, d, speed)

    # ---- timing ----------------------------------------------------------
    plan = None
    if donor is None or duration == "predict":
        durations, forced = predicted, [False] * int(predicted.shape[0])
    elif duration == "copy":
        durations, forced, plan = transfer_plan(donor, tok.phonemes, predicted)
        durations = durations.to(dev)
    else:
        raise ValueError(f'unknown duration mode {duration!r}: use "copy" or "predict"')

    # ---- frame-level curves ----------------------------------------------
    aln = _alignment(durations, dev)
    en = d.transpose(-1, -2) @ aln
    f0_pred, n_pred = model.predictor.F0Ntrain(en, s)
    f0_out, n_out = f0_pred, n_pred

    if donor is not None and f0 != "predict":
        if f0 not in ("copy", "match"):
            raise ValueError(f'unknown f0 mode {f0!r}: use "copy", "match" or "predict"')
        curve = (_splice(donor.f0.to(dev), f0_pred, plan) if plan is not None
                 else _resample(donor.f0.to(dev), f0_pred.shape[-1]))
        f0_out = match_register(curve, f0_pred, mode=f0_match) if f0 == "match" else curve

    if donor is not None and energy != "predict":
        if energy != "copy":
            raise ValueError(f'unknown energy mode {energy!r}: use "copy" or "predict"')
        n_out = (_splice(donor.energy.to(dev), n_pred, plan) if plan is not None
                 else _resample(donor.energy.to(dev), n_pred.shape[-1]))

    # ---- content + vocoder ------------------------------------------------
    t_en = model.text_encoder(tok.ids, lengths, mask)
    audio = model.decoder(t_en @ aln, f0_out, n_out, ref_s[:, :128]).squeeze()

    used = Prosody(phonemes=tok.phonemes, durations=durations.cpu(),
                   f0=f0_out.detach().cpu(), energy=n_out.detach().cpu(), speed=speed)
    return Chunk(phonemes=tok.phonemes,
                 audio=audio.detach().cpu().numpy().astype(np.float32),
                 prosody=used, forced=forced)


def _as_donors(donor, n_chunks, voice, speed, custom_dir):
    """donor may be None, a Prosody, a list of them, or the IPA it should be
    extracted from (one chunk or a list) — always comes back as a list aligned
    with the target chunks."""
    if donor is None:
        return [None] * n_chunks
    if isinstance(donor, Prosody):
        donors = [donor]
    elif isinstance(donor, str):
        donors = [extract_prosody(donor, voice, speed, custom_dir)]
    elif all(isinstance(x, Prosody) for x in donor):
        donors = list(donor)
    else:
        donors = extract_prosody([x for x in donor if x.strip()], voice, speed, custom_dir)
    if len(donors) != n_chunks:
        raise ValueError(f"{len(donors)} donor chunks for {n_chunks} target chunks — "
                         f"donor and target must be the same utterance, chunked the same way")
    return donors


def synthesize(ipa, voice: VoiceLike, speed: float = 1.0, seed=None,
               custom_dir=VOICE_DIR) -> Speech:
    """Plain, fast generation: one chunk or a list of them, straight to audio.

        speech = synthesize(text_to_ipa_chunks(text), "im_nicola")
        speech.play(); speech.save("baseline.wav")
    """
    return synthesize_forced(ipa, voice, donor=None, speed=speed, seed=seed,
                             custom_dir=custom_dir)


def synthesize_forced(ipa, voice: VoiceLike, donor=None, *, speed: float = 1.0,
                      duration: str = "copy", f0: str = "copy", energy: str = "copy",
                      f0_match: str = "median", seed=None, custom_dir=VOICE_DIR) -> Speech:
    """Generate `ipa`, optionally forcing a donor's prosody onto it.

    ipa    : one IPA chunk or a list of them (manipulate() output).
    donor  : the prosody to force — a Prosody, a list of them (extract_prosody
             output), or simply the ORIGINAL ipa chunks, which are then extracted
             in the same voice. None means an ordinary generation, and every mode
             below is ignored.
    duration / f0 / energy / f0_match : see the module docstring.
    seed   : the vocoder excites unvoiced frames with noise, so two identical
             calls are not bit-identical. Pass a seed and they are.

        ipa   = text_to_ipa_chunks(text)
        manip = manipulate(ipa, rules, seed=0)
        forced = synthesize_forced(manip, "im_nicola", donor=ipa)

    Returns a Speech; `speech.n_forced` says how many tokens took the donor's
    timing and how many kept their own.
    """
    chunks = [ipa] if isinstance(ipa, str) else [c for c in ipa if c.strip()]
    donors = _as_donors(donor, len(chunks), voice, speed, custom_dir)
    pack = load_voice(voice, custom_dir)
    out = [_generate_chunk(ps, pack, speed, dn, duration, f0, energy, f0_match, seed)
           for ps, dn in zip(chunks, donors)]
    return Speech(chunks=out, voice=None if isinstance(voice, torch.Tensor) else str(voice))


def synthesize_pair(original, manipulated, voice: VoiceLike, *, speed: float = 1.0,
                    duration: str = "copy", f0: str = "copy", energy: str = "copy",
                    f0_match: str = "median", seed=0, custom_dir=VOICE_DIR):
    """One stimulus pair: the baseline utterance, and the manipulated one wearing
    its prosody. The donor prosody is extracted once and reused, so the baseline
    is rendered from exactly the curves that are forced onto the manipulated
    version — the two differ only where the phonemes differ. Both tracks share
    one `seed`, so even the vocoder's noise matches.

        base, manip = synthesize_pair(ipa, manip_ipa, "im_nicola")

    Returns (baseline Speech, forced Speech).
    """
    chunks = [original] if isinstance(original, str) else [c for c in original if c.strip()]
    donors = extract_prosody(chunks, voice, speed, custom_dir)
    baseline = synthesize_forced(chunks, voice, donors, speed=speed, duration="copy",
                                 f0="copy", energy="copy", seed=seed, custom_dir=custom_dir)
    forced = synthesize_forced(manipulated, voice, donors, speed=speed, duration=duration,
                               f0=f0, energy=energy, f0_match=f0_match, seed=seed,
                               custom_dir=custom_dir)
    return baseline, forced


__all__ = [
    "SR", "FRAME_SAMPLES", "F0_UPSAMPLE", "MODEL_DIR", "VOICE_DIR",
    "get_model", "load_voice", "voice_ref", "encode", "Tokens",
    "Prosody", "extract_prosody", "transfer_plan", "match_register",
    "Chunk", "Speech", "synthesize", "synthesize_forced", "synthesize_pair",
]
