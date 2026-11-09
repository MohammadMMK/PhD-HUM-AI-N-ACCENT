# ============================================================
# italian_voice_bank.py  (v3: 3 native voices -> 9 by post-processing)
#
# Why: the decoder half of a Kokoro voice (dims :128) decides how each phoneme is
# articulated (/r/, vowel quality, ...). Mixing a foreign voice into it brings in
# foreign phonetics, whatever the prosody. So every voice here uses ONLY the native
# Italian voices (and blends of them), and the 9 voices differ in things that do not
# change phoneme identity:
#
#   semitones   : pitch LEVEL shift. Done INSIDE Kokoro: the transferred F0 curve is
#                 shifted before decoding, so the decoder renders the new pitch itself
#                 (no PSOLA artifacts; the Italian contour shape is kept).
#   pitch_range : pitch RANGE factor (log-F0 excursions around the voice's mean),
#                 also inside Kokoro. Keep 0.8-1.2 so the Italian melody stays natural.
#   formant     : vocal-tract length (Praat "Change gender", pitch untouched).
#                 <1 = larger / darker speaker, >1 = smaller / brighter. Keep 0.90-1.10.
#
# Each voice also gets the Italian duration, contour and energy of its reference
# voice (same transfer as accent_tools.render_condition), so all voices that share a
# reference have identical word timings.
#
# Needs next to this file: blend_voices.py, functions.py, accent_tools.py
# ============================================================

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import functions as F                             # model, _voice_pack
from blend_voices import KokoroVoiceBlender      # SLERP blending
from accent_tools import (load_story, make_native, render_condition, export_condition, merge_results,
                          make_alignment, decode, f0_stats, voiced_mask, SR)

# ------------------------------------------------------------
# Prosody transfer from the reference voice (see accent_tools.render_condition)
# ------------------------------------------------------------
TRANSFER = dict(
    duration=False, pitch=False, energy=False,
    dur_mode="raw", dur_scope="sentence",
    f0_mode="meanstd", energy_mode="meanstd", norm_scope="text",
)

# ------------------------------------------------------------
# 3 base voices: native Italian only (SLERP mix applied to BOTH halves)
#   voice : {voice_name: weight}
#   ref   : native voice whose duration / contour / energy are transferred (None = own prosody)
# ------------------------------------------------------------
BASES = {
    "sara":   {"voice": {"if_sara": 0.95,"im_nicola": 0.05 }},
    "nicola": {"voice": {"im_nicola": 0.8, "if_sara": 0.2}},
    "mix":    {"voice": {"im_nicola": 0.55, "if_sara": 0.45}},
}

# ------------------------------------------------------------
# 3 variants per base -> 9 voices.
# Pitch and formants move TOGETHER (lower pitch + longer vocal tract = a larger
# speaker), which sounds like a plausible different person rather than a processed one.
# A base can override these with its own "variants" dict.
# ------------------------------------------------------------
VARIANTS = {
    "a": dict(semitones=0.0,  pitch_range=1.00, formant=1.00),   # the base voice
    "b": dict(semitones=-2.0, pitch_range=0.85, formant=0.93),   # larger, darker, calmer
    "c": dict(semitones=+2.0, pitch_range=1.15, formant=1.07),   # smaller, brighter, livelier
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _fit_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad to n samples (keeps the player's token timings valid)."""
    return audio[:n] if len(audio) >= n else np.pad(audio, (0, n - len(audio)))


def shift_f0(f0: np.ndarray, center: float, semitones: float = 0.0, pitch_range: float = 1.0,
             voiced: np.ndarray | None = None) -> np.ndarray:
    """Shift / scale the voiced frames of an F0 curve (Hz) in the log domain.
    center : mean log-F0 of the voice over the whole text (the range is scaled around it)
    voiced : which frames to change (default voiced_mask(f0)); unvoiced frames are kept."""
    out = np.array(f0, dtype=np.float64, copy=True)
    v = voiced_mask(out) if voiced is None else voiced
    if v.any():
        lf = np.log(out[v])
        out[v] = np.exp(center + pitch_range * (lf - center) + semitones * np.log(2) / 12)
    return out


class ItalianVoiceBank(KokoroVoiceBlender):
    """3 native Italian voices x 3 post-processing variants, with Italian prosody transfer."""

    def __init__(self, out_dir: str | Path = "italian_bank", transfer: dict | None = None, **kwargs):
        super().__init__(out_dir=out_dir, **kwargs)
        self.transfer = {**TRANSFER, **(transfer or {})}

    # ------------------------------------------------------------
    # Voices
    # ------------------------------------------------------------
    def make_base(self, weights: dict[str, float]) -> torch.Tensor:
        """Same SLERP mix for timbre and register (works for a single voice too)."""
        tensor = self.per_half_blend(weights, weights)
        assert tensor.shape == self.EXPECTED_SHAPE, f"wrong shape: {tensor.shape}"
        return tensor

    @staticmethod
    def prepare_text(source: str, name: str = "bank"):
        """source: a story file path (blocks separated by ### lines) or the text itself."""
        return make_native(load_story(source, name=name, verbose=False))

    def render_base(self, cond, name: str, tensor: torch.Tensor, ref: str | None) -> list[dict]:
        if ref is None:
            return render_condition(cond, voices=(name, tensor), ref_voice=None, verbose=False)
        return render_condition(cond, voices=(name, tensor), ref_voice=ref, verbose=False, **self.transfer)

    # ------------------------------------------------------------
    # Variant 1: pitch level / range, re-decoded inside Kokoro
    # ------------------------------------------------------------
    @torch.no_grad()
    def repitch(self, results: list[dict], tensor: torch.Tensor, name: str,
                semitones: float = 0.0, pitch_range: float = 1.0) -> list[dict]:
        """Re-decode every sentence with the same durations and energy and a shifted F0 curve.
        Returns NEW result dicts (the base results are left untouched)."""
        if semitones == 0.0 and pitch_range == 1.0:
            return [dict(r, voice=name, pros=dict(r["pros"])) for r in results]
        pack, _ = F._voice_pack(tensor)
        # one center for the whole text, so every sentence is scaled around the same pitch level
        center = f0_stats([info["f0"] for r in results for _, info in r["pros"]["chunks"]])[0]
        out = []
        for r in results:
            audio, chunks = [], []
            for _, info in r["pros"]["chunks"]:
                f0 = shift_f0(info["f0"], center, semitones, pitch_range)
                a = decode(info["ps"], pack, make_alignment(info["ps"], info["dur"]), f0, info["n"])
                audio.append(a)
                chunks.append((a, {**info, "f0": f0}))
            pros = dict(r["pros"], chunks=chunks)
            curve = pros["f0"]                         # player curve: NaN where unvoiced
            pros["f0"] = shift_f0(curve, center, semitones, pitch_range, voiced=np.isfinite(curve))
            new_audio = np.concatenate(audio)
            assert len(new_audio) == len(r["audio"]), "re-decoded length differs (durations changed?)"
            out.append(dict(r, voice=name, audio=new_audio, pros=pros))
        return out

    # ------------------------------------------------------------
    # Variant 2: formants (vocal-tract length), Praat
    # ------------------------------------------------------------
    @staticmethod
    def shift_formants(audio: np.ndarray, formant: float, sr: int = SR) -> np.ndarray:
        """Praat 'Change gender' with the pitch left unchanged (pip install praat-parselmouth)."""
        if formant == 1.0:
            return audio
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=sr)
        # args: pitch floor, pitch ceiling, formant shift ratio, new pitch median (0 = keep),
        #       pitch range factor, duration factor
        shifted = call(snd, "Change gender", 75, 600, formant, 0, 1.0, 1.0)
        return _fit_length(shifted.values[0].astype(np.float32), len(audio))

    def make_variant(self, base_results: list[dict], tensor: torch.Tensor, name: str,
                     semitones: float = 0.0, pitch_range: float = 1.0, formant: float = 1.0) -> list[dict]:
        res = self.repitch(base_results, tensor, name, semitones, pitch_range)
        for r in res:
            r["audio"] = self.shift_formants(r["audio"], formant)
        return res

    # ------------------------------------------------------------
    # Build the bank
    # ------------------------------------------------------------
    def build_bank(self, source: str, bases: dict = BASES, variants: dict = VARIANTS,
                   story_name: str = "bank") -> dict[str, dict]:
        """For every voice <base>_<variant> writes, in <out_dir>/<voice>/:
             <voice>_block<N>.html/.wav and <voice>_full.html/.wav  (player with pitch / energy panel)
           every full wav in <out_dir>/_listen/, base tensors <out_dir>/<base>.pt,
           and all settings in <out_dir>/bank_settings.json."""
        cond = self.prepare_text(source, story_name)
        listen = self.out_dir / "_listen"
        listen.mkdir(parents=True, exist_ok=True)
        settings = {"transfer": self.transfer, "voices": {}}

        for base, bspec in bases.items():
            tensor = self.make_base(bspec["voice"])
            self.save(base, tensor)
            print(f"\n=== base {base}: {bspec['voice']}, prosody from {bspec['ref']} ===")
            base_results = self.render_base(cond, base, tensor, bspec["ref"])   # decoded once per base

            for var, vspec in bspec.get("variants", variants).items():
                name = f"{base}_{var}"
                res = self.make_variant(base_results, tensor, name, **vspec)
                export_condition(cond, res, self.out_dir / name, prefix=name)
                sf.write(listen / f"{name}.wav", merge_results(res)["audio"], SR)
                settings["voices"][name] = {"base": base, "voice": bspec["voice"], "ref": bspec["ref"], **vspec}
                print(f"   {name}: {vspec}")

        path = self.out_dir / "bank_settings.json"
        path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSettings saved: {path}\nListening folder: {listen}")
        return settings

    # ------------------------------------------------------------
    # Synthesize new text later with a saved voice + its settings
    # ------------------------------------------------------------
    def render_saved(self, source: str, name: str, story_name: str = "text") -> list[dict]:
        """Re-create voice `name` from bank_settings.json for any text."""
        settings = json.loads((self.out_dir / "bank_settings.json").read_text(encoding="utf-8"))
        s = settings["voices"][name]
        self.transfer = settings["transfer"]
        tensor = self.load_saved(s["base"])
        cond = self.prepare_text(source, story_name)
        base_results = self.render_base(cond, s["base"], tensor, s["ref"])
        return self.make_variant(base_results, tensor, name, s["semitones"], s["pitch_range"], s["formant"])

    # ------------------------------------------------------------
    # How different do the voices sound? (pip install resemblyzer)
    # ------------------------------------------------------------
    @staticmethod
    def similarity_report(wav_paths: list[Path]) -> np.ndarray:
        """Cosine similarity between speaker embeddings. Above ~0.85 usually = same speaker."""
        from resemblyzer import VoiceEncoder, preprocess_wav

        wav_paths = [Path(p) for p in wav_paths]
        encoder = VoiceEncoder()
        embs = np.stack([encoder.embed_utterance(preprocess_wav(p)) for p in wav_paths])
        sim = embs @ embs.T                              # resemblyzer embeddings are L2-normalised

        names = [p.stem for p in wav_paths]
        width = max(len(n) for n in names)
        print(" " * (width + 1) + " ".join(f"{i:>5}" for i in range(len(names))))
        for i, n in enumerate(names):
            print(f"{n:<{width}} " + " ".join(f"{s:5.2f}" for s in sim[i]) + f"  [{i}]")
        pairs = [(sim[i, j], names[i], names[j])
                 for i in range(len(names)) for j in range(i + 1, len(names))]
        print("\nMost similar pairs:")
        for s, a, b in sorted(pairs, reverse=True)[:5]:
            print(f"  {s:.2f}  {a} <-> {b}")
        return sim


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    bank = ItalianVoiceBank(out_dir="italian_bank")

    # A story file ("story1.txt", blocks separated by ### lines) or the text itself
    source = (
        "Irene Valli lavorava come botanica all'università di Bologna, e da molti anni studiava le piante che crescono vicino al mare."
        "Quell'anno doveva raccogliere alcuni campioni di una piccola pianta con fiori gialli, che cresceva soltanto sulle rocce di un'isola del sud."
    )

    bank.build_bank(source)

    # Check that the 9 voices are distinct; widen semitones / formant for any pair above ~0.85
    # bank.similarity_report(sorted((bank.out_dir / "_listen").glob("*.wav")))

    # Later, any text with one of the voices:
    # res = bank.render_saved("story1.txt", "nicola_b")

    print("\nDone. Open italian_bank/<voice>/<voice>_full.html to see the pitch / energy curves.")
