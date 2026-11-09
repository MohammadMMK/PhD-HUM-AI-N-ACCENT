# ============================================================
# italian_voice_bank.py  (v2: explicit prosody transfer)
#
# Builds a bank of Italian voices and renders every voice with the DURATION,
# PITCH CONTOUR and ENERGY of a native Italian voice (if_sara / im_nicola),
# using synth_accent() / render_condition() from accent_tools.py.
#
# How a voice is made:
#   timbre   (dims :128)  -> decoder: what the voice sounds like (SLERP mix, blend_voices.py)
#   register (dims 128:)  -> predictor: the voice's OWN pitch level / range and energy level.
#                            The Italian contour is mapped INTO this register
#                            (f0_mode="meanstd": donor log-F0 z-scored, then given the
#                            register's mean and std), so voices differ in pitch level
#                            while sharing the Italian melody.
#   ref                   -> native voice whose timing, contour and energy are transferred
#
# Note: if register == ref (e.g. register if_sara with ref if_sara), the transfer changes
# nothing, because Kokoro's predictor already produced exactly that voice's prosody.
#
# Needs next to this file: blend_voices.py, functions.py, accent_tools.py
# ============================================================

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from blend_voices import KokoroVoiceBlender      # SLERP blending; also sets the espeak-ng env vars
from accent_tools import (load_story, make_native, render_condition, export_condition,
                          merge_results, SR)       # loads the Kokoro model once (functions.py)

# ------------------------------------------------------------
# Prosody transfer settings (passed to render_condition)
#   dur_mode   "raw"  -> exactly the reference voice's frames: every voice has IDENTICAL timing
#              "rate" -> the reference rhythm, stretched to the voice's own speaking rate
#   f0_mode    "meanstd" -> reference contour in the voice's own pitch level AND range
#              "mean"    -> voice's own pitch level, reference range
#              "raw"     -> reference Hz unchanged (every voice at the reference's pitch)
#   energy_mode: same options as f0_mode, for energy
#   norm_scope "text" -> register stats over the whole text | "sentence" -> per sentence
# ------------------------------------------------------------
TRANSFER = dict(
    duration=True, pitch=True, energy=True,
    dur_mode="raw", dur_scope="sentence",
    f0_mode="meanstd", energy_mode="meanstd", norm_scope="text",
)

# ------------------------------------------------------------
# The 9-voice bank
#   timbre   : {voice: weight}  SLERP mix for dims :128
#   register : {voice: weight}  SLERP mix for dims 128: (pitch level / range the contour is mapped into)
#   ref      : native Italian voice whose duration, contour and energy are transferred
#   pitch / formant (optional): Praat post-processing ratios, applied after synthesis
# Keep 20-40% of the Italian voice in the timbre if a donor still sounds foreign
# after the transfer: that colouring comes from the decoder, not from prosody.
# ------------------------------------------------------------
BANK_SPECS = {
    # --- female, Italian prosody from if_sara ---
    "it01_f_sara": {"timbre": {"if_sara": 1.0},                 "register": {"if_sara": 1.0},  "ref": "if_sara"},
    "it02_f_ef":   {"timbre": {"ef_dora": 0.8, "if_sara": 0.2}, "register": {"if_sara": 1.0},  "ref": "if_sara"},
    "it03_f_pf":   {"timbre": {"pf_dora": 0.8, "if_sara": 0.2}, "register": {"if_sara": 1.0},  "ref": "if_sara"},
    "it04_f_ff":   {"timbre": {"ff_siwis": 0.8, "if_sara": 0.3}, "register": {"if_sara": 1.0}, "ref": "if_sara"},
    "it05_f_bf":   {"timbre": {"bf_emma": 0.8, "if_sara": 0.3}, "register": {"if_sara": 1.0},  "ref": "if_sara"},
    # --- male, Italian prosody from im_nicola ---
    "it06_m_nicola": {"timbre": {"im_nicola": 0.8, "if_sara": 0.2},                 "register": {"im_nicola": 1.0}, "ref": "im_nicola"},
    "it07_m_es":     {"timbre": {"em_alex": 0.8, "im_nicola": 0.2}, "register": {"im_nicola": 1.0},   "ref": "im_nicola"},
    "it08_m_pt":     {"timbre": {"pm_santa": 0.8, "im_nicola": 0.2}, "register": {"im_nicola": 1.0}, "ref": "im_nicola"},
    "it09_m_uk":     {"timbre": {"bm_george": 0.6, "im_nicola": 0.4}, "register": {"im_nicola": 1.0},"ref": "im_nicola"},
    # mix of male female 
    "it10_mf_":     {"timbre": {"if_sara": 0.30, "im_nicola": 0.70}, "register": {"im_nicola": 1.0},"ref": "im_nicola"}
}

# Donor candidates for auditioning other timbres (Kokoro v1.0 voice names).
FEMALE_DONORS = [
    "ef_dora", "pf_dora", "ff_siwis", "af_heart", "af_bella", "af_nicole",
    "af_sarah", "bf_emma", "bf_isabella", "hf_alpha", "hf_beta",
]
MALE_DONORS = [
    "em_alex", "em_santa", "pm_alex", "pm_santa", "am_adam", "am_michael",
    "am_eric", "bm_george", "bm_lewis", "bm_daniel", "hm_omega", "hm_psi",
]


def _fit_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad to n samples (keeps the player's token timings valid after post-processing)."""
    return audio[:n] if len(audio) >= n else np.pad(audio, (0, n - len(audio)))


class ItalianVoiceBank(KokoroVoiceBlender):
    """KokoroVoiceBlender + prosody transfer from a native Italian voice (accent_tools.synth_accent)."""

    def __init__(self, out_dir: str | Path = "italian_bank", transfer: dict | None = None, **kwargs):
        super().__init__(out_dir=out_dir, **kwargs)
        self.transfer = {**TRANSFER, **(transfer or {})}

    # ------------------------------------------------------------
    # Voices
    # ------------------------------------------------------------
    def make_voice(self, spec: dict) -> torch.Tensor:
        register = spec.get("register", spec.get("prosody"))     # "prosody" accepted for old specs
        tensor = self.per_half_blend(spec["timbre"], register)
        assert tensor.shape == self.EXPECTED_SHAPE, f"wrong shape: {tensor.shape}"
        return tensor

    def create_voices(self, specs: dict[str, dict]) -> dict[str, torch.Tensor]:
        voices = {}
        for name, spec in specs.items():
            voices[name] = self.make_voice(spec)
            print(f"{name}: timbre={spec['timbre']} register={spec.get('register', spec.get('prosody'))} "
                  f"ref={spec['ref']}")
        return voices

    # ------------------------------------------------------------
    # Text -> native condition (one or more blocks, separated by ### lines)
    # ------------------------------------------------------------
    @staticmethod
    def prepare_text(source: str, name: str = "bank"):
        """source: a story file path or the text itself."""
        return make_native(load_story(source, name=name, verbose=False))

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------
    def render_voice(self, cond, name: str, tensor: torch.Tensor, ref: str | None) -> list[dict]:
        """ref = native voice -> its duration / contour / energy are transferred;
        ref = None -> the voice's own prosody (for comparison)."""
        if ref is None:
            return render_condition(cond, voices=(name, tensor), ref_voice=None, verbose=False)
        return render_condition(cond, voices=(name, tensor), ref_voice=ref, verbose=False, **self.transfer)

    def post_process(self, results: list[dict], pitch: float = 1.0, formant: float = 1.0) -> list[dict]:
        if pitch == 1.0 and formant == 1.0:
            return results
        for r in results:
            n = len(r["audio"])
            r["audio"] = _fit_length(self.shift_voice(r["audio"], pitch, formant), n)
        return results

    @staticmethod
    def shift_voice(audio: np.ndarray, pitch: float = 1.0, formant: float = 1.0, sr: int = SR) -> np.ndarray:
        """Praat 'Change gender' (pip install praat-parselmouth). Keep ratios within ~0.85-1.15.
        The player's pitch panel shows the curves BEFORE this step."""
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=sr)
        median = call(snd.to_pitch(), "Get quantile", 0, 0, 0.5, "Hertz")
        new_median = median * pitch if not np.isnan(median) else 0.0     # 0 = leave pitch unchanged
        # args: pitch floor, pitch ceiling, formant shift ratio, new pitch median, pitch range factor, duration factor
        shifted = call(snd, "Change gender", 75, 600, formant, new_median, 1.0, 1.0)
        return shifted.values[0].astype(np.float32)

    # ------------------------------------------------------------
    # Build the whole bank
    # ------------------------------------------------------------
    def build_bank(self, specs: dict[str, dict], source: str, compare_own: bool = True,
                   story_name: str = "bank") -> dict[str, torch.Tensor]:
        """For every voice writes, in <out_dir>/<voice>/:
             <voice>_transfer_block<N>.html/.wav, <voice>_transfer_full.html/.wav   (Italian prosody)
             <voice>_own_...                                                     (if compare_own)
           the voice tensor <out_dir>/<voice>.pt, every full wav in <out_dir>/_listen/,
           and the settings in <out_dir>/bank_settings.json."""
        cond = self.prepare_text(source, story_name)
        listen = self.out_dir / "_listen"
        listen.mkdir(parents=True, exist_ok=True)
        voices = self.create_voices(specs)
        settings = {"transfer": self.transfer, "voices": {}}

        for name, tensor in voices.items():
            spec = specs[name]
            post = {k: spec.get(k, 1.0) for k in ("pitch", "formant")}
            vdir = self.out_dir / name
            self.save(name, tensor)
            print(f"\n=== {name}: Italian duration + pitch + energy from {spec['ref']} ===")

            res = self.post_process(self.render_voice(cond, name, tensor, spec["ref"]), **post)
            export_condition(cond, res, vdir, prefix=f"{name}_transfer")
            sf.write(listen / f"{name}_transfer.wav", merge_results(res)["audio"], SR)

            if compare_own:
                own = self.post_process(self.render_voice(cond, name, tensor, None), **post)
                export_condition(cond, own, vdir, prefix=f"{name}_own")
                sf.write(listen / f"{name}_own.wav", merge_results(own)["audio"], SR)

            settings["voices"][name] = {**spec, **post}

        path = self.out_dir / "bank_settings.json"
        path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSettings saved: {path}\nListening folder: {listen}")
        return voices

    # ------------------------------------------------------------
    # Audition many timbre donors with the Italian prosody transferred
    # ------------------------------------------------------------
    def audition_donors(self, ref: str, donors: list[str], source: str, amount: float = 0.8,
                        register: str = "donor") -> list[Path]:
        """Writes <out_dir>/_audition/<ref>_<donor>_<amount>.wav for each donor.
        amount   : donor share in the timbre half (the rest is `ref`)
        register : "donor" -> the donor's own pitch level | "ref" -> the Italian voice's"""
        cond = self.prepare_text(source, "audition")
        out = self.out_dir / "_audition"
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for donor in donors:
            name = f"{ref}_{donor}_{int(amount * 100)}"
            tensor = self.per_half_blend({donor: amount, ref: 1.0 - amount},
                                         {donor: 1.0} if register == "donor" else {ref: 1.0})
            res = self.render_voice(cond, name, tensor, ref)
            path = out / f"{name}.wav"
            sf.write(path, merge_results(res)["audio"], SR)
            print(f"Audition saved: {path}")
            paths.append(path)
        return paths

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
        "Buongiorno a tutti, Oggi vi racconto come ho passato il fine settimana: "
        "una passeggiata in centro, un caffè con gli amici e poi una cena tranquilla a casa."
    )

    bank.build_bank(BANK_SPECS, source, compare_own=True)

    # Optional: try other timbre donors with the Italian prosody, then edit BANK_SPECS
    # bank.audition_donors("if_sara", FEMALE_DONORS, source, amount=0.8)
    # bank.audition_donors("im_nicola", MALE_DONORS, source, amount=0.8)

    # Optional: how distinct are the 9 voices after transfer?
    # bank.similarity_report(sorted((bank.out_dir / "_listen").glob("*_transfer.wav")))

    print("\nDone. Open italian_bank/<voice>/<voice>_transfer_full.html to see the pitch / energy curves.")
