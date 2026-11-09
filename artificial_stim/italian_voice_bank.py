# ============================================================
# italian_voice_bank.py
# Builds a bank of 9 distinct Italian voices on top of
# KokoroVoiceBlender (blend_voices.py), using SLERP per half:
#   timbre  (dims :128) -> mostly borrowed from other-language voices
#   prosody (dims 128:) -> always from if_sara / im_nicola (native Italian rhythm/intonation)
# Pronunciation itself comes from lang_code="i" (espeak-ng Italian phonemes).
# ============================================================

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from blend_voices import KokoroVoiceBlender  # also sets the espeak-ng env vars

SAMPLE_RATE = 24000

# ------------------------------------------------------------
# The 9-voice bank
#   timbre / prosody : {voice_name: weight}, SLERP-blended by your per_half_blend
#   speed / pitch / formant (optional): render-time settings, saved to bank_settings.json
#     speed   -> Kokoro speed parameter
#     pitch   -> ratio applied to the median F0 (Praat "Change gender")
#     formant -> formant shift ratio (<1 sounds larger/darker, >1 smaller/brighter)
# Keep a little of the Italian voice in the timbre mix (0.2-0.4) if a donor
# adds a foreign colouring.
# ------------------------------------------------------------
BANK_SPECS = {
    # --- female (Italian prosody from if_sara) ---
    "it01_f_sara": {
        "timbre": {"if_sara": 1.0},
        "prosody": {"if_sara": 1.0},
    },
    "it02_f_es": {
        "timbre": {"ef_dora": 0.8, "if_sara": 0.2},
        "prosody": {"if_sara": 1.0},
    },
    "it03_f_pt": {
        "timbre": {"pf_dora": 0.8, "if_sara": 0.2},
        "prosody": {"if_sara": 1.0},
        "speed": 1.05,
    },
    "it04_f_fr": {
        "timbre": {"ff_siwis": 0.7, "if_sara": 0.3},
        "prosody": {"if_sara": 1.0},
        "speed": 0.95,
    },
    "it05_f_uk": {
        "timbre": {"bf_emma": 0.6, "if_sara": 0.4},
        "prosody": {"if_sara": 1.0},
        "pitch": 0.95,
    },
    # --- male (Italian prosody from im_nicola) ---
    "it06_m_nicola": {
        "timbre": {"im_nicola": 1.0},
        "prosody": {"im_nicola": 1.0},
    },
    "it07_m_es": {
        "timbre": {"em_alex": 0.8, "im_nicola": 0.2},
        "prosody": {"im_nicola": 1.0},
    },
    "it08_m_pt": {
        "timbre": {"pm_santa": 0.8, "im_nicola": 0.2},
        "prosody": {"im_nicola": 1.0},
        "speed": 0.95,
    },
    "it09_m_uk": {
        "timbre": {"bm_george": 0.6, "im_nicola": 0.4},
        "prosody": {"im_nicola": 1.0},
        "pitch": 0.92,
        "formant": 0.96,
    },
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


class ItalianVoiceBank(KokoroVoiceBlender):
    """KokoroVoiceBlender plus per-half bank building, auditioning, audio-level
    variation and a speaker-similarity check."""

    # ------------------------------------------------------------
    # Per-half blends (timbre / prosody), built with your SLERP
    # ------------------------------------------------------------
    def create_half_blends(self, specs: dict[str, dict]) -> dict[str, torch.Tensor]:
        """specs: {name: {"timbre": {...}, "prosody": {...}, ...}}"""
        blends = {}
        for name, spec in specs.items():
            tensor = self.per_half_blend(spec["timbre"], spec["prosody"])
            assert tensor.shape == self.EXPECTED_SHAPE, f"{name} has wrong shape: {tensor.shape}"
            blends[name] = tensor
            print(f"{name}: timbre={spec['timbre']} prosody={spec['prosody']}")
        return blends

    # ------------------------------------------------------------
    # Synthesis (overrides the parent: concatenates all chunks
    # instead of overwriting the wav with each chunk)
    # ------------------------------------------------------------
    def render(self, name: str, tensor: torch.Tensor, text: str, speed: float = 1.0) -> np.ndarray:
        self.pipeline.voices[name] = tensor
        chunks = []
        for _, _, audio in self.pipeline(text, voice=name, speed=speed):
            if audio is None:
                continue
            chunks.append(audio.cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio))
        return np.concatenate(chunks)

    def synthesize(
        self,
        name: str,
        tensor: torch.Tensor,
        text: str,
        speed: float = 1.0,
        pitch: float = 1.0,
        formant: float = 1.0,
    ) -> Path:
        audio = self.render(name, tensor, text, speed)
        if pitch != 1.0 or formant != 1.0:
            audio = self.shift_voice(audio, pitch, formant)
        out_wav = self.out_dir / f"{name}_sample.wav"
        sf.write(out_wav, audio, SAMPLE_RATE)
        print(f"Audio saved: {out_wav}")
        return out_wav

    # ------------------------------------------------------------
    # Audio-level variation (pip install praat-parselmouth)
    # ------------------------------------------------------------
    @staticmethod
    def shift_voice(audio: np.ndarray, pitch: float = 1.0, formant: float = 1.0,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
        """Shift median pitch and formants with Praat's 'Change gender'.
        Keep ratios within about 0.85-1.15 to avoid artifacts."""
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
        median = call(snd.to_pitch(), "Get quantile", 0, 0, 0.5, "Hertz")
        new_median = median * pitch if not np.isnan(median) else 0.0  # 0 = leave pitch unchanged
        # args: pitch floor, pitch ceiling, formant shift ratio, new pitch median,
        #       pitch range factor, duration factor
        shifted = call(snd, "Change gender", 75, 600, formant, new_median, 1.0, 1.0)
        return shifted.values[0].astype(np.float32)

    # ------------------------------------------------------------
    # Build the whole bank
    # ------------------------------------------------------------
    def build_bank(self, specs: dict[str, dict], text: str) -> dict[str, torch.Tensor]:
        blends = self.create_half_blends(specs)
        settings = {}
        for name, tensor in blends.items():
            self.save(name, tensor)
            render = {k: specs[name].get(k, 1.0) for k in ("speed", "pitch", "formant")}
            settings[name] = render
            self.synthesize(name, tensor, text, **render)
        settings_path = self.out_dir / "bank_settings.json"
        settings_path.write_text(json.dumps(settings, indent=2))
        print(f"Render settings saved: {settings_path}")
        return blends

    # ------------------------------------------------------------
    # Audition many timbre donors on one Italian prosody
    # ------------------------------------------------------------
    def audition_donors(self, prosody_voice: str, donors: list[str], text: str,
                        amount: float = 0.8) -> list[Path]:
        """Writes try_<prosody>_<donor>_<amount>_sample.wav for each donor.
        amount = share of the donor in the timbre half (rest is the Italian voice)."""
        paths = []
        for donor in donors:
            name = f"try_{prosody_voice}_{donor}_{int(amount * 100)}"
            tensor = self.per_half_blend(
                {donor: amount, prosody_voice: 1.0 - amount},
                {prosody_voice: 1.0},
            )
            paths.append(self.synthesize(name, tensor, text))
        return paths

    # ------------------------------------------------------------
    # How different do the voices actually sound? (pip install resemblyzer)
    # ------------------------------------------------------------
    @staticmethod
    def similarity_report(wav_paths: list[Path]) -> np.ndarray:
        """Cosine similarity between speaker embeddings of the given wavs.
        Rough guide: above ~0.85 the two usually sound like the same speaker."""
        from resemblyzer import VoiceEncoder, preprocess_wav

        wav_paths = [Path(p) for p in wav_paths]
        encoder = VoiceEncoder()
        embs = np.stack([encoder.embed_utterance(preprocess_wav(p)) for p in wav_paths])
        sim = embs @ embs.T  # resemblyzer embeddings are L2-normalised

        names = [p.stem.replace("_sample", "") for p in wav_paths]
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
    text = (
        "Buongiorno a tutti. Oggi vi racconto come ho passato il fine settimana: "
        "una passeggiata in centro, un caffè con gli amici e poi una cena tranquilla a casa."
    )

    print("\n--- Building the 9-voice bank ---")
    bank.build_bank(BANK_SPECS, text)

    # Optional: try other donors and swap the best ones into BANK_SPECS
    # bank.audition_donors("if_sara", FEMALE_DONORS, text, amount=0.8)
    # bank.audition_donors("im_nicola", MALE_DONORS, text, amount=0.8)

    # Optional: check that the 9 voices are actually distinct
    # bank.similarity_report(sorted(bank.out_dir.glob("it*_sample.wav")))

    print("\nDone. Check the 'italian_bank' folder.")
