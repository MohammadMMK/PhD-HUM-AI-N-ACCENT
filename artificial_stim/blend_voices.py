# ============================================================
# blend_voices.py
# KokoroVoiceBlender: load, blend (N-way SLERP), save, and
# synthesize Kokoro TTS voice embeddings.
# ============================================================

# ------------------------------------------------------------
# Windows / espeak-ng setup (must run before kokoro import)
# ------------------------------------------------------------
import os

os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from kokoro import KPipeline
import soundfile as sf


class KokoroVoiceBlender:
    """Load, blend, save, and synthesize Kokoro voice embeddings.

    Blending uses row-wise SLERP (spherical linear interpolation).
    Blending more than two voices is done by folding voices in one at a
    time, each time SLERPing the running blend towards the next voice by
    however much weight it should end up with relative to what's already
    been folded in. This keeps every intermediate tensor unit-norm-per-row
    like the source embeddings, which a plain weighted average would not.
    """

    REPO_ID = "hexgrad/Kokoro-82M"
    VOICE_DIR = "voices"
    EXPECTED_SHAPE = (510, 1, 256)
    TIMBRE_DIMS = 128  # a[..., :128] -> decoder (timbre), a[..., 128:] -> predictor (prosody)

    def __init__(self, out_dir: str | Path = "custom_voices", lang_code: str = "i", eps: float = 1e-7):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.lang_code = lang_code
        self.eps = eps
        self._voice_cache: dict[str, torch.Tensor] = {}
        self._pipeline: KPipeline | None = None

    # ------------------------------------------------------------
    # Loading source voices
    # ------------------------------------------------------------
    def load_voice(self, name: str) -> torch.Tensor:
        """Download (if needed) and load a Kokoro voice embedding by name. Cached per instance."""
        if name not in self._voice_cache:
            path = hf_hub_download(repo_id=self.REPO_ID, filename=f"{self.VOICE_DIR}/{name}.pt")
            tensor = torch.load(path, weights_only=True)
            assert tensor.shape == self.EXPECTED_SHAPE, (
                f"Unexpected shape for {name}: {tensor.shape}, expected {self.EXPECTED_SHAPE}"
            )
            self._voice_cache[name] = tensor
        return self._voice_cache[name]

    # ------------------------------------------------------------
    # SLERP
    # ------------------------------------------------------------
    @staticmethod
    def slerp(v1: torch.Tensor, v2: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
        """
        Spherical linear interpolation between two batches of vectors.
        v1, v2: shape (..., D) - interpolated independently along the last dim.
        t: interpolation factor, 0.0 -> v1, 1.0 -> v2
        """
        assert v1.shape == v2.shape, "Shape mismatch"
        assert 0.0 <= t <= 1.0

        norm1 = v1.norm(dim=-1, keepdim=True)
        norm2 = v2.norm(dim=-1, keepdim=True)

        u1 = v1 / norm1.clamp_min(eps)
        u2 = v2 / norm2.clamp_min(eps)

        dot = (u1 * u2).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)

        near_parallel = sin_omega.abs() < eps

        slerp_dir = torch.where(
            near_parallel,
            (1 - t) * u1 + t * u2,
            (torch.sin((1 - t) * omega) / sin_omega.clamp_min(eps)) * u1
            + (torch.sin(t * omega) / sin_omega.clamp_min(eps)) * u2,
        )
        slerp_dir = slerp_dir / slerp_dir.norm(dim=-1, keepdim=True).clamp_min(eps)

        interp_norm = (1 - t) * norm1 + t * norm2
        return slerp_dir * interp_norm

    # ------------------------------------------------------------
    # Blending - any number of voices
    # ------------------------------------------------------------
    def blend(self, weights: dict[str, float]) -> torch.Tensor:
        """
        Blend two or more named voices via sequential weighted SLERP.

        weights: {"if_sara": 0.5, "im_nicola": 0.3, "hf_alpha": 0.2}
        Weights don't need to sum to 1 - they're normalized automatically.
        Order of the dict doesn't matter; voices are folded in one by one.
        """
        assert len(weights) >= 2, "Need at least two voices to blend"
        names = list(weights.keys())
        total = sum(weights.values())
        norm_weights = {name: w / total for name, w in weights.items()}

        acc = self.load_voice(names[0])
        acc_weight = norm_weights[names[0]]
        for name in names[1:]:
            w = norm_weights[name]
            t = w / (acc_weight + w)
            acc = self.slerp(acc, self.load_voice(name), t)
            acc_weight += w
        return acc

    def per_half_blend(self, tim_weights: dict[str, float], pro_weights: dict[str, float]) -> torch.Tensor:
        """
        Blend timbre (first 128 dims -> decoder) and prosody (last 128 dims ->
        predictor) with independent voice mixes, each supporting any number of
        voices. e.g. keep timbre close to one voice while blending prosody
        across several others.

        tim_weights, pro_weights: {"voice_name": weight, ...} (need not share
        the same voices or keys).
        """
        tim = self._blend_component(tim_weights, slice(0, self.TIMBRE_DIMS))
        pro = self._blend_component(pro_weights, slice(self.TIMBRE_DIMS, None))
        return torch.cat([tim, pro], dim=-1)

    def _blend_component(self, weights: dict[str, float], dim_slice: slice) -> torch.Tensor:
        assert len(weights) >= 1, "Need at least one voice"
        names = list(weights.keys())
        total = sum(weights.values())
        norm_weights = {name: w / total for name, w in weights.items()}

        acc = self.load_voice(names[0])[..., dim_slice]
        if len(names) == 1:
            return acc
        acc_weight = norm_weights[names[0]]
        for name in names[1:]:
            w = norm_weights[name]
            t = w / (acc_weight + w)
            acc = self.slerp(acc, self.load_voice(name)[..., dim_slice], t)
            acc_weight += w
        return acc

    def create_blends(self, specs: dict[str, dict[str, float]]) -> dict[str, torch.Tensor]:
        """Build several blends at once from a name -> weights spec dict, e.g.
        {"sara50_nicola30_hf20": {"if_sara": 0.5, "im_nicola": 0.3, "hf_alpha": 0.2}}"""
        blends = {name: self.blend(weights) for name, weights in specs.items()}
        for name, tensor in blends.items():
            assert tensor.shape == self.EXPECTED_SHAPE, f"{name} has wrong shape: {tensor.shape}"
            print(f"{name}: shape OK -> {tensor.shape}, dtype {tensor.dtype}")
        return blends

    # ------------------------------------------------------------
    # Save / load blends
    # ------------------------------------------------------------
    def save(self, name: str, tensor: torch.Tensor) -> Path:
        out_path = self.out_dir / f"{name}.pt"
        torch.save(tensor, out_path)
        print(f"Saved: {out_path}")
        return out_path

    def save_all(self, blends: dict[str, torch.Tensor]) -> None:
        for name, tensor in blends.items():
            self.save(name, tensor)

    def load_saved(self, name: str) -> torch.Tensor:
        path = self.out_dir / f"{name}.pt"
        tensor = torch.load(path, weights_only=True)
        assert tensor.shape == self.EXPECTED_SHAPE, f"{path.name} has wrong shape: {tensor.shape}"
        return tensor

    def load_all_saved(self) -> dict[str, torch.Tensor]:
        blends = {}
        for path in self.out_dir.glob("*.pt"):
            tensor = torch.load(path, weights_only=True)
            assert tensor.shape == self.EXPECTED_SHAPE, f"{path.name} has wrong shape: {tensor.shape}"
            blends[path.stem] = tensor
        return blends

    # ------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------
    @property
    def pipeline(self) -> KPipeline:
        if self._pipeline is None:
            self._pipeline = KPipeline(lang_code=self.lang_code)
        return self._pipeline

    def synthesize(self, name: str, tensor: torch.Tensor, text: str) -> Path:
        """Register a blended voice with the pipeline and synthesize a single wav file."""
        self.pipeline.voices[name] = tensor.squeeze(0)
        out_wav = self.out_dir / f"{name}_sample.wav"
        for _, _, audio in self.pipeline(text, voice=name):
            sf.write(out_wav, audio, 24000)
            print(f"Audio saved: {out_wav}")
        return out_wav

    def synthesize_all(self, blends: dict[str, torch.Tensor], text: str) -> None:
        for name, tensor in blends.items():
            self.synthesize(name, tensor, text)


# ------------------------------------------------------------
# Run everything (example: two- and three-voice blends)
# ------------------------------------------------------------
if __name__ == "__main__":
    blender = KokoroVoiceBlender()

    print("\n--- Creating blends ---")
    specs = {
        "hf_alpha_af_heart_t05_p05": {"hf_alpha": 0.5, "af_heart": 0.5},
    }
    blends = blender.create_blends(specs)

    print("\n--- Saving blends ---")
    blender.save_all(blends)

    print("\n--- Synthesizing test audio ---")
    test_text = "Ciao, questa è una voce mescolata creata da due voci italiane."
    blender.synthesize_all(blends, test_text)

    print("\nDone. Check the 'custom_voices' folder for .pt files and .wav samples.")

