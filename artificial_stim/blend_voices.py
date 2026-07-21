# ============================================================
# kokoro_voice_blend.py
# Create, save, and use three Italian voice blends from
# if_sara + im_nicola using SLERP interpolation
# ============================================================

# ------------------------------------------------------------
# SECTION 0 — Windows / espeak-ng setup (must run before kokoro import)
# ------------------------------------------------------------
import os

os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"


# ------------------------------------------------------------
# SECTION 1 — Imports
# ------------------------------------------------------------
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from kokoro import KPipeline
import soundfile as sf

REPO_ID = "hexgrad/Kokoro-82M"
VOICE_DIR = "voices"
OUT_DIR = Path("custom_voices")
OUT_DIR.mkdir(exist_ok=True)
EXPECTED_SHAPE = (510, 1, 256)

# ------------------------------------------------------------
# SECTION 2 — Load a source voice tensor
# ------------------------------------------------------------
def load_voice(name: str) -> torch.Tensor:
    """Download (if needed) and load a Kokoro voice embedding by name."""
    path = hf_hub_download(repo_id=REPO_ID, filename=f"{VOICE_DIR}/{name}.pt")
    tensor = torch.load(path, weights_only=True)
    assert tensor.shape == EXPECTED_SHAPE, (
        f"Unexpected shape for {name}: {tensor.shape}, expected {EXPECTED_SHAPE}"
    )
    return tensor

# ------------------------------------------------------------
# SECTION 3 — SLERP blending
# ------------------------------------------------------------
def slerp(v1: torch.Tensor, v2: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
    """
    Spherical linear interpolation between two batches of vectors.
    v1, v2: shape (..., D) — interpolated independently along the last dim.
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


def blend_voices(v1: torch.Tensor, v2: torch.Tensor, w1: float) -> torch.Tensor:
    """
    Blend two Kokoro voice tensors of shape (510, 1, 256) via row-wise SLERP.
    w1 = weight toward v1 (e.g. w1=0.8 -> 80% v1, 20% v2)
    """
    assert v1.shape == v2.shape, "Voice tensors must match in shape"
    t = 1.0 - w1  # slerp(t=0) = v1
    return slerp(v1, v2, t)


def per_half_blend(a: torch.Tensor, b: torch.Tensor, t_tim: float, t_pro: float, fn=slerp) -> torch.Tensor:
    """Blend timbre (:128 -> decoder) and prosody (128: -> predictor)
    with independent weights. e.g. t_tim=0.2 keeps voice near A,
    t_pro=0.8 moves rhythm/intonation toward B."""
    tim = fn(a[..., :128], b[..., :128], t_tim)
    pro = fn(a[..., 128:], b[..., 128:], t_pro)
    return torch.cat([tim, pro], dim=-1)

# ------------------------------------------------------------
# SECTION 4 — Build the three blends
# ------------------------------------------------------------
def create_blends() -> dict[str, torch.Tensor]:
    voice_sara = load_voice("if_sara")
    voice_nicola = load_voice("im_nicola")

    print("if_sara shape:", voice_sara.shape)
    print("im_nicola shape:", voice_nicola.shape)


        # "if_sara80_im_nicola20": blend_voices(voice_sara, voice_nicola, w1=0.8),
        # "if_sara20_im_nicola80": blend_voices(voice_sara, voice_nicola, w1=0.2),
        # "if_sara50_im_nicola50": blend_voices(voice_sara, voice_nicola, w1=0.5),
        # "per_half_slerp": per_half_blend(voice_sara, voice_nicola, t_tim=0.5, t_pro=0.5),
        # "tim_50_pro_mas1": per_half_blend(voice_sara, voice_nicola, t_tim=0.5, t_pro=1),
        # "tim_mas80_pro_mas1": per_half_blend(voice_sara, voice_nicola, t_tim=0.2, t_pro=1),
        # "tim_fem80_pro_mas1": per_half_blend(voice_sara, voice_nicola, t_tim=0.8, t_pro=1),
    blends = {
        # "if_sara70_im_nicola30": blend_voices(voice_sara, voice_nicola, w1=0.7),
        "nicola95": blend_voices(voice_sara, voice_nicola, w1=0.05),
        # "if_sara45_im_nicola55": blend_voices(voice_sara, voice_nicola, w1=0.45)
    }

    for name, tensor in blends.items():
        assert tensor.shape == EXPECTED_SHAPE, f"{name} has wrong shape: {tensor.shape}"
        print(f"{name}: shape OK -> {tensor.shape}, dtype {tensor.dtype}")

    return blends

# ------------------------------------------------------------
# SECTION 5 — Save blends to disk
# ------------------------------------------------------------
def save_blends(blends: dict[str, torch.Tensor]) -> None:
    for name, tensor in blends.items():
        out_path = OUT_DIR / f"{name}.pt"
        torch.save(tensor, out_path)
        print(f"Saved: {out_path}")

# ------------------------------------------------------------
# SECTION 6 — Load blends back from disk (for reuse later)
# ------------------------------------------------------------
def load_saved_blends() -> dict[str, torch.Tensor]:
    blends = {}
    for path in OUT_DIR.glob("*.pt"):
        tensor = torch.load(path, weights_only=True)
        assert tensor.shape == EXPECTED_SHAPE, f"{path.name} has wrong shape: {tensor.shape}"
        blends[path.stem] = tensor
    return blends

# ------------------------------------------------------------
# SECTION 7 — Synthesize test audio with each blended voice
# ------------------------------------------------------------
def synthesize_all(blends: dict[str, torch.Tensor], text: str) -> None:
    pipeline = KPipeline(lang_code="i")  # Italian

    for name, tensor in blends.items():
        pipeline.voices[name] = tensor.squeeze(0)  # register with pipeline
        generator = pipeline(text, voice=name)

        for i, (gs, ps, audio) in enumerate(generator):
            out_wav = OUT_DIR / f"{name}_sample.wav"
            sf.write(out_wav, audio, 24000)
            print(f"Audio saved: {out_wav}")

# ------------------------------------------------------------
# SECTION 8 — Run everything
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Creating blends ---")
    blends = create_blends()

    print("\n--- Saving blends ---")
    save_blends(blends)

    print("\n--- Synthesizing test audio ---")
    test_text = "Ciao, questa è una voce mescolata creata da due voci italiane."
    synthesize_all(blends, test_text)

    print("\nDone. Check the 'custom_voices' folder for .pt files and .wav samples.")