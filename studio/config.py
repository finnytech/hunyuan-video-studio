"""Central configuration for HunyuanVideo Studio.

Everything is env-overridable so you never have to touch code on the VM.
Defaults are tuned for a single A100 80GB (12 vCPU, ~312 TFLOPS) on Lightning.ai.
"""
from __future__ import annotations

import os
from pathlib import Path


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("STUDIO_MODELS_DIR", ROOT / "models"))
OUTPUT_DIR = Path(os.environ.get("STUDIO_OUTPUT_DIR", ROOT / "outputs"))
CACHE_DIR = Path(os.environ.get("STUDIO_CACHE_DIR", ROOT / ".cache"))

for _d in (MODELS_DIR, OUTPUT_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- Model ids -------------------------------------------------------------
# Diffusers-compatible HunyuanVideo 1.5 checkpoints (community layout).
VIDEO_MODEL_T2V_480 = os.environ.get(
    "VIDEO_MODEL_480", "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
)
VIDEO_MODEL_T2V_720 = os.environ.get(
    "VIDEO_MODEL_720", "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"
)
# 1080p is produced by the 720p base + the built-in super-resolution upscaler.

FOLEY_REPO_URL = os.environ.get(
    "FOLEY_REPO_URL", "https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley"
)
FOLEY_MODEL_REPO = os.environ.get("FOLEY_MODEL_REPO", "tencent/HunyuanVideo-Foley")
FOLEY_DIR = MODELS_DIR / "HunyuanVideo-Foley"          # code checkout
FOLEY_WEIGHTS_DIR = MODELS_DIR / "HunyuanVideo-Foley-weights"


# --- Generation defaults ---------------------------------------------------
FPS = _int("STUDIO_FPS", 24)
MAX_SECONDS = _int("STUDIO_MAX_SECONDS", 16)
DEFAULT_SECONDS = _int("STUDIO_DEFAULT_SECONDS", 5)
DEFAULT_STEPS = _int("STUDIO_STEPS", 30)

# Resolutions offered in the UI. 1080p uses the super-res path.
RESOLUTIONS = {
    "480p": (848, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}
DEFAULT_RESOLUTION = os.environ.get("STUDIO_DEFAULT_RES", "720p")


# --- Optimization toggles (A100-friendly defaults) -------------------------
# FP8/GGUF quantization: ~halves transformer VRAM, near-identical quality on 1.5.
FP8_TRANSFORMER = _flag("FP8_TRANSFORMER", True)
# SageAttention (SSTA sparse/INT8 attention): 30-40% faster + softer VRAM spikes.
USE_SAGE_ATTENTION = _flag("USE_SAGE_ATTENTION", True)
# TeaCache: skip near-unchanged diffusion steps -> up to ~2x less render time.
USE_TEACACHE = _flag("USE_TEACACHE", True)
TEACACHE_THRESH = float(os.environ.get("TEACACHE_THRESH", "0.15"))
# Stream weights to CPU when idle; required to fit 16s comfortably in 80GB.
CPU_OFFLOAD = _flag("CPU_OFFLOAD", True)
# Decode VAE in tiles to avoid decode-time VRAM blowups at 1080p.
VAE_TILING = _flag("VAE_TILING", True)
# Attention backend hint for HunyuanVideo 1.5 (A100 -> flash_hub).
ATTENTION_BACKEND = os.environ.get("ATTENTION_BACKEND", "flash_hub")
# Foley offload (XXL: 20GB -> 12GB). Off by default since A100 has headroom;
# turned on automatically if free VRAM is tight.
FOLEY_OFFLOAD = _flag("FOLEY_OFFLOAD", False)


# --- Server / auth ---------------------------------------------------------
SERVER_NAME = os.environ.get("STUDIO_HOST", "0.0.0.0")
SERVER_PORT = _int("STUDIO_PORT", 7860)
# When True, create a public *.gradio.live link. Token gate still applies.
PUBLIC_SHARE = _flag("STUDIO_SHARE", True)
# Pin a token to reuse across restarts; otherwise a fresh one is generated.
PINNED_TOKEN = os.environ.get("STUDIO_TOKEN") or None
# Max concurrent generations (protects the single GPU from pile-ups).
MAX_CONCURRENCY = _int("STUDIO_MAX_CONCURRENCY", 1)
