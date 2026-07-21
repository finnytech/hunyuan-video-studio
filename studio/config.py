"""Central configuration for HunyuanVideo Studio.

Everything is env-overridable so you never have to touch code on the VM.
Defaults are tuned for a single A100 80GB (12 vCPU, ~312 TFLOPS) on Lightning.ai.

The video backend calls Tencent's OFFICIAL `generate.py` (torchrun) so all the
optimizations (FP8 gemm, SageAttention, TeaCache, 1080p super-res, offloading)
are real, tested flags — not reimplementations.
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


# --- HunyuanVideo 1.5 (video) ---------------------------------------------
# Official inference code (contains generate.py) and official weights.
VIDEO_REPO_URL = os.environ.get(
    "VIDEO_REPO_URL", "https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git"
)
VIDEO_MODEL_REPO = os.environ.get("VIDEO_MODEL_REPO", "tencent/HunyuanVideo-1.5")
VIDEO_CODE_DIR = MODELS_DIR / "HunyuanVideo-1.5"            # code checkout (generate.py)
VIDEO_WEIGHTS_DIR = MODELS_DIR / "HunyuanVideo-1.5-weights"  # HF weights

# --- HunyuanVideo-Foley (sound) -------------------------------------------
FOLEY_REPO_URL = os.environ.get(
    "FOLEY_REPO_URL", "https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley"
)
FOLEY_MODEL_REPO = os.environ.get("FOLEY_MODEL_REPO", "tencent/HunyuanVideo-Foley")
FOLEY_DIR = MODELS_DIR / "HunyuanVideo-Foley"
FOLEY_WEIGHTS_DIR = MODELS_DIR / "HunyuanVideo-Foley-weights"


# --- Generation defaults ---------------------------------------------------
FPS = _int("STUDIO_FPS", 24)             # HunyuanVideo 1.5 exports at 24 fps
MAX_SECONDS = _int("STUDIO_MAX_SECONDS", 16)
DEFAULT_SECONDS = _int("STUDIO_DEFAULT_SECONDS", 5)
DEFAULT_STEPS = _int("STUDIO_STEPS", 50)  # official optimal = 50 steps
DEFAULT_ASPECT = os.environ.get("STUDIO_ASPECT", "16:9")
NPROC = _int("STUDIO_NPROC", 1)          # GPUs for torchrun (A100 single)

# UI resolution -> (generate.py --resolution, needs_super_resolution_to_1080)
# 1080p is produced by rendering 720p then the built-in VSR upscaler (--sr true).
RESOLUTION_MAP = {
    "480p": ("480p", False),
    "720p": ("720p", False),
    "1080p": ("720p", True),
}
DEFAULT_RESOLUTION = os.environ.get("STUDIO_DEFAULT_RES", "720p")


# --- Optimization toggles (A100-friendly, real generate.py flags) ----------
# SageAttention: faster inference, softer VRAM spikes (auto-disables flex-block).
USE_SAGE_ATTENTION = _flag("USE_SAGE_ATTENTION", True)
SAGE_BLOCKS_RANGE = os.environ.get("SAGE_BLOCKS_RANGE", "0-53")
# Feature cache (TeaCache/DeepCache/TaylorCache): up to ~2x less render time.
ENABLE_CACHE = _flag("ENABLE_CACHE", True)
CACHE_TYPE = os.environ.get("CACHE_TYPE", "teacache")  # deepcache | teacache | taylorcache
CACHE_START_STEP = _int("CACHE_START_STEP", 11)   # begin skipping steps here
CACHE_END_STEP = _int("CACHE_END_STEP", 45)       # stop skipping here
CACHE_STEP_INTERVAL = _int("CACHE_STEP_INTERVAL", 4)
# CPU offloading: fit long/1080p renders comfortably in 80GB.
OFFLOADING = _flag("OFFLOADING", True)
OVERLAP_GROUP_OFFLOADING = _flag("OVERLAP_GROUP_OFFLOADING", True)
# CFG-distilled transformer: ~2x speedup (must still use 50 steps).
CFG_DISTILLED = _flag("CFG_DISTILLED", False)
# Sparse attention (SSTA) requires H-series GPUs -> OFF on A100.
SPARSE_ATTN = _flag("SPARSE_ATTN", False)
# Prompt rewrite needs a separate vLLM server -> OFF by default (self-contained).
REWRITE = _flag("REWRITE", False)
DTYPE = os.environ.get("STUDIO_DTYPE", "bf16")  # bf16 | fp32; fp8 gemm via sgl-kernel

# Foley offload (XXL: 20GB -> 12GB). Auto-enabled if free VRAM is tight.
FOLEY_OFFLOAD = _flag("FOLEY_OFFLOAD", False)

# --- Speed / hardware tuning (A100 80GB, 12 vCPU) --------------------------
# TF32 matmul + cuDNN autotune: free throughput on Ampere, no quality hit.
ENABLE_TF32 = _flag("ENABLE_TF32", True)
# CPU threads for data/VAE work. 12 vCPU -> leave a couple for the OS/UI.
CPU_THREADS = _int("STUDIO_CPU_THREADS", 10)
# Hard timeouts (seconds) so a wedged subprocess can't pin the GPU forever.
VIDEO_TIMEOUT = _int("STUDIO_VIDEO_TIMEOUT", 3600)
FOLEY_TIMEOUT = _int("STUDIO_FOLEY_TIMEOUT", 1200)
# Retry a stage once on transient failure (OOM hiccup, kernel autotune, etc.).
STAGE_RETRIES = _int("STUDIO_STAGE_RETRIES", 1)
# Where finished, downloadable videos are collected on the VM.
VIDEOS_DIR = Path(os.environ.get("STUDIO_VIDEOS_DIR", ROOT / "outputs" / "videos"))
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
# Subprocess log directory for post-mortem debugging.
LOG_DIR = Path(os.environ.get("STUDIO_LOG_DIR", ROOT / "outputs" / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


# --- Server / auth ---------------------------------------------------------
SERVER_NAME = os.environ.get("STUDIO_HOST", "0.0.0.0")
SERVER_PORT = _int("STUDIO_PORT", 7860)
PUBLIC_SHARE = _flag("STUDIO_SHARE", True)
PINNED_TOKEN = os.environ.get("STUDIO_TOKEN") or None
MAX_CONCURRENCY = _int("STUDIO_MAX_CONCURRENCY", 1)
