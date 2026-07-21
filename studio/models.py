"""Model presence checks + smart, MINIMAL downloads.

The full tencent/HunyuanVideo-1.5 repo is ~372GB because it ships 11 transformer
variants (480p/720p, t2v/i2v, distilled, sparse, SR ...) at ~33GB each. For this
studio's flow we only need a couple of them, so we pull ONLY the variants listed
in config.VIDEO_VARIANTS plus the shared vae/upsampler/scheduler/config. That
drops the download from ~372GB to ~72GB.

We also enable hf_transfer (Rust multi-threaded downloader) for much faster,
higher-throughput fetches, and warn loudly if HF_TOKEN is missing.

Idempotent: nothing is re-downloaded if it's already on disk.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import config


def _log(msg: str) -> None:
    print(f"[models] {msg}", flush=True)


def _enable_fast_transfer() -> None:
    if not config.HF_FAST:
        return
    try:
        import hf_transfer  # noqa: F401

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        _log("hf_transfer enabled (fast multi-threaded downloads) ✓")
    except Exception:  # noqa: BLE001
        _log("hf_transfer not installed; run `pip install hf_transfer` for max speed")


def _warn_token() -> None:
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        _log("⚠️  HF_TOKEN not set — downloads are rate-limited & slower. "
             "Set:  export HF_TOKEN=hf_xxx")


def _has_weights(path: Path) -> bool:
    if not path.exists():
        return False
    for pat in ("*.safetensors", "*.pth", "*.bin", "*.gguf", "*.ckpt"):
        if any(path.rglob(pat)):
            return True
    return False


def _clone(url: str, dest: Path, marker: str) -> None:
    if (dest / marker).exists():
        _log(f"code present: {dest.name} ✓")
        return
    _log(f"cloning {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def _variant_present(weights_dir: Path, variant: str) -> bool:
    d = weights_dir / "transformer" / variant
    return d.exists() and any(d.rglob("*.safetensors"))


def ensure_video() -> tuple[Path, Path]:
    """Ensure HunyuanVideo-1.5 code + the MINIMAL set of weights we use."""
    _enable_fast_transfer()
    _warn_token()
    _clone(config.VIDEO_REPO_URL, config.VIDEO_CODE_DIR, "generate.py")

    wdir = config.VIDEO_WEIGHTS_DIR
    variants = config.VIDEO_VARIANTS
    missing = [v for v in variants if not _variant_present(wdir, v)]

    if not missing and _has_weights(wdir):
        _log(f"video weights present for variants {variants} ✓")
        return config.VIDEO_CODE_DIR, wdir

    # Whitelist: shared components + only the transformer variants we need.
    allow = [
        "config.json",
        "scheduler/*",
        "vae/*",
        "upsampler/*",          # both SR upsamplers are tiny (<300MB total)
    ]
    for v in variants:
        allow.append(f"transformer/{v}/*")

    from huggingface_hub import snapshot_download

    _log(f"downloading MINIMAL video weights (variants={variants}) -> {wdir}")
    _log(f"  allow_patterns={allow}")
    snapshot_download(
        repo_id=config.VIDEO_MODEL_REPO,
        local_dir=str(wdir),
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        allow_patterns=allow,
        max_workers=8,
    )
    _log("video weights ready ✓")
    return config.VIDEO_CODE_DIR, wdir


def ensure_foley() -> tuple[Path, Path]:
    """Ensure HunyuanVideo-Foley code + weights. Returns (code_dir, weights_dir)."""
    _enable_fast_transfer()
    _warn_token()
    _clone(config.FOLEY_REPO_URL, config.FOLEY_DIR, "infer.py")

    if _has_weights(config.FOLEY_WEIGHTS_DIR):
        _log("foley weights present ✓")
        return config.FOLEY_DIR, config.FOLEY_WEIGHTS_DIR

    from huggingface_hub import snapshot_download

    _log(f"downloading foley weights {config.FOLEY_MODEL_REPO} -> {config.FOLEY_WEIGHTS_DIR}")
    snapshot_download(
        repo_id=config.FOLEY_MODEL_REPO,
        local_dir=str(config.FOLEY_WEIGHTS_DIR),
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        max_workers=8,
    )
    return config.FOLEY_DIR, config.FOLEY_WEIGHTS_DIR


def ensure_all() -> None:
    ensure_video()
    ensure_foley()
    _log("all models ready ✓")


if __name__ == "__main__":
    try:
        ensure_all()
    except Exception as e:  # noqa: BLE001
        _log(f"ERROR: {e}")
        sys.exit(1)
