"""Model presence checks + smart downloads.

Idempotent: nothing is re-downloaded if it's already on disk. Called by
setup.sh and defensively at app startup, so a fresh VM self-heals.

Video  = official HunyuanVideo-1.5 code (generate.py) + official HF weights.
Sound  = official HunyuanVideo-Foley code (infer.py) + HF weights.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import config


def _log(msg: str) -> None:
    print(f"[models] {msg}", flush=True)


def _has_weights(path: Path) -> bool:
    if not path.exists():
        return False
    for pat in ("*.safetensors", "*.pth", "*.bin", "*.gguf", "*.ckpt"):
        if any(path.rglob(pat)):
            return True
    return False


def _clone(url: str, dest: Path, marker: str) -> None:
    """Clone `url` into `dest` unless `dest/marker` already exists."""
    if (dest / marker).exists():
        _log(f"code present: {dest.name} ✓")
        return
    _log(f"cloning {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def _download(repo_id: str, dest: Path) -> None:
    if _has_weights(dest):
        _log(f"weights present: {dest.name} ✓")
        return
    from huggingface_hub import snapshot_download

    _log(f"downloading weights {repo_id} -> {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        token=__import__("os").environ.get("HF_TOKEN"),
        max_workers=8,
    )


def ensure_video() -> tuple[Path, Path]:
    """Ensure HunyuanVideo-1.5 code + weights. Returns (code_dir, weights_dir)."""
    _clone(config.VIDEO_REPO_URL, config.VIDEO_CODE_DIR, "generate.py")
    _download(config.VIDEO_MODEL_REPO, config.VIDEO_WEIGHTS_DIR)
    return config.VIDEO_CODE_DIR, config.VIDEO_WEIGHTS_DIR


def ensure_foley() -> tuple[Path, Path]:
    """Ensure HunyuanVideo-Foley code + weights. Returns (code_dir, weights_dir)."""
    _clone(config.FOLEY_REPO_URL, config.FOLEY_DIR, "infer.py")
    _download(config.FOLEY_MODEL_REPO, config.FOLEY_WEIGHTS_DIR)
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
