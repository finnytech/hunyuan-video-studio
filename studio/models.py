"""Model presence checks + smart downloads.

Idempotent: nothing is re-downloaded if it's already on disk. Called by
setup.sh and defensively at app startup, so a fresh VM self-heals.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import config


def _log(msg: str) -> None:
    print(f"[models] {msg}", flush=True)


def _has_snapshot(path: Path) -> bool:
    """A HF snapshot is 'present' if the dir exists and has real weight files."""
    if not path.exists():
        return False
    for pat in ("*.safetensors", "*.pth", "*.bin", "*.gguf"):
        if any(path.rglob(pat)):
            return True
    return False


def ensure_video_model(repo_id: str) -> Path:
    """Download a diffusers HunyuanVideo 1.5 snapshot if missing. Returns local dir."""
    from huggingface_hub import snapshot_download

    local = config.MODELS_DIR / repo_id.split("/")[-1]
    if _has_snapshot(local):
        _log(f"video model present: {local.name} ✓")
        return local
    _log(f"downloading video model {repo_id} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local),
        token=os.environ.get("HF_TOKEN"),
        max_workers=8,
    )
    _log(f"video model ready: {local}")
    return local


def ensure_foley() -> tuple[Path, Path]:
    """Clone the Foley repo + download its weights if missing.

    Returns (code_dir, weights_dir).
    """
    from huggingface_hub import snapshot_download

    # 1) code checkout
    if not (config.FOLEY_DIR / "infer.py").exists():
        _log("cloning HunyuanVideo-Foley code ...")
        config.FOLEY_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", config.FOLEY_REPO_URL, str(config.FOLEY_DIR)],
            check=True,
        )
    else:
        _log("foley code present ✓")

    # 2) weights
    if _has_snapshot(config.FOLEY_WEIGHTS_DIR):
        _log("foley weights present ✓")
    else:
        _log(f"downloading foley weights {config.FOLEY_MODEL_REPO} ...")
        snapshot_download(
            repo_id=config.FOLEY_MODEL_REPO,
            local_dir=str(config.FOLEY_WEIGHTS_DIR),
            token=os.environ.get("HF_TOKEN"),
            max_workers=8,
        )
    return config.FOLEY_DIR, config.FOLEY_WEIGHTS_DIR


def ensure_all() -> None:
    """Ensure everything the app needs is on disk."""
    ensure_video_model(config.VIDEO_MODEL_T2V_480)
    ensure_video_model(config.VIDEO_MODEL_T2V_720)
    ensure_foley()
    _log("all models ready ✓")


if __name__ == "__main__":
    try:
        ensure_all()
    except Exception as e:  # noqa: BLE001
        _log(f"ERROR: {e}")
        sys.exit(1)
