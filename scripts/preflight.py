"""Preflight dependency + model check.

Exit codes:
  0  -> everything present (core + optional); launch directly.
  1  -> something CORE is missing; caller must run setup.sh.
  2  -> core OK but an OPTIONAL piece is missing (e.g. foley); caller may run
        setup once, but can also launch (video works, sound may be off).

"Core" = python deps to import the app + HunyuanVideo code + weights + ffmpeg.
Foley (sound) is treated as OPTIONAL: the video pipeline runs without it, so a
missing foley checkout/weights only prints a warning.

Run:  python scripts/preflight.py            # core check
      python scripts/preflight.py --verbose  # list every component
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

# Ensure the repo root (parent of scripts/) is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio import config


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:  # noqa: BLE001
        return False


def _has_weights(path) -> bool:
    if not path.exists():
        return False
    for pat in ("*.safetensors", "*.pth", "*.bin", "*.ckpt"):
        if any(path.rglob(pat)):
            return True
    return False


def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    core_missing: list[str] = []
    optional_missing: list[str] = []

    for mod in ("gradio", "torch", "huggingface_hub", "numpy"):
        if not _have(mod):
            core_missing.append(f"pip:{mod}")

    if _have("torch"):
        try:
            import torch

            if not torch.cuda.is_available():
                core_missing.append("torch-cuda (no GPU visible)")
        except Exception as e:  # noqa: BLE001
            core_missing.append(f"torch-import ({e})")

    if not (config.VIDEO_CODE_DIR / "generate.py").exists():
        core_missing.append("hunyuan-code")

    if not _has_weights(config.VIDEO_WEIGHTS_DIR):
        core_missing.append("hunyuan-weights")

    if not shutil.which("ffmpeg"):
        core_missing.append("ffmpeg-binary")

    # optional: foley (sound stage)
    if not (config.FOLEY_DIR / "infer.py").exists():
        optional_missing.append("foley-code")
    if not _has_weights(config.FOLEY_WEIGHTS_DIR):
        optional_missing.append("foley-weights")

    if verbose or core_missing or optional_missing:
        print("== preflight ==")
        print(f"  core missing     : {core_missing or 'none ✓'}")
        print(f"  optional missing : {optional_missing or 'none ✓'}")

    if core_missing:
        print("[preflight] core deps missing -> setup required", flush=True)
        return 1
    if optional_missing:
        print("[preflight] core OK ✓ (foley missing; video works, sound may be off)", flush=True)
        return 2
    print("[preflight] all deps + models present ✓ -> launching directly", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
