"""HunyuanVideo-Foley wrapper.

The official Foley release ships an `infer.py` that takes a video + a text prompt
and writes a video with generated, timeline-synced 48kHz audio. We call it as a
subprocess so we stay compatible with their exact environment/version, and we run
it *after* the video pipeline is torn down so it gets a clean, near-empty GPU.

Offload is auto-enabled when free VRAM is tight (XXL: 20GB -> 12GB).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import config


def _log(msg: str) -> None:
    print(f"[foley] {msg}", flush=True)


def _free_vram_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return 0.0


def add_foley(
    video_in: Path,
    prompt: str,
    out_dir: Optional[Path] = None,
) -> Path:
    """Generate + mux Foley audio for `video_in`. Returns path to the video WITH sound."""
    from .models import ensure_foley

    code_dir, weights_dir = ensure_foley()
    out_dir = out_dir or (config.OUTPUT_DIR / "foley")
    out_dir.mkdir(parents=True, exist_ok=True)

    offload = config.FOLEY_OFFLOAD or _free_vram_gb() < 24.0
    cmd = [
        sys.executable,
        str(code_dir / "infer.py"),
        "--model_path", str(weights_dir),
        "--single_video", str(video_in),
        "--single_prompt", prompt,
        "--output_dir", str(out_dir),
    ]
    if offload:
        cmd.append("--enable_offload")
        _log("offload ON (tight VRAM)")

    _log(f"running foley infer: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(code_dir))

    # infer.py writes one output video (with audio) into out_dir. Grab the newest mp4.
    candidates = sorted(out_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("Foley produced no output video")
    result = candidates[0]
    _log(f"foley output: {result}")
    return result


def has_audio_stream(path: Path) -> bool:
    """True if the file already carries an audio track."""
    ff = shutil.which("ffprobe")
    if not ff:
        return False
    out = subprocess.run(
        [ff, "-i", str(path), "-show_streams", "-select_streams", "a",
         "-loglevel", "error", "-of", "csv=p=0"],
        capture_output=True, text=True,
    )
    return bool(out.stdout.strip())
