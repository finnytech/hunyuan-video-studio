"""HunyuanVideo-Foley wrapper — native, timeline-synced sound.

Runs AFTER the video stage's process has fully exited, so it starts on a clean,
near-empty GPU (no VRAM contention). Foley watches the generated video + reads the
prompt and emits 48kHz audio aligned to the on-screen action, then writes a video
that already carries that synced audio track.

We call the official infer.py through the shared runner (timeout + logs + retry).
Offload is auto-enabled if free VRAM is tight (XXL: 20GB -> 12GB).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import config, runner


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
        return 999.0  # if we can't tell, don't force offload


def add_foley(video_in: Path, prompt: str, out_dir: Optional[Path] = None) -> Path:
    """Generate + mux Foley audio for `video_in`. Returns the video WITH sound."""
    from .models import ensure_foley

    code_dir, weights_dir = ensure_foley()
    out_dir = Path(out_dir) if out_dir else (config.OUTPUT_DIR / "foley")
    out_dir.mkdir(parents=True, exist_ok=True)

    offload = config.FOLEY_OFFLOAD or _free_vram_gb() < 24.0
    cmd = [
        sys.executable, "infer.py",
        "--model_path", str(weights_dir),
        "--single_video", str(video_in),
        "--single_prompt", prompt,
        "--output_dir", str(out_dir),
    ]
    if offload:
        cmd.append("--enable_offload")
        _log("offload ON (tight VRAM)")

    runner.run(
        cmd, cwd=code_dir, stage="foley",
        timeout=config.FOLEY_TIMEOUT, retries=config.STAGE_RETRIES,
    )

    result = _newest_video(out_dir)
    if result is None:
        raise RuntimeError("Foley produced no output video (see outputs/logs)")
    _log(f"foley output: {result}")
    return result


def _newest_video(out_dir: Path) -> Optional[Path]:
    vids = [p for p in out_dir.rglob("*.mp4") if p.stat().st_size > 0]
    if not vids:
        return None
    return max(vids, key=lambda p: p.stat().st_mtime)


def has_audio_stream(path: Path) -> bool:
    """True if the file carries a non-empty audio track."""
    ff = shutil.which("ffprobe")
    if not ff:
        return False
    out = subprocess.run(
        [ff, "-i", str(path), "-show_streams", "-select_streams", "a",
         "-loglevel", "error", "-of", "csv=p=0"],
        capture_output=True, text=True,
    )
    return bool(out.stdout.strip())


def newest_audio(out_dir: Path) -> Optional[Path]:
    """Fallback: locate a standalone audio file if Foley returned audio-only."""
    auds = [p for p in out_dir.rglob("*.wav") if p.stat().st_size > 0]
    if not auds:
        return None
    return max(auds, key=lambda p: p.stat().st_mtime)
