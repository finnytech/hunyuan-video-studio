"""HunyuanVideo 1.5 text-to-video via Tencent's official generate.py.

We drive the official inference script with torchrun so every optimization is a
real, tested flag rather than a reimplementation:

  * --use_sageattn      SageAttention (faster, softer VRAM spikes)
  * --enable_cache --cache_type teacache   TeaCache step-skipping (~2x less time)
  * --offloading        CPU offload (fit long/1080p renders in 80GB)
  * --sr                built-in video super-resolution -> 1080p
  * --cfg_distilled     optional ~2x speedup
  * fp8 gemm            automatic when sgl-kernel is installed (dtype bf16)

1080p = render 720p + super-resolution upscaler. Output is a silent .mp4 that the
Foley stage then scores.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import config


def _log(msg: str) -> None:
    print(f"[video] {msg}", flush=True)


def _bstr(b: bool) -> str:
    return "true" if b else "false"


def seconds_to_frames(seconds: float) -> int:
    """Map seconds to a valid HunyuanVideo frame count (4*k + 1) at FPS."""
    seconds = max(1.0, min(float(seconds), config.MAX_SECONDS))
    frames = max(5, int(round(seconds * config.FPS)))
    k = round((frames - 1) / 4)
    return int(4 * k + 1)


class VideoGenerator:
    """Thin, stateless driver around the official generate.py."""

    def generate(
        self,
        prompt: str,
        seconds: float,
        resolution: str,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        out_path: Optional[Path] = None,
    ) -> Path:
        from .models import ensure_video

        code_dir, weights_dir = ensure_video()

        resolution = resolution if resolution in config.RESOLUTION_MAP else config.DEFAULT_RESOLUTION
        res_flag, needs_sr = config.RESOLUTION_MAP[resolution]
        num_frames = seconds_to_frames(seconds)
        steps = int(steps or config.DEFAULT_STEPS)
        seed_val = int(seed) if seed is not None else 123

        out_path = out_path or (config.OUTPUT_DIR / "video_silent.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "torchrun", f"--nproc_per_node={config.NPROC}", "generate.py",
            "--prompt", prompt,
            "--negative_prompt", "",
            "--resolution", res_flag,
            "--aspect_ratio", config.DEFAULT_ASPECT,
            "--video_length", str(num_frames),
            "--num_inference_steps", str(steps),
            "--total_steps", str(steps),
            "--seed", str(seed_val),
            "--image_path", "none",
            "--model_path", str(weights_dir),
            "--output_path", str(out_path),
            "--dtype", config.DTYPE,
            "--rewrite", _bstr(config.REWRITE),
            "--cfg_distilled", _bstr(config.CFG_DISTILLED),
            "--sparse_attn", _bstr(config.SPARSE_ATTN),
            "--use_sageattn", _bstr(config.USE_SAGE_ATTENTION),
            "--sage_blocks_range", config.SAGE_BLOCKS_RANGE,
            "--enable_cache", _bstr(config.ENABLE_CACHE),
            "--cache_type", config.CACHE_TYPE,
            "--offloading", _bstr(config.OFFLOADING),
            "--overlap_group_offloading", _bstr(config.OVERLAP_GROUP_OFFLOADING),
            "--sr", _bstr(needs_sr),
        ]

        _log(f"render: {num_frames}f, res={resolution} (sr={needs_sr}), {steps} steps")
        _log("cmd: " + " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(code_dir))

        if out_path.exists():
            return out_path
        # generate.py may append its own suffix if it ignored --output_path; grab newest.
        produced = sorted(
            list(out_path.parent.rglob("*.mp4")) + list((Path(code_dir) / "outputs").rglob("*.mp4")),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not produced:
            raise RuntimeError("generate.py produced no output video")
        return produced[0]

    def unload(self) -> None:
        """No in-process model to unload — generate.py exits and frees the GPU
        by itself, so VRAM is already clean before the Foley stage."""
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
