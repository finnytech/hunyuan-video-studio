"""HunyuanVideo 1.5 text-to-video via Tencent's official generate.py.

Driven with torchrun so every optimization is a real, tested upstream flag:

  * --use_sageattn                         SageAttention (faster, softer VRAM spikes)
  * --enable_cache --cache_type teacache   TeaCache step-skipping (~2x less time),
                                           with tuned start/end/interval
  * --offloading / group offload           fit long / 1080p renders in 80GB
  * --sr                                    built-in video super-resolution -> 1080p
  * --cfg_distilled (optional)             ~2x speedup
  * fp8 gemm                                automatic when sgl-kernel is installed
  * TF32 + CPU-thread tuning                via the runner's child env

Each render is its own process. When generate.py exits, the OS reclaims ALL of its
VRAM, so the video model is fully out of VRAM before the Foley stage starts — no
contention. The finished silent video is saved locally in the videos folder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import config, runner


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
    """Stateless driver around the official generate.py."""

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

        out_path = Path(out_path) if out_path else (config.OUTPUT_DIR / "video_silent.mp4")
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
            "--cache_start_step", str(config.CACHE_START_STEP),
            "--cache_end_step", str(config.CACHE_END_STEP),
            "--cache_step_interval", str(config.CACHE_STEP_INTERVAL),
            "--offloading", _bstr(config.OFFLOADING),
            "--overlap_group_offloading", _bstr(config.OVERLAP_GROUP_OFFLOADING),
            "--sr", _bstr(needs_sr),
        ]

        _log(f"render: {num_frames}f, res={resolution} (sr={needs_sr}), {steps} steps, seed={seed_val}")
        runner.run(
            cmd, cwd=code_dir, stage="video",
            timeout=config.VIDEO_TIMEOUT, retries=config.STAGE_RETRIES,
        )

        result = self._resolve_output(out_path, code_dir)
        _log(f"silent video ready: {result}")
        return result

    def _resolve_output(self, out_path: Path, code_dir: Path) -> Path:
        """generate.py may honour --output_path or append its own name; find the file."""
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        search_dirs = [out_path.parent, Path(code_dir) / "outputs"]
        produced = []
        for d in search_dirs:
            if d.exists():
                produced += [p for p in d.rglob("*.mp4") if p.stat().st_size > 0]
        if not produced:
            raise RuntimeError("generate.py produced no output video (see outputs/logs)")
        newest = max(produced, key=lambda p: p.stat().st_mtime)
        if newest != out_path:
            try:
                newest.replace(out_path)
                return out_path
            except Exception:  # noqa: BLE001
                return newest
        return newest

    def unload(self) -> None:
        """generate.py runs as a separate process, so its VRAM is already fully
        released on exit. We still sweep our own (tiny) process for good measure."""
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
