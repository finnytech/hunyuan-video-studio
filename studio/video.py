"""HunyuanVideo 1.5 text-to-video wrapper.

Wraps the diffusers `HunyuanVideo15Pipeline` and layers on every optimization:
  * FP8 quantization of the transformer (~half the VRAM)
  * SageAttention / SSTA attention backend (30-40% faster, softer VRAM spikes)
  * TeaCache diffusion-step skipping (up to ~2x less render time)
  * CPU offload + VAE tiling (fit 16s / 1080p into 80GB)
  * 1080p via the model's super-resolution path
  * VRAM sweep so the Foley stage starts clean

The pipeline is loaded lazily and can be fully torn down to free VRAM before
the Foley stage runs.
"""
from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Optional

from . import config


def _log(msg: str) -> None:
    print(f"[video] {msg}", flush=True)


def seconds_to_frames(seconds: float) -> int:
    """Map seconds to a valid frame count for HunyuanVideo (4*k + 1)."""
    seconds = max(1.0, min(float(seconds), config.MAX_SECONDS))
    frames = int(round(seconds * config.FPS))
    # HunyuanVideo latent temporal compression wants (4k + 1) frames.
    frames = max(5, frames)
    k = round((frames - 1) / 4)
    return int(4 * k + 1)


class VideoGenerator:
    def __init__(self) -> None:
        self._pipe = None
        self._loaded_res: Optional[str] = None

    # -- lifecycle ----------------------------------------------------------
    def _select_repo(self, resolution: str) -> tuple[str, bool]:
        """Return (repo_local_dir, needs_superres) for a requested resolution."""
        from .models import ensure_video_model

        if resolution == "480p":
            return str(ensure_video_model(config.VIDEO_MODEL_T2V_480)), False
        if resolution == "720p":
            return str(ensure_video_model(config.VIDEO_MODEL_T2V_720)), False
        # 1080p -> render at 720p then super-res upscale.
        return str(ensure_video_model(config.VIDEO_MODEL_T2V_720)), True

    def _quantize(self, pipe) -> None:
        if not config.FP8_TRANSFORMER:
            return
        try:
            import torch

            # FP8 e4m3 cast of the transformer weights: big VRAM win, tiny quality cost.
            for name, module in pipe.transformer.named_modules():
                if hasattr(module, "weight") and module.weight is not None:
                    if module.weight.dtype in (torch.float16, torch.bfloat16):
                        module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
            _log("FP8 transformer quantization applied ✓")
        except Exception as e:  # noqa: BLE001
            _log(f"FP8 quant skipped ({e}); continuing in bf16")

    def _enable_sage_attention(self, pipe) -> None:
        if not config.USE_SAGE_ATTENTION:
            return
        try:
            pipe.transformer.set_attention_backend(config.ATTENTION_BACKEND)
            _log(f"attention backend -> {config.ATTENTION_BACKEND} ✓")
        except Exception as e:  # noqa: BLE001
            _log(f"sage/attention backend not set ({e})")

    def _enable_teacache(self, pipe) -> None:
        if not config.USE_TEACACHE:
            return
        try:
            # diffusers exposes cache helpers on recent versions.
            from diffusers.hooks import apply_teacache  # type: ignore

            apply_teacache(pipe.transformer, threshold=config.TEACACHE_THRESH)
            _log(f"TeaCache enabled (thresh={config.TEACACHE_THRESH}) ✓")
        except Exception as e:  # noqa: BLE001
            _log(f"TeaCache not available ({e}); skipping")

    def load(self, resolution: str) -> None:
        if self._pipe is not None and self._loaded_res == resolution:
            return
        self.unload()

        import torch
        from diffusers import HunyuanVideo15Pipeline

        repo, _ = self._select_repo(resolution)
        _log(f"loading pipeline from {repo} (bf16) ...")
        pipe = HunyuanVideo15Pipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)

        self._enable_sage_attention(pipe)
        self._quantize(pipe)
        self._enable_teacache(pipe)

        if config.CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
            _log("model CPU offload enabled ✓")
        else:
            pipe.to("cuda")
        if config.VAE_TILING:
            pipe.vae.enable_tiling()
            _log("VAE tiling enabled ✓")

        self._pipe = pipe
        self._loaded_res = resolution

    def unload(self) -> None:
        if self._pipe is not None:
            _log("unloading video pipeline + sweeping VRAM ...")
        self._pipe = None
        self._loaded_res = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass

    # -- generation ---------------------------------------------------------
    def generate(
        self,
        prompt: str,
        seconds: float,
        resolution: str,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        out_path: Optional[Path] = None,
    ) -> Path:
        import torch
        from diffusers.utils import export_to_video

        resolution = resolution if resolution in config.RESOLUTIONS else config.DEFAULT_RESOLUTION
        self.load(resolution)
        assert self._pipe is not None

        width, height = config.RESOLUTIONS[resolution]
        needs_superres = resolution == "1080p"
        gen_w, gen_h = (config.RESOLUTIONS["720p"] if needs_superres else (width, height))

        num_frames = seconds_to_frames(seconds)
        steps = steps or config.DEFAULT_STEPS
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))

        _log(f"generate: {num_frames}f @ {gen_w}x{gen_h}, {steps} steps, res={resolution}")
        result = self._pipe(
            prompt=prompt,
            width=gen_w,
            height=gen_h,
            num_frames=num_frames,
            num_inference_steps=steps,
            generator=generator,
        )
        frames = result.frames[0]

        if needs_superres:
            frames = self._super_resolve(frames, (width, height))

        out_path = out_path or (config.OUTPUT_DIR / "video_silent.mp4")
        export_to_video(frames, str(out_path), fps=config.FPS)
        _log(f"silent video written: {out_path}")
        return out_path

    def _super_resolve(self, frames, target_wh):
        """Upscale frames to 1080p. Uses the pipeline's SR module if present,
        else a high-quality Lanczos fallback (still crisp, no extra VRAM)."""
        try:
            if hasattr(self._pipe, "super_resolution") and self._pipe.super_resolution:
                _log("applying HunyuanVideo super-resolution ...")
                return self._pipe.super_resolution(frames, target_size=target_wh)
        except Exception as e:  # noqa: BLE001
            _log(f"native SR unavailable ({e}); using Lanczos upscale")
        import numpy as np
        from PIL import Image

        w, h = target_wh
        out = []
        for f in frames:
            img = f if isinstance(f, Image.Image) else Image.fromarray(np.asarray(f))
            out.append(img.resize((w, h), Image.LANCZOS))
        return out
