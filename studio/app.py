"""Gradio web UI: prompt / length / resolution -> video with native-feeling sound.

Flow per request:
  1. HunyuanVideo 1.5 renders a silent video (with all optimizations).
  2. Video pipeline is torn down -> VRAM swept clean.
  3. HunyuanVideo-Foley generates + syncs audio to the timeline.
  4. Final MP4 is normalized and offered via a download button.

The whole app sits behind a one-time token gate (see auth.py), so the public
share link can't be abused by bots.
"""
from __future__ import annotations

import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr

from . import config, foley, mux
from .auth import generate_token, make_auth_callback
from .video import VideoGenerator

# Single shared generator + a lock: one GPU, one job at a time.
_VIDEO = VideoGenerator()
_GPU_LOCK = threading.Lock()


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate(prompt: str, seconds: float, resolution: str, steps: int, seed,
             progress=gr.Progress(track_tqdm=True)):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")
    if not _GPU_LOCK.acquire(blocking=False):
        raise gr.Error("The GPU is busy with another generation. Try again shortly.")
    t0 = time.time()
    try:
        stamp = _stamp()
        work = config.OUTPUT_DIR / stamp
        work.mkdir(parents=True, exist_ok=True)
        seed_val = int(seed) if str(seed).strip() else None

        progress(0.05, desc="Rendering video (HunyuanVideo 1.5)...")
        silent = _VIDEO.generate(
            prompt=prompt,
            seconds=float(seconds),
            resolution=resolution,
            steps=int(steps),
            seed=seed_val,
            out_path=work / "silent.mp4",
        )

        # Free the video model before Foley so it starts on a clean GPU.
        progress(0.55, desc="Freeing VRAM...")
        _VIDEO.unload()

        progress(0.6, desc="Generating + syncing sound (Foley)...")
        with_sound = foley.add_foley(silent, prompt, out_dir=work / "foley")

        progress(0.9, desc="Finalizing MP4...")
        if not foley.has_audio_stream(with_sound):
            # Fallback: Foley returned audio-only or separate track.
            audio = next(iter(sorted((work / "foley").rglob("*.wav"))), None)
            if audio:
                with_sound = mux.mux(silent, audio, work / "muxed.mp4")
        final = mux.normalize_final(with_sound, work / "download.mp4")

        dt = time.time() - t0
        status = f"✅ Done in {dt:.1f}s — {seconds}s @ {resolution}"
        return str(final), str(final), status
    except gr.Error:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise gr.Error(f"Generation failed: {e}")
    finally:
        _GPU_LOCK.release()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="HunyuanVideo Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎬 HunyuanVideo Studio\n"
            "Text → video **with native-feeling sound**. "
            "HunyuanVideo 1.5 + HunyuanVideo-Foley on A100."
        )
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt", lines=3,
                    placeholder="A cinematic wave crashing on rocks at sunset, seagulls overhead",
                )
                with gr.Row():
                    seconds = gr.Slider(1, config.MAX_SECONDS, value=config.DEFAULT_SECONDS,
                                        step=1, label="Length (seconds)")
                    resolution = gr.Radio(
                        list(config.RESOLUTIONS.keys()),
                        value=config.DEFAULT_RESOLUTION, label="Resolution",
                    )
                with gr.Row():
                    steps = gr.Slider(10, 50, value=config.DEFAULT_STEPS, step=1,
                                      label="Steps (quality vs speed)")
                    seed = gr.Textbox(label="Seed (blank = random)", value="")
                go = gr.Button("Generate 🎬", variant="primary")
                status = gr.Markdown("")
            with gr.Column(scale=4):
                video_out = gr.Video(label="Result", autoplay=True)
                download = gr.DownloadButton(label="⬇️ Download MP4")

        go.click(
            generate,
            inputs=[prompt, seconds, resolution, steps, seed],
            outputs=[video_out, download, status],
            concurrency_limit=config.MAX_CONCURRENCY,
        )
    return demo


def main() -> None:
    # Warm model presence check (self-heals a fresh VM).
    try:
        from .models import ensure_all

        ensure_all()
    except Exception as e:  # noqa: BLE001
        print(f"[app] model check warning: {e}")

    token = generate_token()
    demo = build_ui()
    demo.queue(max_size=8)

    print("\n" + "=" * 62)
    print("  HunyuanVideo Studio is starting")
    print("  🔑 Access token (use as password on login):")
    print(f"       {token}")
    print("  Username can be anything.")
    print("=" * 62 + "\n", flush=True)

    demo.launch(
        server_name=config.SERVER_NAME,
        server_port=config.SERVER_PORT,
        share=config.PUBLIC_SHARE,
        auth=make_auth_callback(token),
        auth_message="Enter the access token (as password) to use the studio.",
        show_api=False,
        max_threads=config.MAX_CONCURRENCY + 2,
    )


if __name__ == "__main__":
    main()
