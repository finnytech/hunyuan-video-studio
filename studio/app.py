"""Gradio web UI: prompt / length / resolution -> video with native sound.

Order of operations (exactly what you asked for):
  1. HunyuanVideo 1.5 renders the video (own process; full VRAM to the video model).
  2. That process EXITS -> its VRAM is fully released. Silent video saved locally.
  3. HunyuanVideo-Foley generates + syncs native sound to the timeline.
  4. Only once BOTH video and sound are done is the finished MP4 revealed in the
     browser (streamed) and the download button shown.

Nothing is streamed to the web mid-render. The whole app sits behind a one-time
token gate so bots can't spam the endpoint.
"""
from __future__ import annotations

import shutil
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr

from . import config, foley, mux
from .auth import generate_token, make_auth_callback
from .video import VideoGenerator

_VIDEO = VideoGenerator()
_GPU_LOCK = threading.Lock()          # one GPU -> one job at a time


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _busy_ui(msg: str):
    # status, video(hidden/empty), download(hidden), generate-button(disabled)
    return (msg, None, gr.update(visible=False), gr.update(interactive=False))


def _done_ui(msg: str, final_path: str):
    return (
        msg,
        final_path,                                   # reveal + stream the finished MP4
        gr.update(value=final_path, visible=True),    # show download button
        gr.update(interactive=True),
    )


def generate(prompt, seconds, resolution, steps, seed):
    """Generator: streams status; only reveals the video when fully finished."""
    prompt = (prompt or "").strip()
    if not prompt:
        yield ("⚠️ Please enter a prompt.", None,
               gr.update(visible=False), gr.update(interactive=True))
        return

    if not _GPU_LOCK.acquire(blocking=False):
        yield ("⏳ The GPU is busy with another generation. Try again shortly.", None,
               gr.update(visible=False), gr.update(interactive=True))
        return

    t0 = time.time()
    try:
        yield _busy_ui("🎬 Stage 1/3 — rendering video (HunyuanVideo 1.5)…")
        stamp = _stamp()
        work = config.OUTPUT_DIR / stamp
        work.mkdir(parents=True, exist_ok=True)
        seed_val = int(seed) if str(seed).strip() else None

        # 1) video -> saved locally in the work/videos folder
        silent = _VIDEO.generate(
            prompt=prompt, seconds=float(seconds), resolution=resolution,
            steps=int(steps), seed=seed_val, out_path=work / "silent.mp4",
        )
        t_video = time.time() - t0

        # 2) make sure the video model is out of VRAM before sound
        yield _busy_ui(f"🧹 Stage 2/3 — video done in {t_video:.0f}s, freeing VRAM…")
        _VIDEO.unload()

        # 3) native, timeline-synced sound
        yield _busy_ui("🔊 Stage 3/3 — generating + syncing native sound (Foley)…")
        with_sound = foley.add_foley(silent, prompt, out_dir=work / "foley")

        # ensure audio is actually present + aligned; fallback to explicit mux
        if not foley.has_audio_stream(with_sound):
            audio = foley.newest_audio(work / "foley")
            if not audio:
                raise RuntimeError("Foley returned no audio track")
            with_sound = mux.mux(silent, audio, work / "muxed.mp4")

        # normalize to a browser/download-friendly MP4 (H.264 + AAC + faststart)
        final = mux.normalize_final(with_sound, work / "final.mp4")

        # collect the finished file in the videos folder with a clean name
        dest = config.VIDEOS_DIR / f"hunyuan_{stamp}.mp4"
        shutil.copy2(final, dest)

        dt = time.time() - t0
        msg = (f"✅ Done in {dt:.0f}s  ·  video {t_video:.0f}s + sound "
               f"{dt - t_video:.0f}s  ·  {int(float(seconds))}s @ {resolution}")
        yield _done_ui(msg, str(dest))

    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        yield (f"❌ Generation failed: {e}", None,
               gr.update(visible=False), gr.update(interactive=True))
    finally:
        _GPU_LOCK.release()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="HunyuanVideo Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎬 HunyuanVideo Studio\n"
            "Text → video **with native, timeline-synced sound**. "
            "HunyuanVideo 1.5 + HunyuanVideo-Foley on A100. "
            "_The video appears only once both video and sound are finished._"
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
                    _res_choices = config.available_resolutions(config.VIDEO_VARIANTS)
                    _res_default = (config.DEFAULT_RESOLUTION
                                    if config.DEFAULT_RESOLUTION in _res_choices
                                    else _res_choices[0])
                    resolution = gr.Radio(
                        _res_choices, value=_res_default, label="Resolution",
                    )
                with gr.Row():
                    steps = gr.Slider(20, 50, value=config.DEFAULT_STEPS, step=1,
                                      label="Steps (50 = best quality)")
                    seed = gr.Textbox(label="Seed (blank = random)", value="")
                go = gr.Button("Generate 🎬", variant="primary")
                status = gr.Markdown("")
            with gr.Column(scale=4):
                video_out = gr.Video(label="Result (video + native sound)", autoplay=True)
                download = gr.DownloadButton(label="⬇️ Download MP4", visible=False)

        go.click(
            generate,
            inputs=[prompt, seconds, resolution, steps, seed],
            outputs=[status, video_out, download, go],
            concurrency_limit=config.MAX_CONCURRENCY,
        )
    return demo


def main() -> None:
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
    print("  🔑 Access token (use as the PASSWORD on login):")
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
