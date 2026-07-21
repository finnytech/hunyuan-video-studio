# 🎬 HunyuanVideo Studio — video + native Foley sound, one click

Text → **HunyuanVideo 1.5** (up to 16 s, up to 1080p) → **HunyuanVideo-Foley** generates
matching sound locked to the timeline → muxed MP4 with a **download button** in a web UI.

Runs on a single **A100 80 GB** (12 vCPU, ~312 TFLOPS) on **Lightning.ai**.
The public share link is locked behind a **one-time OAuth-style token**, so random bots
can't spam the endpoint and burn your GPU hours — only you get in.

> **Not a toy.** The video stage drives Tencent's **official `generate.py`** and the sound
> stage drives the **official `infer.py`**. Every optimization is a real, tested flag from
> the upstream repos — not a reimplementation.

---

## 0. Run on the Lightning.ai A100 VM

```bash
git clone https://github.com/finnytech/hunyuan-video-studio.git
cd hunyuan-video-studio
export HF_TOKEN=***           # for HuggingFace weight downloads
bash scripts/setup.sh            # smart: clones repos + downloads only missing weights
bash scripts/run.sh              # prints your PRIVATE share URL + access token
```

`run.sh` prints:

```
🔗 Public share link : https://<random>.gradio.live
🔑 Access token      : 7Kd9...   (paste as the password on the login screen)
```

Open the link, paste the token once (username can be anything), and you're in.

---

## 1. Pipeline

```
prompt ─► HunyuanVideo 1.5  (official generate.py, torchrun) ─► silent .mp4
                                                                    │
                       generate.py exits ─► GPU VRAM is clean       │
                                                                    ▼
prompt (as audio hint) ─► HunyuanVideo-Foley (official infer.py) ─► video WITH sound
                                                                    ▼
                                     ffmpeg normalize ─► download.mp4 (H.264+AAC, faststart)
```

- **Length**: 1–16 s (mapped to valid `4k+1` frame counts at 24 fps).
- **Resolution**: 480p / 720p native; **1080p** via the model's built-in super-resolution
  (`--sr true`, renders 720p then upscales).
- **Sound**: Foley watches the generated video + reads the prompt and emits 48 kHz audio
  synced to the action — sounds like native sound, not a bolted-on loop.

## 2. Optimizations (real generate.py flags)

| Flag | Effect | Notes |
|------|--------|-------|
| `--use_sageattn true` | faster inference, softer VRAM spikes | SageAttention (auto-disables flex-block) |
| `--enable_cache --cache_type teacache` | up to ~2× less render time | skips near-unchanged diffusion steps |
| `--offloading true` + group offload | fits long / 1080p renders in 80 GB | weights streamed to CPU when idle |
| FP8 gemm | ~½ transformer VRAM | auto when `sgl-kernel` is installed (dtype bf16) |
| `--sr true` | 1080p output | built-in video super-resolution network |
| `--cfg_distilled` *(optional)* | ~2× speedup | enable via `CFG_DISTILLED=1` |

`--sparse_attn` (SSTA) is **off** — it needs H-series GPUs. `--rewrite` is **off** — prompt
rewriting needs a separate vLLM server; the studio stays self-contained. All toggles live in
`studio/config.py` and are env-overridable from `run.sh`.

**Extra speed + robustness layer:**
- **TF32 matmul + cuDNN autotune** and **CPU-thread tuning** (10 of 12 vCPU) via the runner's child env.
- **TeaCache tuned** (`cache_start_step / end_step / step_interval`) instead of default on/off.
- Each stage runs as its **own process** → 100% of the video model's VRAM is reclaimed by the OS
  before Foley starts. No in-process contention.
- **Hard timeouts + one automatic retry** per stage; full stdout/stderr saved to `outputs/logs/`
  for post-mortem.
- **Reveal-when-done UI**: nothing streams to the browser mid-render — the video appears (and the
  download button unlocks) only after video **and** sound are finished and the audio track is verified.
- Finished MP4s are collected in `outputs/videos/hunyuan_<timestamp>.mp4` on the VM.

## 3. Security model

- Gradio `share=True` gives a public `*.gradio.live` URL, but the app is wrapped in a
  **token gate** (`studio/auth.py`, constant-time compare). Fresh token every launch, or pin
  one via `STUDIO_TOKEN`.
- Wrong token → no access, no GPU spent. One valid session = you.
- `bash scripts/run.sh --auth-only-local` skips the public link entirely (127.0.0.1 + SSH tunnel).
- `MAX_CONCURRENCY=1` + a GPU lock means requests can never pile up on the single A100.

## 4. Files

```
scripts/setup.sh   smart installer: venv, torch, repos, weights, speed kernels (idempotent)
scripts/run.sh     launches the UI + prints the private share link & token
studio/config.py   all knobs: repos, weights, resolution map, optimization flags
studio/models.py   clone official repos + download weights (only what's missing)
studio/video.py    HunyuanVideo 1.5 driver (official generate.py via torchrun)
studio/foley.py    HunyuanVideo-Foley driver (official infer.py)
studio/mux.py      ffmpeg mux + browser-friendly normalize
studio/auth.py     one-time token OAuth-style gate
studio/app.py      Gradio UI (prompt / length / resolution → video + download)
```

## 5. Requirements

A100/L40S-class GPU, CUDA 12.4 (or 11.8), Python 3.10+, Linux, ffmpeg, git-lfs.
`setup.sh` handles the venv, torch, both official repos, their weights, and the speed kernels.
