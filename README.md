# 🎬 HunyuanVideo Studio — Video + native Foley sound, one click

Text → **HunyuanVideo 1.5** (up to 16 s, up to 1080p) → **HunyuanVideo-Foley** generates
matching sound locked to the timeline → muxed MP4 with a **download button** in a web UI.

Built for **Lightning.ai** on a single **A100 80 GB** (12 vCPU, ~312 TFLOPS).
Heavily VRAM/CPU‑optimized: **FP8/GGUF quant**, **SageAttention (SSTA)**, **TeaCache**,
model CPU‑offload, VAE tiling, and aggressive VRAM cleanup between stages.

The public share link is **locked behind a one‑time OAuth‑style token**, so random bots
can't spam the site and burn your GPU hours — only you get in.

---

## 0. TL;DR — run on the Lightning.ai A100 VM

```bash
git clone https://github.com/finnytech/hunyuan-video-studio.git
cd hunyuan-video-studio
export HF_TOKEN=hf_xxx           # optional, speeds up / avoids rate limits
bash scripts/setup.sh            # smart: only downloads what's missing
bash scripts/run.sh              # prints your PRIVATE share URL + access token
```

`run.sh` prints something like:

```
🔗 Public share link : https://<random>.gradio.live
🔑 Access token      : 7Kd9...  (paste this on the login screen)
```

Open the link, paste the token once, and you're in. That's the whole flow.

---

## 1. What the pipeline does

```
prompt ──► HunyuanVideo 1.5 (t2v)  ──► silent .mp4  ─┐
   │                                                 ├─► ffmpeg mux ─► final .mp4 (video+sound)
   └──────► same prompt as audio hint ──► Foley  ────┘        (audio auto‑aligned to timeline)
```

- **Length**: 1–16 s (mapped to valid frame counts at 24 fps internally).
- **Resolution**: 480p / 720p / 1080p (1080p via the built‑in super‑res upscaler).
- **Sound**: HunyuanVideo‑Foley watches the generated video + reads the prompt, and
  emits 48 kHz audio synced to the action — so the result sounds like native sound, not a
  bolted‑on loop.

## 2. Optimizations (why it fits and stays cheap)

| Step | Effect | How |
|------|--------|-----|
| **FP8 / GGUF quant** | ~½ the VRAM | transformer weights cast to FP8 (`FP8_TRANSFORMER=1`) — HunyuanVideo 1.5 quality stays near‑identical |
| **SageAttention (SSTA)** | 30–40% faster | INT8/sparse attention kernel; also flattens VRAM spikes |
| **TeaCache** | up to ~2× faster | skips recomputing near‑unchanged diffusion steps → less GPU time = less money |
| **CPU offload + VAE tiling** | fits 16 s in 80 GB | weights streamed to CPU when idle, VAE decodes in tiles |
| **VRAM sweep** | no OOM between stages | `torch.cuda.empty_cache()` + model teardown between video → foley |

All toggles live in `studio/config.py` (env‑overridable). Sensible A100 defaults are on.

## 3. Security model

- Gradio `share=True` gives a public `*.gradio.live` URL, but the app is wrapped in a
  **token gate**. The token is generated fresh on every launch (or pinned via `STUDIO_TOKEN`).
- Wrong token → no access to generation, no GPU spent. One valid session = you.
- Add `--auth-only-local` in `run.sh` to skip the public link entirely and keep it on
  `127.0.0.1` behind an SSH tunnel.

## 4. Files

```
scripts/setup.sh   smart installer (idempotent; checks models before downloading)
scripts/run.sh     launches the UI + prints private share link & token
studio/config.py   all knobs: model ids, paths, optimization toggles
studio/models.py   model presence check + downloads (HF)
studio/video.py    HunyuanVideo 1.5 wrapper (quant + SageAttn + TeaCache + offload)
studio/foley.py    HunyuanVideo-Foley wrapper (calls the official infer.py)
studio/mux.py      ffmpeg video+audio mux, timeline‑aligned
studio/auth.py     one‑time token OAuth‑style gate
studio/app.py      Gradio UI (prompt / length / resolution → video + download)
```

## 5. Requirements

A100/L40S‑class GPU, CUDA 12.4 (or 11.8), Python 3.10+, Linux, ffmpeg. `setup.sh` handles the rest.
