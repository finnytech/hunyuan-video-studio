#!/usr/bin/env bash
# Smart, idempotent installer for HunyuanVideo Studio on a Lightning.ai A100 VM.
# Safe to re-run: only does the work that's actually missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "==> HunyuanVideo Studio setup in $ROOT"

PY="${PYTHON:-python3}"
VENV="$ROOT/.venv"
MODELS_DIR="${STUDIO_MODELS_DIR:-$ROOT/models}"

# --- 1. system deps -------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "==> installing ffmpeg + git-lfs"
  (sudo apt-get update -y && sudo apt-get install -y ffmpeg git-lfs) \
    || echo "!! could not apt-get ffmpeg; ensure it's available"
fi
command -v git-lfs >/dev/null 2>&1 && git lfs install || true

# --- 2. python venv -------------------------------------------------------
if [ ! -d "$VENV" ]; then
  echo "==> creating venv"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# --- 3. torch (CUDA 12.4) — only if missing -------------------------------
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "==> installing torch (cu124)"
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
  echo "==> torch + CUDA already present ✓"
fi

# --- 4. studio (UI) requirements -----------------------------------------
echo "==> installing studio requirements"
pip install -r requirements.txt

# --- 5. clone official repos + download weights (idempotent) --------------
echo "==> ensuring HunyuanVideo-1.5 and HunyuanVideo-Foley (code + weights)"
python -m studio.models

# --- 6. install each model's own requirements -----------------------------
for d in "$MODELS_DIR/HunyuanVideo-1.5" "$MODELS_DIR/HunyuanVideo-Foley"; do
  if [ -f "$d/requirements.txt" ]; then
    echo "==> installing requirements for $(basename "$d")"
    pip install -r "$d/requirements.txt" || echo "!! some deps in $d failed; check logs"
  fi
done

# --- 7. speed kernels (best-effort; failures don't block) -----------------
echo "==> installing SageAttention (official fork)"
if ! python -c "import sageattention" 2>/dev/null; then
  ( git clone https://github.com/cooper1637/SageAttention.git "$MODELS_DIR/SageAttention" 2>/dev/null || true
    cd "$MODELS_DIR/SageAttention" 2>/dev/null && \
    EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 python setup.py install ) \
    || echo "   (SageAttention build skipped — app still runs, just without it)"
fi

echo "==> installing sgl-kernel (enables FP8 gemm)"
pip install sgl-kernel==0.3.18 2>/dev/null || echo "   (sgl-kernel optional — skipped)"

echo "==> installing flash-attn (best-effort)"
pip install flash-attn --no-build-isolation 2>/dev/null || echo "   (flash-attn optional — skipped)"

echo ""
echo "==> setup complete ✓   run:  bash scripts/run.sh"
