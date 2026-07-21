#!/usr/bin/env bash
# Smart, idempotent installer for HunyuanVideo Studio on a Lightning.ai A100 VM.
# Safe to re-run: it only does the work that's actually missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "==> HunyuanVideo Studio setup in $ROOT"

PY="${PYTHON:-python3}"
VENV="$ROOT/.venv"

# --- 1. system deps -------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "==> installing ffmpeg"
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
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "==> installing torch (cu124)"
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
  echo "==> torch + CUDA already present ✓"
fi

# --- 4. studio requirements ----------------------------------------------
echo "==> installing studio requirements"
pip install -r requirements.txt

# --- 5. optional speed kernels (best-effort) ------------------------------
echo "==> installing optional speed kernels (SageAttention / flash-attn)"
pip install sageattention 2>/dev/null || echo "   (sageattention optional — skipped)"
pip install flash-attn --no-build-isolation 2>/dev/null || echo "   (flash-attn optional — skipped)"

# --- 6. models (check-then-download) --------------------------------------
echo "==> ensuring models are present (downloads only what's missing)"
python -m studio.models

# --- 7. install Foley's own requirements ----------------------------------
FOLEY_DIR="${STUDIO_MODELS_DIR:-$ROOT/models}/HunyuanVideo-Foley"
if [ -f "$FOLEY_DIR/requirements.txt" ]; then
  echo "==> installing Foley requirements"
  pip install -r "$FOLEY_DIR/requirements.txt" || echo "!! some Foley deps failed; check logs"
fi

echo ""
echo "==> setup complete ✓   run:  bash scripts/run.sh"
