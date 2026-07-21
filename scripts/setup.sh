#!/usr/bin/env bash
# Smart, idempotent installer for HunyuanVideo Studio.
# Works on Lightning.ai Studios (single conda env, no venvs) AND on plain VMs
# (creates a .venv). Safe to re-run: only does the work that's actually missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "==> HunyuanVideo Studio setup in $ROOT"

PY="${PYTHON:-python3}"
VENV="$ROOT/.venv"
MODELS_DIR="${STUDIO_MODELS_DIR:-$ROOT/models}"

# --- 0. decide environment: venv vs current (Lightning forbids venvs) ------
USE_VENV=1
if [ -n "${STUDIO_NO_VENV:-}" ] || [ -d /teamspace/studios ] || [ -n "${CONDA_PREFIX:-}" ]; then
  USE_VENV=0
  echo "==> detected managed/conda environment -> using the ACTIVE env (no venv)"
fi

# --- 1. system deps -------------------------------------------------------
# ffmpeg binary + git-lfs + ffmpeg DEV headers & pkg-config (needed to build
# PyAV / some audio deps from source; cheap no-op if already installed).
APT_PKGS=""
command -v ffmpeg     >/dev/null 2>&1 || APT_PKGS="$APT_PKGS ffmpeg"
command -v git-lfs    >/dev/null 2>&1 || APT_PKGS="$APT_PKGS git-lfs"
command -v pkg-config >/dev/null 2>&1 || APT_PKGS="$APT_PKGS pkg-config"
APT_PKGS="$APT_PKGS libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev"
if [ -n "$APT_PKGS" ]; then
  echo "==> installing system deps:$APT_PKGS"
  # shellcheck disable=SC2086
  (sudo apt-get update -y && sudo apt-get install -y $APT_PKGS) \
    || echo "!! apt-get failed for some packages (video still works; check logs)"
fi
command -v git-lfs >/dev/null 2>&1 && git lfs install || true

# --- 2. python environment ------------------------------------------------
if [ "$USE_VENV" = 1 ]; then
  if [ ! -d "$VENV" ]; then
    echo "==> creating venv"
    if ! "$PY" -m venv "$VENV" 2>/dev/null; then
      echo "!! venv creation not allowed -> falling back to ACTIVE env"
      USE_VENV=0
    fi
  fi
fi
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"
echo "==> using python: $PYBIN  (env kind: $STUDIO_ENV_KIND)"
"$PYBIN" -m pip install --upgrade pip wheel setuptools

PIP() { "$PYBIN" -m pip "$@"; }

# --- 3. torch (CUDA 12.4) — only if missing -------------------------------
if ! "$PYBIN" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "==> installing torch (cu124)"
  PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
  echo "==> torch + CUDA already present ✓"
fi

# --- 4. studio (UI) requirements -----------------------------------------
echo "==> installing studio requirements"
PIP install -r requirements.txt

# --- 5. clone official repos + download weights (idempotent) --------------
echo "==> ensuring HunyuanVideo-1.5 and HunyuanVideo-Foley (code + weights)"
"$PYBIN" -m studio.models

# --- 6. install each model's own requirements -----------------------------
# Strip flash_attn from the pinned requirements: it builds from source and needs
# torch importable in the build env; under pip build isolation it fails and that
# ONE failure aborts the whole -r install (leaving diffusers/transformers/etc
# uninstalled). We install flash-attn separately, best-effort, in step 7.
for d in "$MODELS_DIR/HunyuanVideo-1.5" "$MODELS_DIR/HunyuanVideo-Foley"; do
  if [ -f "$d/requirements.txt" ]; then
    echo "==> installing requirements for $(basename "$d") (flash_attn handled separately)"
    REQ_FILTERED="$(mktemp)"
    grep -viE '^\s*flash[-_]attn' "$d/requirements.txt" > "$REQ_FILTERED" || true
    PIP install -r "$REQ_FILTERED" || echo "!! some deps in $d failed; check logs"
    rm -f "$REQ_FILTERED"
  fi
done

# --- 7. speed kernels (best-effort; failures never block) -----------------
echo "==> installing SageAttention (official fork)"
if ! "$PYBIN" -c "import sageattention" 2>/dev/null; then
  ( git clone https://github.com/cooper1637/SageAttention.git "$MODELS_DIR/SageAttention" 2>/dev/null || true
    cd "$MODELS_DIR/SageAttention" 2>/dev/null && \
    EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 "$PYBIN" setup.py install ) \
    || echo "   (SageAttention build skipped — app still runs, just without it)"
fi

echo "==> installing sgl-kernel (enables FP8 gemm)"
PIP install sgl-kernel==0.3.18 2>/dev/null || echo "   (sgl-kernel optional — skipped)"

if "$PYBIN" -c "import flash_attn" 2>/dev/null; then
  echo "==> flash-attn already present ✓"
else
  echo "==> installing flash-attn (best-effort)"
  PIP install flash-attn --no-build-isolation 2>/dev/null || echo "   (flash-attn optional — skipped)"
fi

# Sentinel so run.sh won't loop re-running setup for a stubborn optional dep.
touch "$ROOT/.setup-complete"

echo ""
echo "==> setup complete ✓ (env: $STUDIO_ENV_KIND)   run:  bash scripts/run.sh"
