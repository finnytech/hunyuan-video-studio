#!/usr/bin/env bash
# Launch the studio UI. Prints a PRIVATE public share link + access token.
#
#   bash scripts/run.sh                      # public *.gradio.live link, token-gated
#   bash scripts/run.sh --auth-only-local    # local only (127.0.0.1), no public link
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Activate .venv if present, else use the active env (Lightning conda base).
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"

# A100-friendly runtime defaults (override by exporting before calling run.sh).
export USE_SAGE_ATTENTION="${USE_SAGE_ATTENTION:-1}"   # SageAttention
export ENABLE_CACHE="${ENABLE_CACHE:-1}"               # TeaCache step-skipping
export CACHE_TYPE="${CACHE_TYPE:-teacache}"
# OFFLOADING: leave UNSET by default so the app's SMART mode decides (resident
# for light renders = faster; offload only for 1080p/long clips). Export
# OFFLOADING=1/0 before run.sh to force it either way.
[ -n "${OFFLOADING:-}" ] && export OFFLOADING
export OVERLAP_GROUP_OFFLOADING="${OVERLAP_GROUP_OFFLOADING:-1}"
export CFG_DISTILLED="${CFG_DISTILLED:-0}"             # set 1 for ~2x speed (needs distilled weights)
export SPARSE_ATTN="${SPARSE_ATTN:-0}"                 # OFF: SSTA needs H-series GPUs
export REWRITE="${REWRITE:-0}"                         # OFF: prompt-rewrite needs a vLLM server
export STUDIO_DTYPE="${STUDIO_DTYPE:-bf16}"            # bf16 + sgl-kernel => FP8 gemm
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"   # fast HF downloads

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "⚠️  HF_TOKEN not set — HF downloads will be rate-limited. export HF_TOKEN=hf_xxx"
fi

if [ "${1:-}" = "--auth-only-local" ]; then
  export STUDIO_SHARE=0
  export STUDIO_HOST=127.0.0.1
  echo "==> local-only mode (no public link). Use an SSH tunnel to reach it."
fi

# --- self-healing preflight -----------------------------------------------
# preflight exit codes: 0 = all present -> launch; 1 = CORE missing -> setup;
# 2 = only OPTIONAL missing (e.g. foley) -> setup once (sentinel), else launch
# anyway so a stubborn optional build can't block the app forever.
PF_RC=0
"$PYBIN" "$ROOT/scripts/preflight.py" || PF_RC=$?
if [ "${STUDIO_NO_AUTOSETUP:-0}" = "1" ]; then
  [ "$PF_RC" != "0" ] && echo "!! missing pieces but STUDIO_NO_AUTOSETUP=1 -> skipping. Run: bash scripts/setup.sh"
elif [ "$PF_RC" = "1" ]; then
  echo "==> core pieces missing -> running setup (installs only what's missing)"
  bash "$ROOT/scripts/setup.sh"
elif [ "$PF_RC" = "2" ]; then
  if [ ! -f "$ROOT/.setup-complete" ]; then
    echo "==> optional pieces missing (first run) -> running setup once"
    bash "$ROOT/scripts/setup.sh"
  else
    echo "==> optional pieces missing but setup already ran -> launching anyway (sound may be off)"
  fi
fi

echo "==> launching HunyuanVideo Studio (env: $STUDIO_ENV_KIND; watch for the link + token below)"
exec "$PYBIN" -m studio.app
