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
export OFFLOADING="${OFFLOADING:-1}"                   # CPU offload
export OVERLAP_GROUP_OFFLOADING="${OVERLAP_GROUP_OFFLOADING:-1}"
export CFG_DISTILLED="${CFG_DISTILLED:-0}"             # set 1 for ~2x speed (needs distilled weights)
export SPARSE_ATTN="${SPARSE_ATTN:-0}"                 # OFF: SSTA needs H-series GPUs
export REWRITE="${REWRITE:-0}"                         # OFF: prompt-rewrite needs a vLLM server
export STUDIO_DTYPE="${STUDIO_DTYPE:-bf16}"            # bf16 + sgl-kernel => FP8 gemm
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

if [ "${1:-}" = "--auth-only-local" ]; then
  export STUDIO_SHARE=0
  export STUDIO_HOST=127.0.0.1
  echo "==> local-only mode (no public link). Use an SSH tunnel to reach it."
fi

echo "==> launching HunyuanVideo Studio (env: $STUDIO_ENV_KIND; watch for the link + token below)"
exec "$PYBIN" -m studio.app
