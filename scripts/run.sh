#!/usr/bin/env bash
# Launch the studio UI. Prints a PRIVATE public share link + access token.
#
#   bash scripts/run.sh                 # public *.gradio.live link, token-gated
#   bash scripts/run.sh --auth-only-local   # local only (127.0.0.1), no public link
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

# A100-friendly runtime env (can be overridden before calling run.sh).
export FP8_TRANSFORMER="${FP8_TRANSFORMER:-1}"
export USE_SAGE_ATTENTION="${USE_SAGE_ATTENTION:-1}"
export USE_TEACACHE="${USE_TEACACHE:-1}"
export CPU_OFFLOAD="${CPU_OFFLOAD:-1}"
export VAE_TILING="${VAE_TILING:-1}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash_hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

if [ "${1:-}" = "--auth-only-local" ]; then
  export STUDIO_SHARE=0
  export STUDIO_HOST=127.0.0.1
  echo "==> local-only mode (no public link). Use an SSH tunnel to reach it."
fi

echo "==> launching HunyuanVideo Studio (watch for the share link + token below)"
exec python -m studio.app
