#!/usr/bin/env bash
# Delete the (possibly huge / partial) video weights and re-download ONLY the
# minimal set of variants, with hf_transfer for max speed.
#
#   bash scripts/reset_weights.sh          # wipe video weights + HF cache, re-fetch minimal
#   KEEP_FOLEY=0 bash scripts/reset_weights.sh   # also wipe Foley weights
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"

MODELS_DIR="${STUDIO_MODELS_DIR:-$ROOT/models}"

echo "==> removing partial/oversized video weights"
rm -rf "$MODELS_DIR/HunyuanVideo-1.5-weights"

# Also clear the HF hub cache for this repo (that's where the 372GB blobs live).
echo "==> clearing HuggingFace hub cache for HunyuanVideo-1.5"
rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--tencent--HunyuanVideo-1.5" 2>/dev/null || true
rm -rf "$HOME/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5" 2>/dev/null || true

if [ "${KEEP_FOLEY:-1}" = "0" ]; then
  echo "==> removing Foley weights too"
  rm -rf "$MODELS_DIR/HunyuanVideo-Foley-weights"
  rm -rf "$HOME/.cache/huggingface/hub/models--tencent--HunyuanVideo-Foley" 2>/dev/null || true
fi

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "⚠️  HF_TOKEN not set — set it for faster, un-throttled downloads: export HF_TOKEN=hf_xxx"
fi

echo "==> re-downloading MINIMAL weights (variants: ${STUDIO_VIDEO_VARIANTS:-720p_t2v,1080p_sr_distilled})"
"$PYBIN" -m studio.models
echo "==> done ✓  disk usage:"
du -sh "$MODELS_DIR"/* 2>/dev/null || true
