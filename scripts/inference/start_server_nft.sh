#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Start NFT inference server (single GPU)
# ──────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   bash scripts/inference/start_server_nft.sh [PORT] [CUDA_DEV] [NFT_LORA_PATH] [LORA_ALPHA]
#
# Examples:
#   # Default (same as vidar/start_server.sh, with IDM included):
#   bash scripts/inference/start_server_nft.sh 25400 0
#
#   # Different GPU:
#   bash scripts/inference/start_server_nft.sh 25401 1
#
#   # Custom NFT LoRA checkpoint:
#   bash scripts/inference/start_server_nft.sh 25400 0 data/outputs/nft_put_object_cabinet/checkpoints/lora_step000100.pt 1.0
#
#   # Without IDM:
#   IDM_PATH="" bash scripts/inference/start_server_nft.sh 25400 0

PORT=${1:-25400}
CUDA_DEV=${2:-0}

# NFT LoRA checkpoint
export NFT_LORA_PATH="${3:-data/outputs/nft_put_object_cabinet/checkpoints/lora_final.pt}"
export LORA_ALPHA="${4:-1.0}"

# Model paths
export CKPT_DIR="${CKPT_DIR:-ckpts/Wan2.2-TI2V-5B}"
export PT_DIR="${PT_DIR:-ckpts/vidar_ckpts/vidar_merged_lora.pt}"

# Device
export DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=${CUDA_DEV}

# Optional settings
export T5_CPU="${T5_CPU:-false}"
export OFFLOAD_MODEL="${OFFLOAD_MODEL:-false}"
export SIZE="${SIZE:-640*736}"
export FRAME_NUM="${FRAME_NUM:-121}"

# IDM model (same default as vidar/start_server.sh)
export IDM_PATH="${IDM_PATH:-ckpts/vidar_ckpts/idm.pt}"
export IDM_CPU="${IDM_CPU:-false}"

# Video saving (optional)
export SAVE_VIDEO="${SAVE_VIDEO:-false}"
export SAVE_VIDEO_DIR="${SAVE_VIDEO_DIR:-/tmp/nft_server_videos}"

# ── Port check ────────────────────────────────────────────────────────────────
python3 - <<'PY'
import socket, os, sys
port = int(os.environ.get("PORT", "25400"))
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.settimeout(1)
    if s.connect_ex(("127.0.0.1", port)) == 0:
        print(f"[ERROR] Port {port} already in use.", file=sys.stderr)
        sys.exit(1)
PY
export PORT=${PORT}
if [[ $? -ne 0 ]]; then
    echo "[ERROR] Port ${PORT} already in use."
    exit 1
fi

echo "=============================================="
echo " NFT Inference Server"
echo "=============================================="
echo " PORT:            ${PORT}"
echo " CUDA_DEV:        ${CUDA_DEV}"
echo " NFT_LORA_PATH:   ${NFT_LORA_PATH}"
echo " LORA_ALPHA:      ${LORA_ALPHA}"
echo " CKPT_DIR:        ${CKPT_DIR}"
echo " PT_DIR:          ${PT_DIR}"
echo " T5_CPU:          ${T5_CPU}"
echo " IDM_PATH:        ${IDM_PATH:-<not set>}"
echo " SAVE_VIDEO:      ${SAVE_VIDEO}"
echo "=============================================="

# ── Launch server ─────────────────────────────────────────────────────────────
if command -v uvicorn >/dev/null 2>&1; then
    exec uvicorn fastvideo.server_nft:api --host 0.0.0.0 --port ${PORT} --workers 1
else
    exec python -m uvicorn fastvideo.server_nft:api --host 0.0.0.0 --port ${PORT} --workers 1
fi
