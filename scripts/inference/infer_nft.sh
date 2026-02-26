#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Inference with Wan2.2 TI2V + DiffusionNFT LoRA checkpoint
# ──────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   # Single video:
#   bash scripts/inference/infer_nft.sh
#
#   # Override NFT checkpoint:
#   NFT_LORA_PATH=data/outputs/nft_put_object_cabinet/checkpoints/lora_step000100.pt \
#       bash scripts/inference/infer_nft.sh
#
#   # Custom prompt + image:
#   PROMPT="A robot opens the cabinet." IMAGE=/path/to/img.png \
#       bash scripts/inference/infer_nft.sh

CKPT_DIR="${CKPT_DIR:-ckpts/Wan2.2-TI2V-5B}"
PT_DIR="${PT_DIR:-ckpts/vidar_ckpts/vidar_merged_lora.pt}"

# NFT LoRA checkpoint from training
NFT_LORA_PATH="${NFT_LORA_PATH:-data/outputs/nft_put_object_cabinet/checkpoints/lora_final.pt}"
LORA_ALPHA="${LORA_ALPHA:-1.0}"

# Dataset or single sample
DATASET_JSON="${DATASET_JSON:-}"
PROMPT="${PROMPT:-}"
IMAGE="${IMAGE:-}"

# Generation config
OUTPUT_DIR="${OUTPUT_DIR:-data/outputs/nft_inference}"
SIZE="${SIZE:-640*736}"
FRAME_NUM="${FRAME_NUM:-121}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
SHIFT="${SHIFT:-5.0}"
SEED="${SEED:-42}"
NUM_VIDEOS="${NUM_VIDEOS:-1}"
DEVICE_ID="${DEVICE_ID:-0}"

export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

ARGS=(
    --ckpt_dir "${CKPT_DIR}"
    --pt_dir "${PT_DIR}"
    --nft_lora_path "${NFT_LORA_PATH}"
    --lora_alpha "${LORA_ALPHA}"
    --output_dir "${OUTPUT_DIR}"
    --size "${SIZE}"
    --frame_num "${FRAME_NUM}"
    --sampling_steps "${SAMPLING_STEPS}"
    --guide_scale "${GUIDE_SCALE}"
    --shift "${SHIFT}"
    --seed "${SEED}"
    --num_videos "${NUM_VIDEOS}"
    --device_id 0
)

# Add dataset or single sample args
if [[ -n "${DATASET_JSON}" ]]; then
    ARGS+=(--dataset_json "${DATASET_JSON}")
elif [[ -n "${PROMPT}" && -n "${IMAGE}" ]]; then
    ARGS+=(--prompt "${PROMPT}" --image "${IMAGE}")
else
    # Default: use the training dataset for batch inference
    DATASET_JSON="data/rl_train/robotwin_put_object_cabinet.json"
    ARGS+=(--dataset_json "${DATASET_JSON}")
fi

echo "=============================================="
echo " NFT Inference"
echo "=============================================="
echo " CKPT_DIR:       ${CKPT_DIR}"
echo " PT_DIR:         ${PT_DIR}"
echo " NFT_LORA_PATH:  ${NFT_LORA_PATH}"
echo " LORA_ALPHA:     ${LORA_ALPHA}"
echo " OUTPUT_DIR:     ${OUTPUT_DIR}"
echo " SIZE:           ${SIZE}"
echo " FRAME_NUM:      ${FRAME_NUM}"
echo " SAMPLING_STEPS: ${SAMPLING_STEPS}"
echo " GUIDE_SCALE:    ${GUIDE_SCALE}"
echo " SEED:           ${SEED}"
echo " DEVICE:         cuda:${DEVICE_ID}"
echo "=============================================="

python fastvideo/infer_nft.py "${ARGS[@]}"
