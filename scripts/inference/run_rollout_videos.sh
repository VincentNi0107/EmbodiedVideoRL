#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU parallel video rollout launcher
#
# Reads robotwin_121.json dataset, generates 8 rollout videos per sample
# using Wan2.2 TI2V model with SDE sampling across multiple GPUs.
#
# Each GPU loads the full model independently (no FSDP).
# Data items are split across GPUs in round-robin.
# ──────────────────────────────────────────────────────────────────────────────

# ── GPU config ────────────────────────────────────────────────────────────────
GPU_NUM=4                       # Number of GPUs to use
MASTER_PORT=19015               # Avoid port conflict with other jobs

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_DIR="ckpts/Wan2.2-TI2V-5B"

# Fine-tuned weights (set to "" or remove --pt_dir line to use base model)
PT_DIR="ckpts/vidar_ckpts/vidar_merged_lora.pt"

# Dataset
DATASET_JSON="data/rl_train/robotwin_put_object_cabinet.json"

# Output directory
OUTPUT_DIR="data/outputs/rollout_robotwin_121"

# ── Sampling config ───────────────────────────────────────────────────────────
NUM_ROLLOUTS=8                  # Number of rollout videos per data item
SAMPLE_STEPS=20                 # Denoising steps
SAMPLE_SHIFT=5.0                # SD3 time-shift
GUIDE_SCALE=5.0                 # Classifier-free guidance scale
ETA=1.0                         # SDE noise (0=ODE deterministic, >0=SDE diverse)
FRAME_NUM=121                   # Number of video frames
SEED=42                         # Base seed (each rollout uses seed + rollout_index)

# ── LoRA config (optional, set USE_LORA=false to disable) ────────────────────
USE_LORA=false
LORA_RANK=64
LORA_ALPHA=64
LORA_CHECKPOINT=""              # Path to LoRA .pt file

# ── Launch ────────────────────────────────────────────────────────────────────
echo "=========================================="
echo "  Multi-GPU Video Rollout"
echo "  GPUs: ${GPU_NUM}"
echo "  Dataset: ${DATASET_JSON}"
echo "  Rollouts per sample: ${NUM_ROLLOUTS}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

CMD="torchrun --nproc_per_node=${GPU_NUM} --master_port ${MASTER_PORT} \
    scripts/inference/rollout_videos.py \
    --task ti2v-5B \
    --size '640*736' \
    --frame_num ${FRAME_NUM} \
    --ckpt_dir ${CKPT_DIR} \
    --dataset_json ${DATASET_JSON} \
    --output_dir ${OUTPUT_DIR} \
    --sample_steps ${SAMPLE_STEPS} \
    --sample_shift ${SAMPLE_SHIFT} \
    --sample_guide_scale ${GUIDE_SCALE} \
    --eta ${ETA} \
    --num_rollouts ${NUM_ROLLOUTS} \
    --seed ${SEED} \
    --convert_model_dtype \
    --skip_existing"

# Add --pt_dir if specified
if [ -n "${PT_DIR}" ]; then
    CMD="${CMD} --pt_dir ${PT_DIR}"
fi

# Add LoRA args if enabled
if [ "${USE_LORA}" = "true" ] && [ -n "${LORA_CHECKPOINT}" ]; then
    CMD="${CMD} \
    --use_lora true \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_checkpoint ${LORA_CHECKPOINT}"
fi

eval ${CMD}
