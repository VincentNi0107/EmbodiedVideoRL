#!/bin/bash
# Rollout-only: generate 8 videos per sample for put_bottles_dustbin
# ODE sampling with 50 steps (instead of default 20)
# No training, no reward scoring — just video generation.
#
# Usage (on qgpu2002):
#   ssh qgpu2002 "conda run -n wanx --no-capture-output --cwd /gpfs/projects/p33048/DanceGRPO \
#       bash scripts/finetune/rollout_bottles_dustbin.sh"
#
# Background:
#   ssh qgpu2002 "nohup conda run -n wanx --no-capture-output --cwd /gpfs/projects/p33048/DanceGRPO \
#       bash scripts/finetune/rollout_bottles_dustbin.sh \
#       > data/outputs/rollout_bottles_dustbin_50steps.log 2>&1 &"

CKPT_DIR="ckpts/Wan2.2-TI2V-5B"
PT_DIR="ckpts/vidar_ckpts/vidar_merged_lora.pt"

DATASET_JSON="data/rl_train/robotwin_put_bottles_dustbin.json"
OUTPUT_DIR="data/outputs/rollout_bottles_dustbin_50steps"

python rollout_bottles.py \
    --task ti2v-5B \
    --size "640*736" \
    --frame_num 121 \
    --ckpt_dir ${CKPT_DIR} \
    --pt_dir ${PT_DIR} \
    --dataset_json ${DATASET_JSON} \
    --output_dir ${OUTPUT_DIR} \
    --sample_steps 50 \
    --sample_shift 5.0 \
    --sample_guide_scale 5.0 \
    --num_generations 8 \
    --seed 42 \
    --convert_model_dtype
