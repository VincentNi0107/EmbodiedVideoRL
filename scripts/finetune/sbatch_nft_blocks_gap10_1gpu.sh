#!/bin/bash
#SBATCH --account=p33048
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=sxm
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=20
#SBATCH --job-name=nft_blocks_gap10
#SBATCH --output=data/outputs/nft_blocks_ranking_rgb_gap10/slurm_%j.out
#SBATCH --error=data/outputs/nft_blocks_ranking_rgb_gap10/slurm_%j.err

# ── NFT training: blocks_ranking_rgb with occlusion_gap_max=10 ──
# Experiment: increased occlusion suppression window from 5→10 frames
# to avoid false-positive hallucination from robot arm occlusion (~9 frames).

set -euo pipefail

mkdir -p data/outputs/nft_blocks_ranking_rgb_gap10

export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

GPU_NUM=1
MASTER_PORT=19014

CKPT_DIR="ckpts/Wan2.2-TI2V-5B"
PT_DIR="ckpts/vidar_ckpts/vidar_merged_lora.pt"

DATASET_JSON="data/rl_train/robotwin_blocks_ranking_rgb.json"
OUTPUT_DIR="data/outputs/nft_blocks_ranking_rgb_gap10"

LORA_RANK=64
LORA_ALPHA=64

conda run -n wanx --no-capture-output --cwd /gpfs/projects/p33048/DanceGRPO \
    torchrun --nproc_per_node=${GPU_NUM} --master_port ${MASTER_PORT} \
    fastvideo/train_nft_wan_2_2_ti2v.py \
    --task ti2v-5B \
    --size "640*736" \
    --frame_num 121 \
    --ckpt_dir ${CKPT_DIR} \
    --pt_dir ${PT_DIR} \
    --dataset_json ${DATASET_JSON} \
    --output_dir ${OUTPUT_DIR} \
    --sample_steps 20 \
    --sample_shift 5.0 \
    --sample_guide_scale 5.0 \
    --num_generations 8 \
    --seed 42 \
    --max_samples -1 \
    --reward_backend hallucination \
    --hallucination_prompts "red block" "green block" "blue block" \
    --hallucination_crop_top_ratio 0.6667 \
    --occlusion_gap_max 10 \
    --occlusion_pos_thr 0.15 \
    --convert_model_dtype \
    --offload_model true \
    --max_train_steps 200 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --max_grad_norm 2.0 \
    --nft_beta 1.0 \
    --kl_beta 0.0001 \
    --adv_clip_max 1.0 \
    --timestep_fraction 0.5 \
    --decay_type 1 \
    --gradient_accumulation_steps 4 \
    --checkpointing_steps 10 \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA}
