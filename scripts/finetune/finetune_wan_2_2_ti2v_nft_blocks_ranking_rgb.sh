export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

GPU_NUM=${GPU_NUM:-8}
MASTER_PORT=${MASTER_PORT:-19012}

CKPT_DIR="ckpts/Wan2.2-TI2V-5B"
PT_DIR="ckpts/vidar_ckpts/merged_vidar_lora.pt"

# Full dataset: blocks_ranking_rgb (10 scenes)
DATASET_JSON="data/rl_train/robotwin_blocks_ranking_rgb.json"

# Tunable hyperparameters (override via env vars)
NUM_GEN=${NUM_GEN:-16}
SEED=${SEED:-42}
TEMPORAL_LAMBDA=${TEMPORAL_LAMBDA:-0.1}

# Auto-generate OUTPUT_DIR from hyperparams
OUTPUT_DIR=${OUTPUT_DIR:-"data/outputs/nft_blocks_ranking_rgb/ng${NUM_GEN}_s${SEED}_tl${TEMPORAL_LAMBDA}"}

# LoRA config
LORA_RANK=64
LORA_ALPHA=64

source .env

echo ">>> Output dir: ${OUTPUT_DIR}"

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
    --num_generations ${NUM_GEN} \
    --seed ${SEED} \
    --max_samples -1 \
    --reward_backend hallucination \
    --hallucination_prompts "red block" "green block" "blue block" \
    --hallucination_crop_top_ratio 0.6667 \
    --occlusion_gap_max 7 \
    --occlusion_pos_thr 0.15 \
    --duplication_spike_max 1 \
    --convert_model_dtype \
    --offload_model false \
    --max_train_steps 400 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --gpt_api_base ${GPT_API_BASE} \
    --gpt_api_key ${GPT_API_KEY} \
    --max_grad_norm 2.0 \
    --nft_beta 1.0 \
    --kl_beta 0.0001 \
    --adv_clip_max 1.0 \
    --timestep_fraction 0.5 \
    --decay_type 1 \
    --temporal_lambda ${TEMPORAL_LAMBDA} \
    --gradient_accumulation_steps 4 \
    --checkpointing_steps 10 \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA}
    # --resume_from_lora_checkpoint data/outputs/nft_blocks_ranking_rgb/checkpoints/lora_step000200.pt
