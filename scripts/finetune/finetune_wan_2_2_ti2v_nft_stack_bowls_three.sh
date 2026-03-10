export WANDB_DISABLED=false
export WANDB_PROJECT="DanceGRPO"
export WANDB_MODE=online

GPU_NUM=${GPU_NUM:-8}
MASTER_PORT=${MASTER_PORT:-19016}

CKPT_DIR="ckpts/Wan2.2-TI2V-5B"
PT_DIR="ckpts/vidar_ckpts/merged_vidar_lora.pt"

# Full dataset: stack_bowls_three (10 scenes)
DATASET_JSON="data/rl_train/robotwin_stack_bowls_three.json"

# Tunable hyperparameters (override via env vars)
NUM_GEN=${NUM_GEN:-16}
SEED=${SEED:-42}
TEMPORAL_LAMBDA=${TEMPORAL_LAMBDA:-0.0}
KL_BETA=${KL_BETA:-0.001}

# Auto-generate OUTPUT_DIR from hyperparams
OUTPUT_DIR=${OUTPUT_DIR:-"data/outputs/nft_stack_bowls_three/ng${NUM_GEN}_s${SEED}_tl${TEMPORAL_LAMBDA}_kl${KL_BETA}"}

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
    --sample_steps 50 \
    --sample_shift 5.0 \
    --sample_guide_scale 5.0 \
    --num_generations ${NUM_GEN} \
    --seed ${SEED} \
    --max_samples -1 \
    --reward_backend hallucination_bowls \
    --hallucination_crop_top_ratio 0.6667 \
    --bowl_stack_prompt "bowl" \
    --bowl_initial_count 3 \
    --bowl_convergence_thr 0.30 \
    --bowl_check_window_frac 0.20 \
    --bowl_gap_max 5 \
    --bowl_reappear_max 10 \
    --bowl_reappear_pos_thr 0.15 \
    --convert_model_dtype \
    --offload_model false \
    --max_train_steps 400 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --max_grad_norm 2.0 \
    --nft_beta 1.0 \
    --kl_beta ${KL_BETA} \
    --adv_clip_max 1.0 \
    --timestep_fraction 0.5 \
    --decay_type 1 \
    --temporal_lambda ${TEMPORAL_LAMBDA} \
    --gradient_accumulation_steps 4 \
    --checkpointing_steps 10 \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA}
    # --resume_from_lora_checkpoint data/outputs/nft_stack_bowls_three/checkpoints/lora_step000060.pt
