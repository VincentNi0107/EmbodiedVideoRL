export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

GPU_NUM=4
MASTER_PORT=19012

CKPT_DIR="ckpts/Wan2.2-TI2V-5B"
PT_DIR="ckpts/vidar_ckpts/vidar_merged_lora.pt"

# Single-task dataset: put_object_cabinet only (10 scenes)
DATASET_JSON="data/rl_train/robotwin_put_object_cabinet.json"
OUTPUT_DIR="data/outputs/grpo_put_object_cabinet"

# LoRA config (set USE_LORA=true to enable)
USE_LORA=true
LORA_RANK=64
LORA_ALPHA=64
# Resume from an existing LoRA checkpoint; set to empty to train from scratch.
# RESUME_FROM_LORA_CKPT="/home/omz1504/code/DanceGRPO/data/outputs/grpo_put_object_cabinet/checkpoints/lora_step000040.pt"

# Note: this is the global target step after resume.
# Example: resume from lora_step000040.pt and set 200 -> run 41..200.
MAX_TRAIN_STEPS=200

torchrun --nproc_per_node=${GPU_NUM} --master_port ${MASTER_PORT} \
    fastvideo/train_grpo_wan_2_2_ti2v.py \
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
    --eta 1.0 \
    --num_generations 8 \
    --bestofn 8 \
    --seed 42 \
    --max_samples 10 \
    --reward_backend gpt \
    --gpt_model gemini-3-flash-preview \
    --gpt_temperature 0.0 \
    --skip_reward_debug_video true \
    --convert_model_dtype \
    --offload_model false \
    --max_train_steps ${MAX_TRAIN_STEPS} \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --max_grad_norm 2.0 \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --timestep_fraction 0.5 \
    --gradient_accumulation_steps 4 \
    --checkpointing_steps 10 \
    --use_lora ${USE_LORA} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --resume_from_lora_checkpoint "${RESUME_FROM_LORA_CKPT}"
