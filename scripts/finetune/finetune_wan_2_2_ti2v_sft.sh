export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

GPU_NUM=${GPU_NUM:-1}
MASTER_PORT=${MASTER_PORT:-19020}

CKPT_DIR="ckpts/Wan2.2-TI2V-5B"
PT_DIR="ckpts/vidar_ckpts/merged_vidar_lora.pt"

# SFT dataset (JSON with video_path entries)
DATASET_JSON=${DATASET_JSON:-"data/sft_train/robotwin_sft.json"}

OUTPUT_DIR=${OUTPUT_DIR:-"data/outputs/sft_robotwin"}

# LoRA config
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}

echo ">>> Output dir: ${OUTPUT_DIR}"

torchrun --nproc_per_node=${GPU_NUM} --master_port ${MASTER_PORT} \
    fastvideo/train_sft_wan_2_2_ti2v.py \
    --task ti2v-5B \
    --size "640*736" \
    --frame_num 121 \
    --ckpt_dir ${CKPT_DIR} \
    --pt_dir ${PT_DIR} \
    --dataset_json ${DATASET_JSON} \
    --output_dir ${OUTPUT_DIR} \
    --sample_shift 5.0 \
    --convert_model_dtype \
    --offload_model false \
    --num_epochs 100 \
    --max_train_steps 200 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 2.0 \
    --num_train_timesteps 1000 \
    --timestep_fraction 0.05 \
    --use_bsmntw true \
    --gradient_accumulation_steps 4 \
    --checkpointing_steps 10 \
    --gradient_checkpointing true \
    --use_8bit_adam true \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA}
    # Add FFN to LoRA targets for full linear coverage:
    # --lora_target_modules self_attn.q self_attn.k self_attn.v self_attn.o \
    #   cross_attn.q cross_attn.k cross_attn.v cross_attn.o ffn.0 ffn.2
    # Resume from checkpoint:
    # --resume_from_lora_checkpoint data/outputs/sft_robotwin/checkpoints/lora_step000100.pt
