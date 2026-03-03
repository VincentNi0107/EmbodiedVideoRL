export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

GPU_NUM=${GPU_NUM:-4}
MASTER_PORT=${MASTER_PORT:-19015}

CKPT_DIR="ckpts/Wan2.2-TI2V-5B"
PT_DIR="ckpts/vidar_ckpt/merged_vidar_lora.pt"

# Full dataset: put_bottles_dustbin (10 scenes)
DATASET_JSON="data/rl_train/robotwin_put_bottles_dustbin.json"
OUTPUT_DIR="data/outputs/nft_put_bottles_dustbin"

# LoRA config
LORA_RANK=64
LORA_ALPHA=64

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
    --num_generations 8 \
    --seed 42 \
    --max_samples -1 \
    --reward_backend hallucination_bottles \
    --hallucination_crop_top_ratio 0.6667 \
    --bottle_hall_prompt "bottle" \
    --bottle_cx_cutoff 0.26 \
    --bottle_spike_max 3 \
    --bottle_filter_max_gap 5 \
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
    --temporal_lambda 0.0 \
    --gradient_accumulation_steps 4 \
    --checkpointing_steps 10 \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} 

    # -resume_from_lora_checkpoint data/outputs/nft_put_bottles_dustbin/checkpoints/lora_step000060.pt
