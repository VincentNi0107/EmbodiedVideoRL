#!/bin/bash
conda activate RoboTwin-hb


# --- 配置区域 ---
# 任务配置
TASK_CONFIG=${1:-"hd_clean"}
MODEL=${2:-"../vidar_ckpts/vidar.pt"}
IDM=${3:-"../vidar_ckpts/idm.pt"}
PREFIX=${4:-"debug_ddp"}

# 采样参数
NUM_NEW_FRAMES=${5:-60}
NUM_SAMPLING_STEP=${6:-20}
CFG=${7:-5.0}

# Server 脚本位置 (根据需要修改，支持 T2V 或 I2V)
SERVER_SCRIPT="../server/stand_worker.sh"

# --- 启动 ---
echo "Starting Unified DDP Evaluation..."
echo "Model: $MODEL"
echo "Prefix: $PREFIX"
echo "Server: $SERVER_SCRIPT"

# 设置 Master Port 防止冲突
export MASTER_PORT=11452
# 使用 torchrun 启动
# nproc_per_node 自动设为 GPU 数量 (或者手动指定 8)
GPU_COUNT=$(nvidia-smi -L | wc -l)

torchrun --nproc_per_node=8 --master_port=$MASTER_PORT \
    policy/AR/run_eval_ddp.py \
    --server_script "$SERVER_SCRIPT" \
    --model "$MODEL" \
    --idm "$IDM" \
    --prefix "$PREFIX" \
    --task_config "$TASK_CONFIG" \
    --rollout_prefill_num 1 \
    --num_new_frames "$NUM_NEW_FRAMES" \
    --num_sampling_step "$NUM_SAMPLING_STEP" \
    --cfg "$CFG"

echo "Evaluation finished."

