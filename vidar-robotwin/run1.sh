#!/bin/bash
module load git
set -euo pipefail

# 单独启动 client（单卡）
# 用法:
#   bash run_client.sh [TASK_NAME] [TASK_CONFIG] [PREFIX] [NUM_NEW_FRAMES] [NUM_SAMPLING_STEP] [CFG] [PORT]
#   不传 TASK_NAME 时按顺序测试 all_tasks 列表
# 示例:
#   bash run_client.sh
#   bash run_client.sh stack_bowls_two hd_clean single_test 60 20 5 25400

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH"
  exit 1
fi
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
set +u
conda activate RoboTwin-hb
set -u

TASK_NAME=${1:-""}
TASK_CONFIG=${2:-"hd_clean"}
PREFIX=${3:-"oneround_test"}
NUM_NEW_FRAMES=${4:-60}
NUM_SAMPLING_STEP=${5:-20}
CFG=${6:-5.0}
PORT=${7:-25400}

OUTPUT_DIR=${OUTPUT_DIR:-"eval_result/ar"}
export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS="ignore::UserWarning"
export PYTHONUNBUFFERED=1

all_tasks=(
  blocks_ranking_size
  handover_block
  handover_mic
  hanging_mug
  scan_object
)

if [[ -n "$TASK_NAME" ]]; then
  tasks=("$TASK_NAME")
else
  tasks=("${all_tasks[@]}")
fi

for task in "${tasks[@]}"; do
  # if [[ -e "$OUTPUT_DIR/$PREFIX/$task" ]]; then
  #   echo "[INFO] Skip (exists): $OUTPUT_DIR/$PREFIX/$task"
  #   continue
  # fi
  python script/eval_policy.py \
    --config policy/AR/deploy_policy.yml \
    --overrides \
    --task_name "$task" \
    --task_config "$TASK_CONFIG" \
    --port "$PORT" \
    --seed 1234 \
    --policy_name AR \
    --num_new_frames "$NUM_NEW_FRAMES" \
    --num_sampling_step "$NUM_SAMPLING_STEP" \
    --guide_scale "$CFG" \
    --rollout_bound 60 \
    --rollout_prefill_num 1 \
    --save_dir "$OUTPUT_DIR/$PREFIX/$task"

  echo "[INFO] Done. Output: $OUTPUT_DIR/$PREFIX/$task"
done
