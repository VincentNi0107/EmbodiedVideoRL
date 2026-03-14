#!/bin/bash

# 单独启动 client（单卡）
# 用法:
#   bash run_client.sh [TASK_NAME] [TASK_CONFIG] [PREFIX] [NUM_NEW_FRAMES] [NUM_SAMPLING_STEP] [CFG] [PORT]
#   不传 TASK_NAME 时按顺序测试 all_tasks 列表
# 示例:
#   bash run_client.sh
#   bash run_client_cl.sh stack_bowls_two hd_clean causal_test 16 16 3 25400

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
PREFIX=${3:-"causal_test"}
NUM_NEW_FRAMES=${4:-16}
NUM_SAMPLING_STEP=${5:-16}
CFG=${6:-3.0}
PORT=${7:-25400}



OUTPUT_DIR=${OUTPUT_DIR:-"eval_result/ar"}
CUDA_DEV=${CUDA_DEV:-0}

export CUDA_VISIBLE_DEVICES="$CUDA_DEV"
export PYTHONWARNINGS="ignore::UserWarning"
export PYTHONUNBUFFERED=1

all_tasks=(
  stack_bowls_two
  place_cans_plasticbox
  beat_block_hammer
  pick_dual_bottles
  click_alarmclock
  click_bell
  shake_bottle_horizontally
  open_laptop
  turn_switch
  press_stapler
  shake_bottle
  place_bread_basket
  grab_roller
  place_burger_fries
  place_phone_stand
  place_object_stand
  place_container_plate
  place_a2b_right
  place_empty_cup
  adjust_bottle
  dump_bin_bigbin
)

if [[ -n "$TASK_NAME" ]]; then
  tasks=("$TASK_NAME")
else
  tasks=("${all_tasks[@]}")
fi

for task in "${tasks[@]}"; do
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
