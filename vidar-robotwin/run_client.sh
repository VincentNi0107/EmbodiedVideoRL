#!/bin/bash
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
PREFIX=${3:-"nft_test"}
NUM_NEW_FRAMES=${4:-120}
NUM_SAMPLING_STEP=${5:-20}
CFG=${6:-5.0}
PORT=${7:-25400}

OUTPUT_DIR=${OUTPUT_DIR:-"eval_result/ar"}
CUDA_DEV=${CUDA_DEV:-0}

export CUDA_VISIBLE_DEVICES="$CUDA_DEV"
export PYTHONWARNINGS="ignore::UserWarning"
export PYTHONUNBUFFERED=1

all_tasks=(
  # adjust_bottle
  # beat_block_hammer
  # blocks_ranking_rgb
  # blocks_ranking_size
  # click_alarmclock
  # click_bell
  # dump_bin_bigbin
  # grab_roller
  # handover_block
  # handover_mic
  # hanging_mug
  # lift_pot
  # move_can_pot
  # move_pillbottle_pad
  # move_playingcard_away
  # move_stapler_pad
  # open_laptop
  # open_microwave
  # pick_diverse_bottles
  # pick_dual_bottles
  # place_a2b_left
  # place_a2b_right
  # place_bread_basket
  # place_bread_skillet
  # place_burger_fries
  # place_can_basket
  # place_cans_plasticbox
  # place_container_plate
  # place_dual_shoes
  # place_empty_cup
  # place_fan
  # place_mouse_pad
  # place_object_basket
  # place_object_scale
  # place_object_stand
  # place_phone_stand
  # place_shoe
  # press_stapler
  # put_bottles_dustbin
  put_object_cabinet
  # rotate_qrcode
  # scan_object
  # shake_bottle_horizontally
  # shake_bottle
  # stack_blocks_three
  # stack_blocks_two
  # stack_bowls_three
  # stack_bowls_two
  # stamp_seal
  # turn_switch
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
    --rollout_bound 120 \
    --rollout_prefill_num 1 \
    --save_dir "$OUTPUT_DIR/$PREFIX/$task"

  echo "[INFO] Done. Output: $OUTPUT_DIR/$PREFIX/$task"
done
