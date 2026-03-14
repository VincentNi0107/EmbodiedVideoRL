#!/bin/bash
set -euo pipefail

# Collect all 50 tasks, 20 episodes each, using vidar-robotwin/collect_data.sh
# Usage:
#   bash collect_all_tasks.sh <task_config> <gpu_id> [episode_num]
# Example:
#   bash collect_all_tasks.sh demo_clean 0 20

task_config="${1:?task_config required}"
gpu_id="${2:?gpu_id required}"
episode_num="${3:-20}"

tasks=(
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
  place_object_basket
  place_object_scale
  place_object_stand
  place_phone_stand
  place_shoe
  press_stapler
  put_bottles_dustbin
  put_object_cabinet
  rotate_qrcode
  scan_object
  shake_bottle_horizontally
  shake_bottle
  stack_blocks_three
  stack_blocks_two
  stack_bowls_three
  stack_bowls_two
  stamp_seal
  turn_switch
)

for task in "${tasks[@]}"; do
  echo "===== Collecting: ${task} (${episode_num} eps) ====="
  bash collect_data.sh "${task}" "${task_config}" "${gpu_id}" "${episode_num}"
done
