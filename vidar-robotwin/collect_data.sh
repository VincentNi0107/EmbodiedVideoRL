#!/bin/bash
set -euo pipefail

# Usage:
#   bash collect_data.sh <task_name> <task_config> <gpu_id> [episode_num]
# Example:
#   bash collect_data.sh beat_block_hammer demo_clean 0 20

task_name="${1:?task_name required}"
task_config="${2:?task_config required}"
gpu_id="${3:?gpu_id required}"
episode_num="${4:-20}"

repo_root="$(cd "$(dirname "$0")" && pwd)"
config_dir="${repo_root}/task_config"
src_cfg="${config_dir}/${task_config}.yml"

if [[ ! -f "${src_cfg}" ]]; then
  echo "[ERROR] config not found: ${src_cfg}" >&2
  exit 1
fi

tmp_cfg_name="${task_config}_ep${episode_num}_vidar"
tmp_cfg="${config_dir}/${tmp_cfg_name}.yml"

cp "${src_cfg}" "${tmp_cfg}"
if grep -qE '^[[:space:]]*episode_num:' "${tmp_cfg}"; then
  sed -i -E "s/^[[:space:]]*episode_num:.*/episode_num: ${episode_num}/" "${tmp_cfg}"
else
  echo "episode_num: ${episode_num}" >> "${tmp_cfg}"
fi

# Force vidar embodiment + camera types
if grep -qE '^[[:space:]]*embodiment:' "${tmp_cfg}"; then
  sed -i -E "s/^[[:space:]]*embodiment:.*/embodiment: [aloha-vidar]/" "${tmp_cfg}"
else
  echo "embodiment: [aloha-vidar]" >> "${tmp_cfg}"
fi

if grep -qE '^[[:space:]]*head_camera_type:' "${tmp_cfg}"; then
  sed -i -E "s/^[[:space:]]*head_camera_type:.*/  head_camera_type: Large_D435/" "${tmp_cfg}"
else
  echo "  head_camera_type: Large_D435" >> "${tmp_cfg}"
fi

if grep -qE '^[[:space:]]*wrist_camera_type:' "${tmp_cfg}"; then
  sed -i -E "s/^[[:space:]]*wrist_camera_type:.*/  wrist_camera_type: Large_D435/" "${tmp_cfg}"
else
  echo "  wrist_camera_type: Large_D435" >> "${tmp_cfg}"
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py "${task_name}" "${tmp_cfg_name}"

# Clean temp cache
rm -rf "data/${task_name}/${tmp_cfg_name}/.cache"
