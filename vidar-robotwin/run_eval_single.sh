#!/bin/bash
set -euo pipefail

# 简化版单任务评测脚本（单卡）
# 用法:
#   bash run_eval_single.sh [TASK_NAME] [TASK_CONFIG] [MODEL] [IDM] [PREFIX] [NUM_NEW_FRAMES] [NUM_SAMPLING_STEP] [CFG] [PORT]
# 示例:
#   bash run_eval_single.sh stack_bowls_two hd_clean ../vidar_ckpts/vidar.pt ../vidar_ckpts/idm.pt single_test 60 20 5 25400

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH"
  exit 1
fi
# Ensure conda activate works in non-interactive shells
CONDA_BASE="${ROBOTWIN_CONDA_BASE:-$(conda info --base)}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
# Some conda activate scripts use unbound vars; relax nounset during activation.
set +u
conda activate RoboTwin-hb
set -u

TASK_NAME=${1:-"stack_bowls_two"}
TASK_CONFIG=${2:-"hd_clean"}
MODEL="/home/omz1504/code/vidar/vidar_ckpts/vidar.pt"
IDM="/home/omz1504/code/vidar/vidar_ckpts/idm.pt"
PREFIX=${5:-"single_test"}
NUM_NEW_FRAMES=${6:-60}
NUM_SAMPLING_STEP=${7:-20}
CFG=${8:-5.0}
PORT=${9:-25400}
export PORT

SERVER_SCRIPT=${SERVER_SCRIPT:-"../vidar/server/stand_worker.sh"}
SERVER_CWD=${SERVER_CWD:-"../vidar"}
OUTPUT_DIR=${OUTPUT_DIR:-"eval_result/ar"}
CUDA_DEV=${CUDA_DEV:-0}

SERVER_LOG=${SERVER_LOG:-"/tmp/vidar_server_${PORT}.log"}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill -9 "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# 检查端口是否已被占用
if python - <<'PY'
import socket, os
port = int(os.environ.get("PORT", "25400"))
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.settimeout(1)
    if s.connect_ex(("127.0.0.1", port)) == 0:
        raise SystemExit(1)
PY
then
  :
else
  echo "[ERROR] Port $PORT already in use. Please stop the old server or choose another port."
  exit 1
fi

# 启动 server（使用 vidar 环境）
# 若你的 vidar 环境名不是默认的 vidar，可在调用前设置：
#   export VIDAR_CONDA_ENV=wanx
touch "$SERVER_LOG"
(
  cd "$SERVER_CWD"
  echo "[INFO] Launching server: bash $SERVER_SCRIPT $MODEL $IDM $PORT $CUDA_DEV" >>"$SERVER_LOG"
  bash "$SERVER_SCRIPT" "$MODEL" "$IDM" "$PORT" "$CUDA_DEV" >>"$SERVER_LOG" 2>&1
) &
SERVER_PID=$!

echo "[INFO] Server PID: $SERVER_PID, log: $SERVER_LOG"

# 等待端口就绪
if ! python - <<'PY'
import socket, time, os
port = int(os.environ.get("PORT", "25400"))
start = time.time()
while time.time() - start < 1200:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            print("[INFO] Server is ready on", port)
            raise SystemExit(0)
    time.sleep(1)
print("[ERROR] Server not ready within timeout")
raise SystemExit(1)
PY
then
  echo "[ERROR] Server failed to start. Last 200 lines of log:"
  tail -n 200 "$SERVER_LOG" || true
  exit 1
fi

# 单任务评测
export CUDA_VISIBLE_DEVICES="$CUDA_DEV"
export PYTHONWARNINGS="ignore::UserWarning"
export PYTHONUNBUFFERED=1

python script/eval_policy.py \
  --config policy/AR/deploy_policy.yml \
  --overrides \
  --task_name "$TASK_NAME" \
  --task_config "$TASK_CONFIG" \
  --port "$PORT" \
  --seed 1234 \
  --policy_name AR \
  --num_new_frames "$NUM_NEW_FRAMES" \
  --num_sampling_step "$NUM_SAMPLING_STEP" \
  --guide_scale "$CFG" \
  --rollout_bound 60 \
  --rollout_prefill_num 1 \
  --save_dir "$OUTPUT_DIR/$PREFIX/$TASK_NAME"

echo "[INFO] Done. Output: $OUTPUT_DIR/$PREFIX/$TASK_NAME"
