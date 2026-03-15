#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Evaluate closed-loop Goal-Conditioned IDM on put_object_cabinet
# ──────────────────────────────────────────────────────────────────────────────
#
# Usage (on a GPU node):
#   cd /gpfs/projects/p33175/EmbodiedVideoRL
#   bash scripts/eval/eval_gc_idm_put_object_cabinet.sh
#
# Env overrides:
#   GC_IDM_CKPT=data/outputs/gc_idm_cabinet/checkpoint_best.pt
#   PORT=25400  CUDA_DEV=0  PREFIX=gc_idm_test

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# ── Configuration ────────────────────────────────────────────────────────────
PORT=${PORT:-25400}
CUDA_DEV=${CUDA_DEV:-0}
PREFIX=${PREFIX:-"gc_idm_test"}
TASK_NAME=${TASK_NAME:-"put_object_cabinet"}
TASK_CONFIG=${TASK_CONFIG:-"hd_clean"}
NUM_NEW_FRAMES=${NUM_NEW_FRAMES:-120}
NUM_SAMPLING_STEP=${NUM_SAMPLING_STEP:-20}
GUIDE_SCALE=${GUIDE_SCALE:-5.0}

# Model paths
export CKPT_DIR="${CKPT_DIR:-ckpts/Wan2.2-TI2V-5B}"
export PT_DIR="${PT_DIR:-ckpts/vidar_ckpt/merged_vidar_lora.pt}"
export IDM_PATH="${IDM_PATH:-ckpts/vidar_ckpt/idm.pt}"
export NFT_LORA_PATH="${NFT_LORA_PATH:-}"
export LORA_ALPHA="${LORA_ALPHA:-1.0}"

# Goal-conditioned IDM checkpoint
export GC_IDM_PATH="${GC_IDM_CKPT:-data/outputs/gc_idm_cabinet/checkpoint_best.pt}"

export DEVICE_ID=0
export CUDA_VISIBLE_DEVICES="$CUDA_DEV"
export FRAME_NUM=121
export SIZE="${SIZE:-640*736}"
export T5_CPU="${T5_CPU:-false}"
export OFFLOAD_MODEL="${OFFLOAD_MODEL:-true}"
export IDM_CPU="${IDM_CPU:-false}"
export SAVE_VIDEO="${SAVE_VIDEO:-false}"

EVAL_DIR="vidar-robotwin/eval_result/ar/${PREFIX}/${TASK_NAME}"

echo "=============================================="
echo " GC-IDM Closed-Loop Eval: ${TASK_NAME}"
echo "=============================================="
echo " PORT:          ${PORT}"
echo " CUDA_DEV:      ${CUDA_DEV}"
echo " PT_DIR:        ${PT_DIR}"
echo " IDM_PATH:      ${IDM_PATH}"
echo " GC_IDM_PATH:   ${GC_IDM_PATH}"
echo " NFT_LORA_PATH: ${NFT_LORA_PATH:-<none>}"
echo " Config:        ${NUM_NEW_FRAMES} frames, ${NUM_SAMPLING_STEP} steps, CFG ${GUIDE_SCALE}"
echo " Output:        ${EVAL_DIR}"
echo "=============================================="

# ── Verify GC-IDM checkpoint exists ──────────────────────────────────────────
if [ ! -f "${GC_IDM_PATH}" ]; then
    echo "[ERROR] GC-IDM checkpoint not found: ${GC_IDM_PATH}"
    exit 1
fi

# ── Check port ───────────────────────────────────────────────────────────────
python3 -c "
import socket, sys
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.settimeout(1)
    if s.connect_ex(('127.0.0.1', ${PORT})) == 0:
        print(f'[ERROR] Port ${PORT} already in use.', file=sys.stderr)
        sys.exit(1)
"

# ── Start server (background) ───────────────────────────────────────────────
echo "[INFO] Starting server with GC-IDM on port ${PORT}..."
export PORT=${PORT}
python -m uvicorn fastvideo.server_nft:api --host 0.0.0.0 --port ${PORT} --workers 1 &
SERVER_PID=$!

cleanup() {
    echo "[INFO] Stopping server (PID=$SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Wait for server ──────────────────────────────────────────────────────────
echo "[INFO] Waiting for server to be ready..."
TIMEOUT=600
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[ERROR] Server process died."
        exit 1
    fi
    if python3 -c "
import socket, sys
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.settimeout(1)
    sys.exit(0 if s.connect_ex(('127.0.0.1', ${PORT})) == 0 else 1)
" 2>/dev/null; then
        echo "[INFO] Server is ready."
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "[ERROR] Server timed out after ${TIMEOUT}s."
    exit 1
fi

# ── Run client evaluation (closed-loop mode) ─────────────────────────────────
echo "[INFO] Running closed-loop evaluation: ${TASK_NAME} (${TASK_CONFIG})..."
cd vidar-robotwin

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
    --guide_scale "$GUIDE_SCALE" \
    --rollout_bound 120 \
    --rollout_prefill_num 1 \
    --closed_loop true \
    --save_dir "eval_result/ar/${PREFIX}/${TASK_NAME}"

cd "$ROOT_DIR"

# ── Results ──────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo " Results: ${TASK_NAME} (closed-loop GC-IDM)"
echo "=============================================="
if [ -f "${EVAL_DIR}/_result.txt" ]; then
    cat "${EVAL_DIR}/_result.txt"
else
    echo "[WARN] No _result.txt found."
fi
echo ""
echo "=============================================="
