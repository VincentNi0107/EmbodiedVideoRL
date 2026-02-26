#!/bin/bash
# Batch optical-flow anomaly detection for all mp4s under rollout_robotwin_121.
# Usage: bash run_flow_anomaly_batch.sh [JOBS]
# JOBS: parallel workers (default 4)

set -euo pipefail

INPUT_ROOT="/gpfs/projects/p33048/DanceGRPO/data/outputs/rollout_robotwin_121"
OUTPUT_ROOT="/gpfs/projects/p33048/DanceGRPO/data/outputs/rollout_robotwin_121_flow"
SCRIPT="/gpfs/projects/p33048/DanceGRPO/tools/detect_flow_anomalies.py"
JOBS="${1:-4}"

echo "Input:   $INPUT_ROOT"
echo "Output:  $OUTPUT_ROOT"
echo "Workers: $JOBS"

# Collect all mp4 paths
VIDEOS=$(find "$INPUT_ROOT" -name "*.mp4" | sort)
TOTAL=$(echo "$VIDEOS" | wc -l)
echo "Found $TOTAL videos"
echo "----------------------------------------"

process_one() {
    local mp4="$1"
    # Mirror the subdirectory structure under OUTPUT_ROOT
    local rel_dir
    rel_dir=$(dirname "${mp4#$INPUT_ROOT/}")
    local out_dir="$OUTPUT_ROOT/$rel_dir/$(basename "$mp4" .mp4)"

    # Skip if already done (anomaly video exists)
    local stem
    stem=$(basename "$mp4" .mp4)
    if [[ -f "$out_dir/${stem}_anomaly.mp4" ]]; then
        echo "[SKIP] $stem"
        return
    fi

    echo "[RUN ] $stem"
    conda run -n wanx python "$SCRIPT" \
        --input "$mp4" \
        --out-dir "$out_dir" \
        2>&1 | tail -1   # only print the last summary line
}

export -f process_one
export INPUT_ROOT OUTPUT_ROOT SCRIPT

echo "$VIDEOS" | xargs -P "$JOBS" -I{} bash -c 'process_one "$@"' _ {}

echo "========================================"
echo "All done. Results under: $OUTPUT_ROOT"
