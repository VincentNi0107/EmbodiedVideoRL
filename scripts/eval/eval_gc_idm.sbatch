#!/bin/bash
#SBATCH --account=p33175
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=sxm
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=20
#SBATCH --job-name=gc_idm_eval
#SBATCH --output=data/outputs/gc_idm_cabinet/eval_slurm_%j.log
#SBATCH --error=data/outputs/gc_idm_cabinet/eval_slurm_%j.log

set -euo pipefail

cd /gpfs/projects/p33175/EmbodiedVideoRL

echo "=== Eval Job $SLURM_JOB_ID on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

# Verify checkpoint exists
GC_CKPT="data/outputs/gc_idm_cabinet/checkpoint_best.pt"
if [ ! -f "$GC_CKPT" ]; then
    echo "[ERROR] GC-IDM checkpoint not found: $GC_CKPT"
    exit 1
fi
echo "Using GC-IDM checkpoint: $GC_CKPT"

export GC_IDM_CKPT="$GC_CKPT"
export OFFLOAD_MODEL=true

conda run -n wanx --no-capture-output \
    bash scripts/eval/eval_gc_idm_put_object_cabinet.sh

echo "Eval done: $(date)"
