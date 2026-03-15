#!/bin/bash
#SBATCH --account=p33175
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=gc_idm_train
#SBATCH --output=data/outputs/gc_idm_cabinet/slurm_%j.log
#SBATCH --error=data/outputs/gc_idm_cabinet/slurm_%j.log

set -euo pipefail

cd /gpfs/projects/p33175/EmbodiedVideoRL
mkdir -p data/outputs/gc_idm_cabinet

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

conda run -n wanx --no-capture-output python tools/train_gc_idm.py \
    --data_dir vidar-robotwin/data/put_object_cabinet/demo_clean_ep20_vidar/data \
    --pretrained_idm ckpts/vidar_ckpt/idm.pt \
    --output_dir data/outputs/gc_idm_cabinet \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --max_grad_norm 1.0 \
    --num_workers 4 \
    --save_every 10 \
    --goal_min 1 \
    --goal_max 20 \
    --device 0

echo "Done: $(date)"
