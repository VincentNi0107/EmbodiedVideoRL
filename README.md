# EmbodiedVideoRL

Reinforcement learning for embodied video generation — fine-tuning [Wan-2.2 TI2V (5B)](https://github.com/Wan-Video/Wan2.2) with reward feedback on robotic manipulation tasks.

Built on [DanceGRPO](https://arxiv.org/abs/2505.07818) and the [Vidar/RoboTwin](https://robotwin-benchmark.github.io/early-version/) ecosystem.

## Overview

This repo applies **GRPO** (Group Relative Policy Optimization) and **DiffusionNFT** (contrastive RL) to fine-tune a pre-trained text-image-to-video model so that generated robot manipulation videos better match task-completion criteria. Reward signals come from vision-language models (Gemini API) or SAM3-based object tracking.

### Supported Tasks

Six manipulation tasks from the [RoboTwin](https://robotwin-benchmark.github.io/early-version/) benchmark, each with 10 scenes (reference image + text prompt):

| Task | Description | Reward Backend | Status |
|------|-------------|----------------|--------|
| **put_object_cabinet** | Open drawer with one arm, place object with the other | Gemini API | Training converged |
| **blocks_ranking_rgb** | Arrange R/G/B blocks left-to-right | SAM3 hallucination (constant count) | Training in progress |
| **put_bottles_dustbin** | Place 3 bottles into dustbin | SAM3 trajectory (monotonic decrease) | Training in progress |
| **blocks_ranking_size** | Arrange 3 blocks by size | TBD | Not started |
| **stack_blocks_three** | Stack R/G/B blocks | TBD | Not started |
| **stack_bowls_three** | Stack 3 bowls | TBD | Not started |

### Training Algorithms

| Algorithm | Script | Sampling | Key Idea |
|-----------|--------|----------|----------|
| **GRPO** | `train_grpo_wan_2_2_ti2v.py` | SDE (noise injection) | PPO-style clipped policy gradient with log-prob |
| **DiffusionNFT** | `train_nft_wan_2_2_ti2v.py` | ODE (deterministic) | Contrastive loss pushing toward high-reward trajectories |

### Reward Backends

| Backend | Flag | Method |
|---------|------|--------|
| Gemini API | `--reward_backend gpt` | Sends 4 sampled frames to Gemini, asks pass/fail |
| SAM3 Hallucination | `--reward_backend hallucination` | Tracks objects across frames, checks constant count |
| SAM3 Bottle Trajectory | `--reward_backend hallucination_bottles` | Trajectory-based tracking with monotonic decrease check |

## Getting Started

### Prerequisites

- Python 3.10, PyTorch 2.6+, CUDA
- Conda environment setup:
```bash
conda create -n wanx python=3.10
conda activate wanx
pip install -r requirements.txt  # or ./env_setup.sh fastvideo
```

### Model Checkpoints

Download and place under `ckpts/`:

| Checkpoint | Path | Source |
|-----------|------|--------|
| Wan2.2-TI2V-5B (base model) | `ckpts/Wan2.2-TI2V-5B/` | [Wan-AI](https://huggingface.co/Wan-AI) |
| Vidar merged LoRA | `ckpts/vidar_ckpts/merged_vidar_lora.pt` | [HuggingFace](https://huggingface.co/VincentNi/EmbodiedVideoRL) |
| IDM weights (optional, for server) | `ckpts/vidar_ckpts/idm.pt` | [HuggingFace](https://huggingface.co/VincentNi/EmbodiedVideoRL) |
| SAM3 | `sam3/` | [SAM3 repo](https://github.com/anthropics/segment-anything-3) |

### API Keys

The Gemini reward backend requires an API key. **Never hardcode keys in source files.**

```bash
cp .env.example .env
# Edit .env and fill in your GPT_API_KEY
source .env
```

## Training

### DiffusionNFT + Gemini reward (put_object_cabinet)
```bash
# 4x A100-80GB
bash scripts/finetune/finetune_wan_2_2_ti2v_nft_put_object_cabinet.sh
```

### DiffusionNFT + SAM3 hallucination reward (blocks_ranking_rgb)
```bash
# Single GPU (default)
bash scripts/finetune/finetune_wan_2_2_ti2v_nft_blocks_ranking_rgb.sh

# Multi-GPU (4x A100-80GB)
GPU_NUM=4 bash scripts/finetune/finetune_wan_2_2_ti2v_nft_blocks_ranking_rgb.sh
```

### DiffusionNFT + SAM3 bottle trajectory reward (put_bottles_dustbin)
```bash
# Single GPU (default)
bash scripts/finetune/finetune_wan_2_2_ti2v_nft_put_bottles_dustbin.sh

# Multi-GPU (4x A100-80GB)
GPU_NUM=4 bash scripts/finetune/finetune_wan_2_2_ti2v_nft_put_bottles_dustbin.sh
```

### GRPO + Gemini reward (put_object_cabinet)
```bash
bash scripts/finetune/finetune_wan_2_2_ti2v_grpo.sh
```

### Key Training Arguments

| Arg | Description |
|-----|-------------|
| `--ckpt_dir` | Wan2.2-TI2V-5B base model directory |
| `--pt_dir` | Pre-merged LoRA `.pt` weights |
| `--dataset_json` | JSON with `{prompt, image}` pairs |
| `--num_generations 8` | Videos per prompt per step |
| `--reward_backend` | `gpt`, `hallucination`, `hallucination_bottles`, or `none` |
| `--use_lora true --lora_rank 64` | LoRA training config |
| `--resume_from_lora_checkpoint` | Resume from a saved `.pt` |
| `--offload_model true` | Offload VAE/T5 to CPU during training |

## Inference

```bash
# CLI batch inference
bash scripts/inference/infer_nft.sh

# Override LoRA checkpoint
NFT_LORA_PATH=data/outputs/nft_put_object_cabinet/checkpoints/lora_step000100.pt \
    bash scripts/inference/infer_nft.sh

# Single prompt + image
PROMPT="A robot opens the cabinet." IMAGE=/path/to/img.png \
    bash scripts/inference/infer_nft.sh

# API server (vidar-compatible)
bash scripts/inference/start_server_nft.sh [PORT] [CUDA_DEV] [NFT_LORA_PATH] [LORA_ALPHA]
```

## Standalone Analysis Tools

```bash
# Blocks hallucination detection (constant object count)
python tools/detect_hallucination.py --input /path/to/video.mp4 --out-dir /path/to/out

# Bottle trajectory analysis (monotonic decrease)
python tools/run_bottle_hallucination_pipeline.py \
    --input-root data/outputs/rollout_robotwin_121 \
    --out-root data/outputs/bottle_hall_v2

# Optical flow anomaly detection
python tools/detect_flow_anomalies.py --input /path/to/video.mp4 --out-dir /path/to/out

# Test bottle algorithm against ground truth (no GPU needed)
python tests/test_bottle_hallucination.py
```

## Project Structure

```
fastvideo/
  train_grpo_wan_2_2_ti2v.py        # GRPO training
  train_nft_wan_2_2_ti2v.py         # DiffusionNFT training
  infer_nft.py                      # CLI inference
  server_nft.py                     # FastAPI server
  reward/
    gpt.py                          # Gemini API reward scorer
    hallucination.py                # SAM3 constant-count reward (blocks)
    hallucination_bottles.py        # SAM3 trajectory reward (bottles)
    builder.py                      # Reward backend factory
  models/wan/                       # Wan2.2 model (vendored from vidar)

scripts/
  finetune/                         # Training launch scripts
  inference/                        # Inference launch scripts

tools/                              # Standalone analysis & visualization
tests/                              # Unit tests

data/
  rl_train/                         # Training datasets (prompt + reference images)
  outputs/                          # Experiment outputs (gitignored)
```

## Acknowledgement

Built on top of:
- [DanceGRPO](https://github.com/XueZeyue/DanceGRPO) — GRPO for visual generation
- [FastVideo](https://github.com/hao-ai-lab/FastVideo) — scalable video generation framework
- [Vidar / RoboTwin](https://robotwin-benchmark.github.io/early-version/) — robotics benchmark & model
- [SAM3](https://github.com/anthropics/segment-anything-3) — Segment Anything Model 3

## Citation

If you use this work, please cite:

```bibtex
@article{xue2025dancegrpo,
  title={DanceGRPO: Unleashing GRPO on Visual Generation},
  author={Xue, Zeyue and Wu, Jie and Gao, Yu and Kong, Fangyuan and Zhu, Lingting and Chen, Mengzhao and Liu, Zhiheng and Liu, Wei and Guo, Qiushan and Huang, Weilin and others},
  journal={arXiv preprint arXiv:2505.07818},
  year={2025}
}
```
