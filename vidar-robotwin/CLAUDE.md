# CLAUDE.md — vidar-robotwin

## Overview

**vidar-robotwin** is a client-server evaluation framework for testing video-based robot manipulation policies on the RoboTwin benchmark (SAPIEN physics simulator). It measures manipulation success rates by:

1. **Server** — runs a video generation model (Wan2.2 TI2V) + IDM (Inverse Dynamics Model) to predict future video frames and extract robot actions
2. **Client** — runs SAPIEN simulation, feeds observations to server, executes predicted actions, checks task success

## Architecture

```
┌─────────────────────────────┐     HTTP POST     ┌──────────────────────────────┐
│  Client (SAPIEN simulation) │ ──────────────────→│  Server (video gen + IDM)    │
│  policy/AR/deploy_policy.py │                    │  fastvideo/server_nft.py     │
│  script/eval_policy.py      │ ←──────────────────│  (in parent EmbodiedVideoRL) │
│  Conda env: RoboTwin-hb     │   actions + frames │  Conda env: wanx             │
└─────────────────────────────┘                    └──────────────────────────────┘
```

## Key Files

| Component | Path | Description |
|-----------|------|-------------|
| **Eval entry** | `script/eval_policy.py` | Main evaluation loop: setup env → run demo → generate instruction → execute policy → check success |
| **Client launcher** | `run_client.sh` | Single-GPU eval: `bash run_client.sh [TASK] [CONFIG] [PREFIX] [FRAMES] [STEPS] [CFG] [PORT]` |
| **DDP launcher** | `run_eval_ddp.sh` / `policy/AR/run_eval_ddp.py` | Multi-GPU eval: auto-starts servers, distributes tasks across GPUs |
| **AR policy** | `policy/AR/ar.py` | Client-side policy: caches observations, sends to server, receives actions |
| **Policy config** | `policy/AR/deploy_policy.yml` | Default eval params (frames, steps, rollout bounds) |
| **Task envs** | `envs/*.py` | SAPIEN task definitions: `load_actors()`, `play_once()`, `check_success()` |
| **Task configs** | `task_config/*.yml` | Environment configs (hd_clean, hd_randomized, etc.) |
| **Instructions** | `description/task_instruction/*.json` | Natural language instruction templates per task |
| **Step limits** | `task_config/_eval_step_limit.yml` | Max actions per task (put_object_cabinet: 800) |
| **Results** | `eval_result/ar/{PREFIX}/{TASK}/_result.txt` | Output: success rate (0.0–1.0) |

## Evaluation Flow

1. For each of 10 episodes (seeds):
   a. Setup SAPIEN env with random object placement
   b. Run expert demo to verify feasibility (skip infeasible seeds)
   c. Generate natural language instruction from template
   d. Get initial observation (head cam + 2 wrist cams → 640×736 composite)
   e. Policy loop: observation → server inference → actions → execute in sim → check success
   f. Record episode video
2. Output: `success_count / 10` to `_result.txt`

## Task: put_object_cabinet

- **File:** `envs/put_object_cabinet.py`
- **Description:** Open a cabinet drawer with one arm, place a random object inside with the other
- **Objects:** 10 variants (mouse, stapler, toy car, Rubik's cube, bread, phone, playing cards, tea box, coffee box, soap)
- **Success criteria:** Object height 7–120mm above origin, within 5cm of cabinet functional point, gripper open
- **Step limit:** 800 actions

## Server Protocol (vidar-compatible)

Client sends `POST http://localhost:{port}` with JSON:
```json
{
  "prompt": "The whole scene is in a realistic...",
  "imgs": ["<base64_jpeg>", ...],
  "num_conditional_frames": 1,
  "num_new_frames": 120,
  "seed": 1234,
  "num_sampling_step": 20,
  "guide_scale": 5.0,
  "password": "r49h8fieuwK",
  "return_imgs": true
}
```

Server returns: `{"actions": "[...]", "imgs": [...], "masks": [...]}`

## Running Evaluation

### Prerequisites
- **Server conda env:** `wanx` (in parent EmbodiedVideoRL repo)
- **Client conda env:** `RoboTwin-hb` (SAPIEN + RoboTwin deps)
- **GPU node** with at least 1× A100

### Step 1: Start server (in EmbodiedVideoRL root)
```bash
# Base vidar model (no NFT LoRA):
NFT_LORA_PATH="" PT_DIR=ckpts/vidar_ckpt/merged_vidar_lora.pt \
    conda run -n wanx bash scripts/inference/start_server_nft.sh 25400 0

# With NFT LoRA:
PT_DIR=ckpts/vidar_ckpt/merged_vidar_lora.pt \
    NFT_LORA_PATH=data/outputs/nft_put_object_cabinet/checkpoints/lora_final.pt \
    conda run -n wanx bash scripts/inference/start_server_nft.sh 25400 0
```

### Step 2: Run client (in vidar-robotwin/)
```bash
conda run -n RoboTwin-hb bash run_client.sh put_object_cabinet hd_clean test_run 120 20 5.0 25400
```

### Step 3: Check results
```bash
cat eval_result/ar/test_run/put_object_cabinet/_result.txt
```

## Existing Results

| Test | Date | Config | Success Rate |
|------|------|--------|-------------|
| long_horizon_121_test | 2026-02-10 | 120 frames, 20 steps | 0% (0/10) |
| nft_test | 2026-02-17 | 120 frames, 20 steps | 0% (0/10) |

## Conda Environments

- `RoboTwin-hb`: SAPIEN simulator, numpy, opencv, PyYAML, requests
- `wanx`: PyTorch, Wan2.2 model, PEFT, flash-attn, uvicorn (for server)

## Notes

- `run_client.sh` does NOT start the server — server must be running before client starts
- `run_eval_ddp.sh` auto-starts servers via `../server/stand_worker.sh` (may not exist; use manual server start instead)
- Default `rollout_bound=120, rollout_prefill_num=1` in `run_client.sh` means: generate all 120 frames in one shot from a single conditioning frame, no multi-round rollout
- SAPIEN requires headless rendering support (EGL or Vulkan) — works on GPU nodes, not login nodes
