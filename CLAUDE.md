# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DanceGRPO** is the official implementation of "DanceGRPO: Unleashing GRPO on Visual Generation" (arXiv:2505.07818). It applies Group Relative Policy Optimization (GRPO) and DiffusionNFT (a contrastive RL alternative) to fine-tune pre-trained image and video generation models using reward feedback.

**Active work in this repo** focuses on Wan-2.2 TI2V (Text-Image-to-Video, 5B), using the Vidar/RoboTwin robotics domain. The model loader code lives in `fastvideo/models/wan/` (vendored from the vidar repository), and checkpoints are under `ckpts/`.

**Project goal:** Develop a **universal reward function** for judging the quality of manipulation video generation across 6 RoboTwin tasks, and validate it with GRPO / DiffusionNFT / other flow-model RL algorithms.

## RoboTwin Tasks

Six manipulation tasks from the RoboTwin benchmark, each with 10 scenes (reference images + prompts).

| # | Task | Description | Objects to track | Reward status |
|---|------|-------------|------------------|---------------|
| 1 | **put_object_cabinet** | Open a cabinet drawer with one arm, place an object inside with the other arm | cabinet, object (varies) | ✅ Gemini API reward working; NFT training converged |
| 2 | **blocks_ranking_rgb** | Arrange red, green, blue blocks left-to-right in order | red block, green block, blue block (count=1 each) | ✅ SAM3 hallucination reward working; NFT training in progress |
| 3 | **put_bottles_dustbin** | Grab 3 bottles and put them into the dustbin | bottle (count: 3→2→1→0, monotonic decrease) | ✅ SAM3 trajectory-based reward working; NFT training in progress |
| 4 | **blocks_ranking_size** | Move 3 blocks (random colors) to center, arrange largest→smallest left-to-right | 3 blocks (varying colors & sizes) | ❌ No reward yet |
| 5 | **stack_blocks_three** | Move red/green/blue blocks to center, stack blue→green→red | red block, green block, blue block | ❌ No reward yet |
| 6 | **stack_bowls_three** | Stack 3 bowls on top of each other | 3 bowls | ❌ No reward yet |

**Reward design challenges by task type:**
- **Static object count** (blocks_ranking_rgb, stack_blocks_three): SAM3 tracks N distinct objects, count should stay constant → existing `tools/detect_hallucination.py` works
- **Monotonic decrease** (put_bottles_dustbin): Objects legitimately disappear when placed; count must monotonically non-increase → requires `tools/detect_hallucination_bottles.py` with committed-count baseline
- **Interaction with containers** (put_object_cabinet): Object enters a container (drawer) — hard to track with SAM3 alone → Gemini API currently used
- **Spatial arrangement** (blocks_ranking_size, stacking): Need to verify relative position/order, not just object count → neither SAM3 count nor Gemini gives reliable signal yet

## Training Algorithms

Both algorithms fine-tune flow-based diffusion models (Wan-2.2 TI2V) with LoRA, using reward feedback from generated videos. The key difference is how they modify the flow model's ODE/SDE sampling to enable RL.

### GRPO — Flow GRPO (`train_grpo_wan_2_2_ti2v.py`)

Converts the flow model's ODE into an **SDE** by injecting noise at each denoising step, enabling log-probability computation for PPO-style policy gradient.

**Pipeline:**
1. Sample prompts + reference images from `dataset_json`
2. Generate `num_generations` videos per prompt via **SDE sampling** (Euler + noise injection)
3. Collect per-step latents and **log-probabilities** during rollout
4. Decode videos and compute rewards
5. Z-score normalize rewards within each group → advantages
6. Optional best-of-N selection (`--bestofn`)
7. Replay denoising steps with gradients → **PPO clipped objective** → optimizer step

**Key characteristics:**
- Requires log-prob computation → more VRAM, slower per step
- `--clip_range 1e-4`: PPO clip range
- Gradient accumulation: `accumulate_samples * num_timesteps`

**Debugging tip:** Before gradient update, the probability ratio must be exactly 1.0. Set rollout and training batch sizes to 1 to verify.

```bash
bash scripts/finetune/finetune_wan_2_2_ti2v_grpo.sh
```

### DiffusionNFT — Contrastive RL (`train_nft_wan_2_2_ti2v.py`)

Uses the flow model's original **deterministic ODE** sampling (no noise injection, no log-prob). Instead applies a contrastive loss that pushes model velocity toward high-reward trajectories and away from low-reward ones.

**Pipeline:**
1. Load next prompt + reference image from dataset (shuffled, cycles on exhaustion)
2. Generate `num_generations` videos via **ODE sampling** (deterministic, no noise injection)
3. Each video evaluated by reward backend → binary reward (0 or 1)
   - Gemini API (`--reward_backend gpt`): LLM judges task completion from video frames
   - SAM3 Hallucination (`--reward_backend hallucination`): tracks objects across frames, binary reward
4. Rewards z-score normalized → advantages clipped to `[-adv_clip_max, adv_clip_max]` → mapped to `[0, 1]` as `r` values
5. If all rewards uniform (all 0 or all 1) → skip update (zero contrastive gradient)
6. For each video, sample `timestep_fraction × sample_steps` random timesteps:
   - `cur_pred`: current model (default adapter) forward
   - `old_pred`: EMA copy (old adapter) forward, no grad
   - `ref_pred`: base model (no adapter) forward, no grad — skipped if `kl_beta < 1e-3`
7. NFT contrastive loss:
   - `positive_pred = nft_beta × cur + (1-nft_beta) × old`
   - `negative_pred = (1+nft_beta) × old - nft_beta × cur`
   - `loss_i = r_i × positive_loss + (1-r_i) × negative_loss`
   - Optional temporal consistency: `temporal_lambda × (r × ||Δt(x0_pos) - Δt(x0)||² + (1-r) × ||Δt(x0_neg) - Δt(x0)||²)` where `Δt` = frame-to-frame diff along latent temporal axis. Inspired by DreamDojo.
   - Optional KL penalty: `kl_beta × ||cur - ref||²`
8. Accumulate gradients over `gradient_accumulation_steps` prompts → `optimizer.step()`
9. EMA update old adapter: `old ← old × decay + default × (1-decay)`

**Key characteristics:**
- Dual LoRA adapters: `default` (trainable) + `old` (EMA reference copy)
- No log-prob overhead → faster, less VRAM than GRPO
- Key args: `--nft_beta`, `--kl_beta`, `--decay_type`

**NFT hyperparameters:**
| Arg | Default | Effect |
|-----|---------|--------|
| `--nft_beta 1.0` | 1.0 | Interpolation between cur/old; 1.0 = full contrastive |
| `--kl_beta 0.0001` | 0.0001 | KL penalty against base model; < 1e-3 skips ref forward pass |
| `--adv_clip_max 1.0` | 1.0 | Clip range for z-score advantages; also scales total loss |
| `--timestep_fraction 0.5` | 0.5 | Fraction of denoising steps used for training |
| `--decay_type 1` | 1 | EMA decay schedule type for old adapter update |
| `--temporal_lambda 0.0` | 0.0 | Temporal consistency loss weight; 0 = disabled, 0.1 = recommended |
| `--gradient_accumulation_steps 4` | 4 | Prompts accumulated before optimizer.step(); set to 1 for single-scene |
| `--num_generations 8` | 8 | Videos per prompt; must be divisible by world_size for multi-GPU |
| `--raw_reward_as_r` | false | Use raw binary reward (0/1) directly as `r` instead of z-score normalization. Recommended for SAM3 binary rewards. |
| `--nft_bestofn 0` | 0 | Among CLEAN (reward>=0.5) videos, keep only the best (lowest `motion_score`) as positive sample. All FAIL videos kept. 0 = disabled. Requires `--raw_reward_as_r`. |

**Best-of-N selection (`--nft_bestofn`):**
When enabled, after reward scoring, positive samples are ranked by `motion_score = total_trajectory_length + max_speed` (computed from SAM3 object tracking). Only the best (lowest score = smoothest trajectory) positive is kept; all negatives are kept. This gives a cleaner contrastive signal than treating all CLEAN videos equally. If no negatives remain after filtering, training is skipped for that step (no contrastive gradient). The `motion_score` is logged to JSONL and wandb for analysis.

### SFT — Supervised Fine-Tuning (`train_sft_wan_2_2_ti2v.py`)

Standard supervised fine-tuning with ground-truth demonstration videos using flow-matching MSE loss. Based on DiffSynth-Studio's implementation.

**Pipeline:**
1. Load ground-truth video + prompt + reference image from dataset JSON
2. Encode text (T5), reference image (VAE), and video (VAE) to latents
3. Sample random timesteps from the 1000-step training schedule
4. For each timestep:
   - Construct noisy latent `x_t = (1-σ)*x0 + σ*noise`
   - First-frame fusion: keep image latent at frame 0 (mask2 mechanism)
   - Forward model → velocity prediction
   - Velocity target = `noise - x0`
   - BSMNTW-weighted MSE loss, excluding first frame
5. Optimizer step with gradient accumulation

**Key characteristics:**
- Single LoRA adapter (no dual adapters, no reward scoring)
- 1000-step training sigma schedule (not 20-step inference schedule)
- BSMNTW timestep weighting (DiffSynth-Studio style)
- Direct dense supervision → faster convergence than RL, but requires GT videos
- Useful for: pre-training before RL, imitation learning, baseline comparison

**SFT hyperparameters:**
| Arg | Default | Effect |
|-----|---------|--------|
| `--num_train_timesteps 1000` | 1000 | Sigma schedule resolution for training |
| `--timestep_fraction 0.05` | 0.05 | Fraction of 1000 timesteps trained per sample (50 timesteps) |
| `--use_bsmntw true` | true | BSMNTW timestep weighting (false = uniform) |
| `--num_epochs 100` | 100 | Epochs over the dataset |
| `--learning_rate 1e-4` | 1e-4 | Higher than RL (1e-5) since supervision is stronger |
| `--lora_rank 32` | 32 | Smaller rank often sufficient for SFT |
| `--gradient_accumulation_steps 4` | 4 | Samples accumulated before optimizer.step() |

**Dataset JSON format** (extends RL format with `video_path`):
```json
[{"prompt": "...", "media_path": "ref_img.png", "video_path": "demo.mp4", "filename_stem": "scene_001"}]
```

```bash
bash scripts/finetune/finetune_wan_2_2_ti2v_sft.sh
```

### Distributed Training (all algorithms)

- **GRPO/NFT:** `num_generations` videos split across GPUs: each rank generates `num_generations / world_size`. Rewards gathered via `all_gather` before z-score normalization.
- **SFT:** Each rank processes different samples via `DistributedSampler`. No reward gathering needed.
- Gradients synchronized via DDP `allreduce`
- Default: 4 GPUs (A100-80GB)

### General Hyperparameter Notes

- Learning rate: 1e-5 to 2e-5 for RL (below 5e-6 leads to training failure); 1e-4 for SFT
- `--max_grad_norm 2.0`: Reward collapse can occur if too high; reduce if unstable
- `--timestep_fraction 0.5` (RL) / `0.05` (SFT): Fraction of denoising steps to train on
- Checkpoints: `lora_stepXXXXXX.pt`; `--resume_from_lora_checkpoint` resumes from specific step
- `--offload_model true`: Offload VAE/T5 to CPU during training phases

## Reward Functions

Two reward backends are currently implemented and integrated into the training loop:

### 1. Gemini API (`--reward_backend gpt`)

LLM-based reward: extracts 4 frames from generated video, sends to Gemini as images, asks whether the task was completed → binary reward.

- Used for: **put_object_cabinet** (successfully trained)
- Pros: General-purpose, can judge complex task semantics
- Cons: API cost, latency, non-deterministic, hard to scale
- Debug output: `reward_debug/step{NNNN}_{stem}/{stem}_g{i}_s{seed}_PASS.jpg` or `_FAIL.jpg` (2×2 frame grid with Gemini response text)

### 2. SAM3 Hallucination Detection (`--reward_backend hallucination`)

SAM3-based object tracking: tracks specified objects across all video frames using text prompts, counts per-frame instances vs expected counts → binary reward (1.0 = no hallucination, 0.0 = hallucination detected).

- Used for: **blocks_ranking_rgb** (NFT training in progress)
- Pros: Deterministic, no API cost, fast (~45s per 121-frame video)
- Cons: Only checks object count consistency, not task completion semantics

**Implementation:** `HallucinationRewardScorer` class in `fastvideo/reward/hallucination.py` wraps `fastvideo.reward.hallucination_process.process_video()` behind the `RewardScorer` interface.

**Critical design choice:** Uses base `Sam3VideoPredictor` (NOT `Sam3VideoPredictorMultiGPU`) — the multi-GPU variant overwrites `MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE` env vars (lines 303-306 of `sam3_video_predictor.py`) which would corrupt the torchrun DDP process group.

**Multi-GPU notes:**
- Each rank loads its own SAM3 predictor on its local GPU (~2-3GB VRAM per rank)
- Device placement: `torch.cuda.set_device(local_rank)` before `Sam3VideoPredictor()` init
- With `--offload_model true`, single-GPU peak ~28GB; multi-GPU works within A100-80GB

**Two-stage post-processing:**
1. **Occlusion suppression** (disappearance): When count < expected for ≤ `occlusion_gap_max` consecutive frames and the object reappears within `occlusion_pos_thr` L∞ distance, suppress those frames.
2. **Duplication spike suppression**: When count > expected for ≤ `duplication_spike_max` consecutive frames, suppress those frames. This filters SAM3 false positives where the robot gripper or shadows are briefly misidentified as an extra object.

**Hallucination CLI args:**
| Arg | Default | Description |
|-----|---------|-------------|
| `--hallucination_prompts` | `"red block" "green block" "blue block"` | Text prompts for SAM3 object tracking |
| `--hallucination_expected_counts` | 1 per prompt | Expected count per object |
| `--hallucination_crop_top_ratio` | 0.6667 | Crop top portion of frame (isolate main camera) |
| `--occlusion_gap_max` | 5 | Max frames for occlusion suppression (disappearance) |
| `--occlusion_pos_thr` | 0.15 | Position threshold for occlusion suppression |
| `--duplication_spike_max` | 0 | Max frames for duplication spike suppression (0=disabled) |

**Debug output:** `reward_debug/step{NNNN}_{stem}/{video_stem}_CLEAN.mp4` or `_HALL.mp4` (SAM3 annotated video with bounding boxes)

**Motion score:** All three SAM3 reward scorers (hallucination, hallucination_bottles, hallucination_bowls) return a continuous `motion_score` alongside the binary reward. Computed as `total_trajectory_length + max_speed` from SAM3 per-object tracking data (see `sam3_utils.compute_motion_score_from_objects()`). Lower = smoother trajectory. Used by `--nft_bestofn` for best-of-N selection among CLEAN videos.

### 3. SAM3 Bottle Trajectory Hallucination (`--reward_backend hallucination_bottles`)

SAM3-based **trajectory tracking** for put_bottles_dustbin: tracks "bottle" across all frames, builds per-object trajectories, then applies a **two-stage** approach — first filter SAM3 tracking errors, then check 3 simple hallucination criteria. Binary reward: 1.0 = CLEAN, 0.0 = FAIL.

- Used for: **put_bottles_dustbin** (NFT training in progress)
- Pros: Handles legitimately disappearing objects (bottles placed in dustbin); deterministic, no API cost
- Cons: Tuned specifically for put_bottles_dustbin spatial layout (dustbin on the left side)

**Implementation:** `BottleHallucinationRewardScorer` class in `fastvideo/reward/hallucination_bottles.py`. Uses SAM3 single text prompt "bottle" + per-object trajectory analysis.

**Two-stage algorithm (`_analyze()`):**

*Stage 1 — Filter SAM3 tracking errors:*
1. **Spurious object filter:** Objects first appearing after frame 5 are discarded (SAM3 artifacts)
2. **Gap-based trajectory splitting:** Each trajectory is split at gaps > `filter_max_gap` (5) frames. Only the **first continuous segment** is kept. Detections after a large gap are likely tracking the gripper or scene noise, not the real bottle. This naturally handles:
   - **Teleportation:** bottle disappears for >5 frames then appears at dustbin → first segment doesn't reach dustbin → fails all_placed
   - **Pre-placement reappearance:** large gap mid-trajectory → split, first segment only
   - **Post-placement ghosts:** SAM3 re-acquires gripper after bottle is placed → separate segment, discarded

*Stage 2 — 3-criteria hallucination check (on cleaned trajectories):*
1. **(a) all_placed:** ≥ 3 bottles survive filtering AND all reach cx < `cx_cutoff` (0.26)
2. **(b) monotonic:** Active bottle count must be monotonically non-increasing (brief spikes ≤ `spike_max` frames suppressed)
3. **(c) no_merge:** No two bottles share the same `placed_frame`

**Key design decisions:**
- **Trajectory truncation at cx_cutoff:** Once a bottle's center-x drops below 0.26 (dustbin area), all subsequent SAM3 detections for that object are ignored — SAM3 frequently tracks the robot gripper as a bottle after the real bottle is thrown away
- **Gap splitting > h/w filtering:** Per-detection h/w filtering was tested but found harmful — removing individual low-h/w detections creates artificial gaps in otherwise continuous trajectories, triggering false monotonic failures. Gap-based splitting at the trajectory level is sufficient and more robust.
- **filter_max_gap = 5:** In CLEAN videos, all pre-placement tracking is perfectly continuous (zero gaps). FAIL videos have gaps of 8–44 frames before placement. Threshold of 5 cleanly separates the two.

**Validation:** Tested on 80 base-model rollout videos (10 scenes × 8 videos) with ground truth at `data/outputs/bottle_hall_50steps/ground_truth.json`. Results: **100% accuracy** (38 CLEAN, 42 FAIL correctly classified). Test script: `test_bottle_hallucination.py`.

**Bottle Hallucination CLI args:**
| Arg | Default | Description |
|-----|---------|-------------|
| `--bottle_hall_prompt` | `"bottle"` | SAM3 text prompt for bottle tracking |
| `--bottle_cx_cutoff` | 0.26 | Truncate trajectory when cx < this (bottle reached dustbin) |
| `--bottle_spike_max` | 3 | Max frames for brief count-spike suppression in monotonic check |
| `--bottle_filter_max_gap` | 5 | Split trajectory at gaps > this many frames (SAM3 error filtering) |
| `--hallucination_crop_top_ratio` | 0.6667 | Shared with blocks hallucination; crops wrist cameras |

**Debug output:** `reward_debug/step{NNNN}_{stem}/{video_stem}_CLEAN.mp4` or `_FAIL.mp4` (SAM3 annotated tracking video with color-coded bboxes, trajectory trails, and cx_cutoff line)

### 4. Flow AEPE — Forward-Backward Consistency (`--reward_backend flow_aepe`)

SEA-RAFT optical flow-based temporal consistency check: computes optical flow between consecutive frames in both directions (forward and backward), measures forward-backward End-Point Error (EPE). High EPE ⇒ temporal inconsistency ⇒ hallucination. Binary reward: 1.0 if `score >= threshold`, 0.0 otherwise.

- Used for: **task-agnostic** hallucination detection (applicable to any task)
- Pros: Task-agnostic (no object-specific prompts needed), detects temporal inconsistencies like object teleportation/morphing
- Cons: Requires GPU (SEA-RAFT neural network); doesn't detect semantic errors (e.g. wrong object count if temporally smooth)

**Implementation:** `FlowAEPERewardScorer` class in `fastvideo/reward/flow_aepe.py`. Based on WorldArena's `flow_aepe_metrics.py`. Uses SEA-RAFT (ECCV 2024) model for optical flow estimation.

**Algorithm:**
1. Read video frames (optionally crop wrist cameras via `crop_top_ratio`, subsample via `frame_step`)
2. For each consecutive frame pair: compute forward flow (f₁→f₂) and backward flow (f₂→f₁) using SEA-RAFT
3. Compute forward-backward consistency EPE: warp pixel through forward flow, then backward flow, measure displacement from original position
4. Average EPE across all pairs → `avg_epe`
5. Compute dynamic motion score (mean magnitude of top-5% motion pixels, soft-thresholded) → `dynamic_degree`
6. Final score = `1/avg_epe`, modulated by `dynamic_degree` if scene is near-static (≤0.1213)
7. Binary reward: `score >= threshold` → 1.0, else 0.0

**SEA-RAFT model:** Uses `Tartan-C-T-TSKH-spring540x960-M` checkpoint (19.7M params) from HuggingFace `MemorySlices/Tartan-C-T-TSKH-spring540x960-M`. Config: `spring-M.json` (ResNet34 backbone, 4 iters, 540×960 resolution).

**External dependency:** SEA-RAFT code at `/gpfs/projects/p33048/WorldArena/video_quality/WorldArena/third_party/SEA-RAFT/`, checkpoint at `third_party/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth`.

**Flow AEPE CLI args:**
| Arg | Default | Description |
|-----|---------|-------------|
| `--flow_aepe_cfg` | `.../SEA-RAFT/config/eval/spring-M.json` | SEA-RAFT config JSON path |
| `--flow_aepe_ckpt` | `.../checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth` | SEA-RAFT checkpoint path |
| `--flow_aepe_threshold` | 0.5 | Flow score threshold for binary reward (score ≥ thr → reward=1) |
| `--flow_aepe_frame_step` | 1 | Subsample frames (1=every frame, 2=every other, etc.) |
| `--hallucination_crop_top_ratio` | 0.6667 | Shared; crops wrist cameras from bottom |

**Performance:** ~3.4s per frame pair on A100 GPU, ~34s per frame pair on CPU. For 121-frame video (120 pairs): ~7 min (GPU) or ~68 min (CPU). Use `--flow_aepe_frame_step 5` for ~5× speedup.

### Reward Backend Summary

| Backend | Flag | Used for | Status |
|---------|------|----------|--------|
| Gemini API | `--reward_backend gpt --gpt_model <model>` | put_object_cabinet | ✅ Training converged |
| SAM3 Hallucination | `--reward_backend hallucination` | blocks_ranking_rgb | ✅ Training in progress |
| SAM3 Bottle Trajectory | `--reward_backend hallucination_bottles` | put_bottles_dustbin | ✅ Training in progress |
| Flow AEPE | `--reward_backend flow_aepe` | Task-agnostic temporal consistency | 🔧 Testing |
| HPS-v2.1 | `--use_hpsv2` | Image models (SD, FLUX, Wan-2.1) | (upstream) |
| VideoAlign | `--use_videoalign` | Video models (HunyuanVideo, SkyReels) | (upstream) |
| PickScore | `--use_pickscore` | Alternative for FLUX | (upstream) |

## Hallucination Detection Tools (standalone)

SAM3-based video analysis tools, used both as standalone analysis and as reward functions during training.

### Blocks Hallucination (`tools/detect_hallucination.py`)

Tracks N distinct objects (e.g. red/green/blue blocks) across frames. Object count should stay **constant** throughout the video. Any count deviation = hallucination (after occlusion suppression).

```bash
# Single video
ssh <node> "conda run -n wanx python tools/detect_hallucination.py \
    --input /path/to/video.mp4 --out-dir /path/to/out_dir"

# Batch (preferred, with occlusion suppression)
ssh <node> "conda run -n wanx python tools/run_hallucination_batch_improved.py \
    --input-root data/outputs/rollout_robotwin_121 \
    --out-root  data/outputs/rollout_robotwin_121_hallucination \
    --pattern   robotwin_blocks_ranking_rgb_123500001"
```

**Occlusion suppression:** Brief disappearances ≤ `--occlusion-gap-max` frames (default 5) that reappear within `--occlusion-pos-thr` (0.15) normalised L∞ distance are not counted as hallucinations.

### Bottles Hallucination (`tools/run_bottle_hallucination_pipeline.py`)

Full pipeline for put_bottles_dustbin: SAM3 tracking → per-object trajectory extraction → trajectory truncation at cx_cutoff → hallucination check → annotated video rendering. This is the standalone version of the logic used by `BottleHallucinationRewardScorer` in training.

```bash
# Batch analysis (all 10 scenes, 80 videos)
ssh <node> "conda run -n wanx python tools/run_bottle_hallucination_pipeline.py \
    --input-root data/outputs/rollout_robotwin_121 \
    --out-root   data/outputs/bottle_hall_v2 \
    --pattern    'robotwin_put_bottles_dustbin_*' \
    --cx-cutoff 0.26 --gap-max 3"
```

**Two-stage check:** Stage 1 filters SAM3 tracking errors by splitting trajectories at gaps > 5 frames. Stage 2 applies 3 criteria: (a) all 3 bottles placed, (b) monotonic count, (c) no merge. See "SAM3 Bottle Trajectory Hallucination" reward section above for full details.

**Ground truth + test:** `data/outputs/bottle_hall_50steps/ground_truth.json` has 80 manually-verified labels (38 CLEAN, 42 FAIL). `tests/test_bottle_hallucination.py` runs the simplified algorithm against ground truth and reports accuracy/precision/recall/F1.

```bash
# Test algorithm against ground truth (no GPU needed)
python tests/test_bottle_hallucination.py
python tests/test_bottle_hallucination.py --filter-max-gap 8   # sweep params
python tests/test_bottle_hallucination.py -v                    # verbose mismatch diagnostics
```

**Output per scene:** `{stem}_CLEAN.mp4` / `{stem}_FAIL.mp4` (tracking video), `{stem}_trajectory.json`, `{scene}_analysis.json`, `grand_summary.json`.

**Legacy:** `tools/detect_hallucination_bottles.py` uses the older frame-count-based approach (monotonic baseline). The trajectory-based pipeline (`tools/run_bottle_hallucination_pipeline.py`) supersedes it with better accuracy (handles SAM3 gripper-tracking artifacts).

### Optical Flow Anomaly Detection (`tools/detect_flow_anomalies.py`)

Detects motion anomalies (object drift, sudden appear/disappear) via Farneback optical flow. Outputs a 2×2 grid video (annotated frame, magnitude heatmap, HSV direction, quiver arrows).

```bash
ssh <node> "conda run -n wanx python tools/detect_flow_anomalies.py \
    --input /path/to/video.mp4 --out-dir /path/to/output"
# Batch
ssh <node> "nohup bash tools/run_flow_anomaly_batch.sh 4 > flow_batch.log 2>&1 &"
```

### Flow AEPE Batch Test (`tools/test_flow_aepe_reward.py`)

Batch-evaluates `FlowAEPERewardScorer` on rollout videos. Outputs per-video scores, per-scene summaries, and a ranked list (worst EPE first = most likely hallucination).

```bash
# GPU (recommended)
ssh <node> "conda run -n wanx --no-capture-output --cwd /gpfs/projects/p33048/DanceGRPO \
    python tools/test_flow_aepe_reward.py \
    --input-root data/outputs/rollout_robotwin_121 \
    --out-dir    data/outputs/flow_aepe_blocks_ranking_rgb \
    --pattern    'robotwin_blocks_ranking_rgb_*' \
    --frame-step 1 --device 0"

# Faster (subsample every 5th frame)
python tools/test_flow_aepe_reward.py --frame-step 5

# CPU fallback (slow)
CUDA_VISIBLE_DEVICES="" python tools/test_flow_aepe_reward.py --frame-step 5
```

**Output:** `all_results.json` (per-video), `summary.json` (aggregate + per-scene), `ranked_by_epe.json` (sorted worst-first).

### SAM3 API Notes

- Text prompts call `reset_state` internally — cannot track multiple text prompts simultaneously
- Point prompts (`add_tracker_new_points`) do NOT reset state — multiple objects in single propagation
- `out_boxes_xywh` uses top-left (x, y, w, h) format, normalized 0–1
- `Sam3VideoPredictorMultiGPU` overwrites DDP env vars — **never use during torchrun training**

## Training Commands

### NFT + Gemini reward (put_object_cabinet)
```bash
# 4 GPU (default)
bash scripts/finetune/finetune_wan_2_2_ti2v_nft_put_object_cabinet.sh
```

### NFT + SAM3 hallucination reward (blocks_ranking_rgb)
```bash
# Single GPU (default, tested on qgpu2003)
bash scripts/finetune/finetune_wan_2_2_ti2v_nft_blocks_ranking_rgb.sh

# Multi-GPU (4× A100-80GB)
GPU_NUM=4 bash scripts/finetune/finetune_wan_2_2_ti2v_nft_blocks_ranking_rgb.sh

# Background training (replace <node> with your allocated GPU node from `squeue -u $USER`)
ssh <node> "nohup conda run -n wanx --no-capture-output --cwd /gpfs/projects/p33048/DanceGRPO \
    bash scripts/finetune/finetune_wan_2_2_ti2v_nft_blocks_ranking_rgb.sh \
    > data/outputs/nft_blocks_ranking_rgb_train.log 2>&1 &"
```

### NFT + SAM3 bottle trajectory reward (put_bottles_dustbin)
```bash
# Single GPU (default)
bash scripts/finetune/finetune_wan_2_2_ti2v_nft_put_bottles_dustbin.sh

# Multi-GPU (4× A100-80GB)
GPU_NUM=4 bash scripts/finetune/finetune_wan_2_2_ti2v_nft_put_bottles_dustbin.sh

# Background training
ssh <node> "nohup conda run -n wanx --no-capture-output --cwd /gpfs/projects/p33048/DanceGRPO \
    GPU_NUM=4 bash scripts/finetune/finetune_wan_2_2_ti2v_nft_put_bottles_dustbin.sh \
    > /gpfs/projects/p33048/DanceGRPO/data/outputs/nft_put_bottles_dustbin_train.log 2>&1 &"
```

### GRPO (put_object_cabinet)
```bash
bash scripts/finetune/finetune_wan_2_2_ti2v_grpo.sh
```

### SFT (supervised fine-tuning with GT videos)
```bash
# Single GPU (default)
bash scripts/finetune/finetune_wan_2_2_ti2v_sft.sh

# Multi-GPU (4× A100-80GB)
GPU_NUM=4 bash scripts/finetune/finetune_wan_2_2_ti2v_sft.sh

# Custom dataset
DATASET_JSON=data/sft_train/my_task.json OUTPUT_DIR=data/outputs/sft_my_task \
    bash scripts/finetune/finetune_wan_2_2_ti2v_sft.sh
```

### Key training arguments
- `--ckpt_dir`: Wan2.2-TI2V-5B model directory (default: `ckpts/Wan2.2-TI2V-5B`)
- `--pt_dir`: Pre-merged LoRA `.pt` weights (default: `ckpts/vidar_ckpts/vidar_merged_lora.pt`)
- `--vidar_root`: (optional fallback) Path to vidar repo, only needed if `fastvideo/models/wan/` is not present
- `--dataset_json`: JSON with `{prompt, image}` pairs for rollout
- `--num_generations 8`: Videos generated per prompt per step
- `--reward_backend gpt --gpt_model gemini-3-flash-preview`: Reward via Gemini API
- `--reward_backend hallucination`: Reward via SAM3 hallucination detection — constant count (binary: 0/1)
- `--reward_backend hallucination_bottles`: Reward via SAM3 trajectory-based hallucination — monotonic decrease (binary: 0/1)
- `--use_lora true --lora_rank 64 --lora_alpha 64`: LoRA training
- `--lora_target_modules`: Override LoRA target modules (see below)

### LoRA Target Modules

**Default (attention-only):** When `--lora_target_modules` is not specified, both GRPO and NFT scripts target attention layers only:
```
self_attn.q  self_attn.k  self_attn.v  self_attn.o
cross_attn.q cross_attn.k cross_attn.v cross_attn.o
```
This yields ~67M trainable params per adapter (rank=64, 32 DiT blocks).

**Full linear layers (attention + FFN):** To also target FFN layers (empirically better for fine-tuning quality, see QLoRA paper), add:
```bash
--lora_target_modules \
    "self_attn.q" "self_attn.k" "self_attn.v" "self_attn.o" \
    "cross_attn.q" "cross_attn.k" "cross_attn.v" "cross_attn.o" \
    "ffn.0" "ffn.2"
```
This targets the two Linear layers in `WanAttentionBlock.ffn` (an `nn.Sequential(Linear(2048,8192), GELU, Linear(8192,2048))`), increasing params to ~109M per adapter (+63%). With dual NFT adapters, adds ~168MB VRAM at bf16 — fits on A100-80GB.

**DiffSynth-Studio comparison:** DiffSynth's SFT training uses `q,k,v,o,ffn.0,ffn.2` (all linear) with rank=32, alpha=32 by default.
- `--resume_from_lora_checkpoint`: Resume from a `.pt` checkpoint
- `--max_train_steps`: Global step count (including any resumed steps)
- `--checkpointing_steps 10`: Saves `lora_stepXXXXXX.pt` under `output_dir/checkpoints/`

## Inference

**Key files:**
- `fastvideo/infer_nft.py` — CLI inference: loads WanTI2V + NFT LoRA, generates videos from prompt+image
- `fastvideo/server_nft.py` — FastAPI server wrapping the same model; vidar-protocol compatible
- `scripts/inference/infer_nft.sh` — shell launcher for CLI inference
- `scripts/inference/start_server_nft.sh` — shell launcher for server (mirrors `vidar/start_server.sh`)

**LoRA key conversion:** NFT training saves peft-format keys (`base_model.model.blocks.0.self_attn.q.lora_A.default.weight`). `fastvideo/infer_nft.py::convert_nft_lora_keys()` strips the peft prefix and filters out the `old` adapter before calling vidar's `load_lora()`. After fusion, `model.model.to(device)` is required because `fuse_lora_to_model` runs on CPU.

**Batch inference (dataset JSON or single prompt):**
```bash
bash scripts/inference/infer_nft.sh
# Override LoRA checkpoint:
NFT_LORA_PATH=data/outputs/nft_put_object_cabinet/checkpoints/lora_step000100.pt \
    bash scripts/inference/infer_nft.sh
# Single prompt + image:
PROMPT="A robot opens the cabinet." IMAGE=/path/to/img.png \
    bash scripts/inference/infer_nft.sh
```

**API server (compatible with vidar `run_client.sh`):**
```bash
bash scripts/inference/start_server_nft.sh [PORT] [CUDA_DEV] [NFT_LORA_PATH] [LORA_ALPHA]
bash scripts/inference/start_server_nft.sh 25400 0
bash scripts/inference/start_server_nft.sh 25401 1 data/outputs/nft_model/checkpoints/lora_step000100.pt 1.0
# Without IDM:
IDM_PATH="" bash scripts/inference/start_server_nft.sh 25400 0
```

Server endpoints: `POST /` (vidar-compatible), `POST /generate` (simplified), `GET /info`.

Key env vars: `CKPT_DIR`, `PT_DIR`, `NFT_LORA_PATH`, `LORA_ALPHA`, `IDM_PATH`, `IDM_CPU`, `SAVE_VIDEO`, `SAVE_VIDEO_DIR`.

## SAPIEN Evaluation (vidar-robotwin)

The `vidar-robotwin/` subdirectory contains a client-server evaluation framework for testing video-based manipulation policies in the SAPIEN physics simulator. It measures manipulation success rates on RoboTwin benchmark tasks.

### Architecture

- **Server** (`fastvideo/server_nft.py`): Runs Wan2.2 TI2V video generation + IDM (Inverse Dynamics Model) to predict future frames and extract robot actions
- **Client** (`vidar-robotwin/script/eval_policy.py`): Runs SAPIEN simulation, feeds observations to server, executes predicted actions, checks task success
- Both run in the same `wanx` conda environment on a single GPU node

### Running Evaluation

**All-in-one wrapper (recommended):**
```bash
# On a GPU node with 1× A100-80GB:
cd /path/to/EmbodiedVideoRL
bash scripts/eval/eval_vidar_put_object_cabinet.sh
```

This script:
1. Starts the video generation server on port 25400 (background)
2. Waits for server readiness
3. Runs SAPIEN client for 10 episodes of put_object_cabinet
4. Kills server, reports success rate

**Key environment variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `PT_DIR` | `ckpts/vidar_ckpt/merged_vidar_lora.pt` | Video generation model weights |
| `IDM_PATH` | `ckpts/vidar_ckpt/idm.pt` | Inverse dynamics model weights |
| `NFT_LORA_PATH` | (empty) | Optional NFT LoRA checkpoint |
| `CUDA_DEV` | `0` | GPU device index |
| `PORT` | `25400` | Server port |
| `PREFIX` | `vidar_ckpt_test` | Output subdirectory name |

**Manual server + client (for debugging):**
```bash
# Terminal 1 — start server:
NFT_LORA_PATH="" bash scripts/inference/start_server_nft.sh 25400 0

# Terminal 2 — run client:
cd vidar-robotwin
conda run -n wanx bash run_client.sh put_object_cabinet hd_clean test_run 120 20 5.0 25400

# Check results:
cat vidar-robotwin/eval_result/ar/test_run/put_object_cabinet/_result.txt
```

### SAPIEN Environment Verification

Test that SAPIEN + curobo work correctly (no server needed):
```bash
conda run -n wanx python scripts/eval/test_sapien_env.py
# Expected: setup_demo: SUCCESS, play_once: SUCCESS
```

### Environment Setup

See `vidar-robotwin/SETUP.md` for full portable setup instructions. Key steps:
1. Install SAPIEN/simulation deps into wanx: `sapien==3.0.0b1`, `mplib==0.2.1`, `gymnasium`, `transforms3d`, `uvicorn`, `fastapi`
2. Install ffmpeg: `conda install -n wanx -c conda-forge ffmpeg`
3. Apply sapien/mplib patches (encoding fix, collision check fix)
4. Download assets: `bash vidar-robotwin/script/_download_assets.sh`

### Path Portability (${VIDAR_ROOT})

CuRobo YAML configs (`vidar-robotwin/assets/embodiments/*/curobo*.yml`) use `${VIDAR_ROOT}` placeholders for all absolute paths. These are resolved at runtime by `envs/robot/robot.py::_resolve_curobo_yml()` using the vidar-robotwin root directory (computed from `envs/_GLOBAL_CONFIGS.py::ROOT_PATH`). No hardcoded absolute paths — the configs work on any machine as long as the relative directory structure is preserved.

## Data Directory

```
data/rl_train/                              # Training datasets (prompt + reference image per scene)
  robotwin_put_object_cabinet.json          # 10 scenes
  robotwin_blocks_ranking_rgb.json          # 10 scenes
  robotwin_put_bottles_dustbin.json         # 10 scenes
  robotwin_blocks_ranking_size.json         # 11 scenes
  robotwin_stack_blocks_three.json          # 10 scenes
  robotwin_stack_bowls_three.json           # 10 scenes
  put_object_cabinet/*.png                  # Reference images per task
  blocks_ranking_rgb/*.png
  put_bottles_dustbin/*.png
  blocks_ranking_size/*.png
  stack_blocks_three/*.png
  stack_bowls_three/*.png

data/sft_train/                             # SFT datasets (prompt + reference image + GT video)
  robotwin_sft.json                         # JSON with video_path entries (user-provided)

data/outputs/                               # All experiment outputs
  nft_put_object_cabinet/                   # NFT + Gemini reward (converged)
  nft_blocks_ranking_rgb/                   # NFT + SAM3 hallucination (in progress)
  sft_robotwin/                             # SFT with GT videos
  rollout_robotwin_121/                     # Base model rollout videos (all 6 tasks)
  bottle_hall_50steps/                      # Bottle trajectory analysis (38/80 clean) + ground_truth.json
```

**RL Dataset JSON format:**
```json
[
  {
    "prompt": "The whole scene is in a realistic, industrial art style ...",
    "filename_stem": "robotwin_put_object_cabinet_123500051",
    "media_path": "/path/to/reference_image.png"
  }
]
```

**SFT Dataset JSON format** (extends RL format with `video_path`):
```json
[
  {
    "prompt": "The whole scene is in a realistic, industrial art style ...",
    "filename_stem": "robotwin_put_object_cabinet_123500051",
    "media_path": "/path/to/reference_image.png",
    "video_path": "/path/to/ground_truth_video.mp4"
  }
]
```

**Training output structure:**
```
data/outputs/nft_<task>/
  checkpoints/
    lora_step000010.pt              # saved every --checkpointing_steps steps
    lora_final.pt                   # saved at end of training
  videos/
    step0001/{stem}_g{i}_s{seed}.mp4
  reward_debug/
    step0001_{stem}/                # Gemini: .jpg grids; Hallucination: _CLEAN.mp4 / _HALL.mp4
  reward_curve.png                  # mean reward over steps
  training_log.jsonl                # per-step JSON logs (loss, reward, grad_norm, etc.)
```

## Key File Paths

```
server/                             # IDM model code (vendored from vidar repo)
  idm.py                            #   IDM class (inverse dynamics model, optional)

ckpts/                              # Model checkpoints (symlinks or real files)
  Wan2.2-TI2V-5B/                   #   Base Wan2.2 TI2V 5B checkpoint directory
  vidar_ckpts/
    vidar_merged_lora.pt            #   Pre-merged LoRA weights (base + prior fine-tuning)
    idm.pt                          #   IDM model weights (optional, for server)

fastvideo/
  train_grpo_wan_2_2_ti2v.py        # GRPO training (SDE sampling + PPO)
  train_nft_wan_2_2_ti2v.py         # DiffusionNFT training (ODE sampling + contrastive)
  train_sft_wan_2_2_ti2v.py         # SFT training (flow-matching MSE with GT videos)
  reward/
    sam3_utils.py                   #   Shared SAM3 helpers: extract_frames_to_jpeg, track_prompt, save_video_libx264
    hallucination_process.py        #   Core process_video() for constant-count hallucination detection
    hallucination.py                #   HallucinationRewardScorer (blocks: constant count)
    hallucination_bottles.py        #   BottleHallucinationRewardScorer (bottles: two-stage trajectory)
    flow_aepe.py                    #   FlowAEPERewardScorer (SEA-RAFT forward-backward EPE)
    builder.py                      #   Reward backend factory + CLI args
  infer_nft.py                      # CLI inference script (NFT LoRA)
  server_nft.py                     # FastAPI server (NFT LoRA, vidar-compatible)
  predict.py                        # Cog predictor interface (upstream, dormant)
  train_grpo_*.py                   # Other model GRPO variants (upstream)
  models/wan/                       # Wan2.2 model code (vendored from vidar repo)
    __init__.py                     #   Exports WanI2V, WanT2V, WanTI2V
    textimage2video.py              #   WanTI2V class (main model used in training/inference)
    configs/                        #   Model configs (WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS)
    modules/                        #   WanModel (DiT), T5EncoderModel, Wan2_2_VAE
    utils/                          #   save_video, masks_like, best_output_size, fuse_lora_to_model
    distributed/                    #   FSDP/SP utilities (optional)
  models/videoalign/                # VideoAlign reward model
  utils/checkpoint.py               # FSDP checkpoint save/load
  utils/fsdp_util.py                # FSDP sharding config

scripts/finetune/
  finetune_wan_2_2_ti2v_grpo.sh                     # GRPO + Gemini
  finetune_wan_2_2_ti2v_nft_put_object_cabinet.sh   # NFT + Gemini (4 GPU)
  finetune_wan_2_2_ti2v_nft_blocks_ranking_rgb.sh   # NFT + SAM3 hallucination (1 GPU default)
  finetune_wan_2_2_ti2v_nft_put_bottles_dustbin.sh  # NFT + SAM3 bottle trajectory (1 GPU default)
  finetune_wan_2_2_ti2v_nft_stack_bowls_three.sh    # NFT + SAM3 bowl stack (1 GPU default)
  finetune_wan_2_2_ti2v_sft.sh                      # SFT with GT videos (1 GPU default)

scripts/tasks/                                        # Ablation experiment scripts (fixed configs)
  nft_{task}_raw.sh                                   # --raw_reward_as_r ablation (no bestofn)
  nft_{task}_bestofn.sh                               # --raw_reward_as_r + --nft_bestofn 1

scripts/inference/
  infer_nft.sh                      # CLI inference
  start_server_nft.sh               # API server
  rollout_videos.py                 # Multi-GPU torchrun rollout for any task
  run_rollout_videos.sh             # Multi-GPU rollout launcher

scripts/eval/
  eval_vidar_put_object_cabinet.sh  # Full eval pipeline (server + client + results)
  test_sapien_env.py                # Standalone SAPIEN environment test (no server)

vidar-robotwin/                     # SAPIEN evaluation environment (integrated subdirectory)
  SETUP.md                          #   Portable environment setup guide
  CLAUDE.md                         #   Codebase documentation
  script/eval_policy.py             #   Main evaluation loop (client)
  run_client.sh                     #   Single-GPU eval launcher
  policy/AR/ar.py                   #   AR policy: caches observations, calls server
  envs/*.py                         #   SAPIEN task environments
  envs/robot/robot.py               #   Robot controller + curobo planner
  envs/_GLOBAL_CONFIGS.py           #   ROOT_PATH auto-detection
  task_config/*.yml                 #   Environment configs (hd_clean, etc.)
  assets/embodiments/*/curobo*.yml  #   CuRobo configs (${VIDAR_ROOT} placeholders)

tools/                              # Standalone analysis, validation & visualization scripts
  detect_hallucination.py           # SAM3 hallucination: constant object count (blocks, standalone CLI)
  detect_hallucination_bottles.py   # SAM3 bottle tracking (monotonic baseline, standalone CLI)
  merge_dit_lora.py                 # Merge LoRA weights into base DiT checkpoint
  run_hallucination_batch_improved.py # Batch blocks hallucination
  run_bottle_hallucination_pipeline.py # Full bottle trajectory pipeline (SAM3 → 2-stage → video)
  extract_bottle_trajectories.py    # SAM3 bottle tracking + trajectory extraction
  detect_hallucination_parall.py    # Parallel-session blocks hallucination variant
  detect_flow_anomalies.py          # Optical flow anomaly detection
  run_flow_anomaly_batch.sh         # Batch optical flow
  test_reward.py                    # Test Gemini API reward on single video
  batch_test_reward.py              # Batch Gemini API scoring
  validate_block_size_reward.py     # Batch validation of BlockSizeRankingRewardScorer
  validate_bowl_stack_reward.py     # Batch validation of BowlStackRewardScorer
  test_flow_aepe_reward.py          # Batch flow-AEPE scoring on rollout videos
  extract_frames.py                 # Extract 6-frame grid PNGs from rollout videos
  rollout_bottles.py                # Single-GPU rollout for bottles task
  visualize_optical_flow.py         # Farneback flow visualization (HSV)
  visualize_anomaly_frames.py       # Export anomaly frames to PNGs + contact sheet

tests/
  test_bottle_hallucination.py      # Test bottle algorithm against ground truth (no GPU)

sam3/                               # SAM3 model (Segment Anything Model 3)

videoalign_ckpt/                    # VideoAlign reward model weights
hps_ckpt/                          # HPS-v2.1 reward model weights
```

## Model Loading

The Wan-2.2 TI2V model is loaded via the vendored `fastvideo/models/wan/` module (originally from the vidar repository, not HuggingFace diffusers). The `fastvideo/models/wan/` code is self-contained within this repo — no external vidar installation is needed. `--ckpt_dir` points to the base model checkpoint directory (`ckpts/Wan2.2-TI2V-5B`), and `--pt_dir` is a pre-merged `.pt` file containing both the base model and prior LoRA weights (`ckpts/vidar_ckpts/vidar_merged_lora.pt`). The optional `--vidar_root` flag is a fallback for importing `wan` from an external vidar repo if the local `fastvideo/models/wan/` directory is absent.

## API Keys & Secrets

API keys are **never** hardcoded in source files. They are read from environment variables at runtime.

**Setup:**
1. Copy the template: `cp .env.example .env`
2. Fill in your real keys in `.env`
3. Before running any training or reward script that uses the Gemini API: `source .env`

**Required variables (for `--reward_backend gpt`):**
| Variable | Description |
|----------|-------------|
| `GPT_API_KEY` | API key for the Gemini/GPT reward endpoint |
| `GPT_API_BASE` | Base URL for the API (default: `http://35.220.164.252:3888/v1/`) |

`.env` is in `.gitignore` and will never be committed. `.env.example` is committed as a reference template.

## Environment & Cluster

### Software
- **Conda env:** `wanx` (Python 3.10, PyTorch 2.6, SAM3, OpenCV, imageio)
- **Key deps:** PyTorch 2.5.0, transformers 4.46.1, diffusers 0.32.0, PEFT 0.13.2, flash-attn 2.7.0

### Cluster (Quest SLURM)

- **Login node:** `quser41` — no GPUs, used for code editing, job submission, and file management
- **GPU nodes:** Allocated dynamically via SLURM, **not guaranteed** — node names change between allocations
- **Accounts:** `p33048` (DanceGRPO), `p33175` (EmbodiedVideoRL)
- **GPU partition:** `gengpu` — has A100 (4×per node on qgpu200x, 2×per node on qgpu0x0x) and H100 (4×per node on qgpu300x)
- **Project dirs:**
  - `/gpfs/projects/p33048/DanceGRPO` — original DanceGRPO training codebase
  - `/gpfs/projects/p33175/EmbodiedVideoRL` — EmbodiedVideoRL with integrated vidar-robotwin eval

**Check current allocations:**
```bash
squeue -u $USER
```

**Request GPU nodes (interactive):**
```bash
# 1× A100 (single GPU tasks: inference, debugging, 1-GPU training, eval)
srun --account=p33175 --partition=gengpu --gres=gpu:a100:1 \
    --time=48:00:00 --mem=128G --cpus-per-task=20 --pty bash

# 4× A100 (multi-GPU training)
srun --account=p33175 --partition=gengpu --gres=gpu:a100:4 \
    --time=48:00:00 --mem=256G --cpus-per-task=32 --pty bash
```

**Run commands on an allocated GPU node:**
```bash
# First, find which node you have:
squeue -u $USER   # Look at NODELIST column, e.g. qgpu2016

# Then SSH to that node to run GPU tasks:
ssh <node> "conda run -n wanx --no-capture-output --cwd /gpfs/projects/p33175/EmbodiedVideoRL \
    <command>"

# Run SAPIEN evaluation:
ssh <node> "conda run -n wanx --no-capture-output --cwd /gpfs/projects/p33175/EmbodiedVideoRL \
    bash scripts/eval/eval_vidar_put_object_cabinet.sh"

# Background training example:
ssh <node> "nohup conda run -n wanx --no-capture-output --cwd /gpfs/projects/p33175/EmbodiedVideoRL \
    bash scripts/finetune/finetune_wan_2_2_ti2v_nft_blocks_ranking_rgb.sh \
    > data/outputs/nft_blocks_ranking_rgb_train.log 2>&1 &"
```

**Important:** All `ssh <node>` commands in this doc and in scripts are examples — replace `qgpu2016` with whatever node `squeue -u $USER` shows as your currently allocated node. If no node is allocated, request one with `srun` first.

**Common `conda run` flags:**
- `--no-capture-output`: Let stdout/stderr pass through (needed for progress bars and real-time logs)
- `--cwd /gpfs/projects/p33175/EmbodiedVideoRL`: Set working directory (critical — `torchrun` needs to find `fastvideo/` relative to cwd)
