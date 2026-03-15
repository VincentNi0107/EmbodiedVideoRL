#!/usr/bin/env python3
"""Collect training data for Goal-Conditioned IDM.

Two-phase pipeline:
  Phase 1 (SAPIEN, conda env: RoboTwin-hb):
    Run expert demos, save observations + actions per timestep.

  Phase 2 (Wan2.2, conda env: wanx):
    Generate predicted videos from first-frame observations, pair with demos.

This script handles Phase 2 — it reads expert trajectories (observations saved
as images + actions saved as JSON), generates Wan2.2 videos, and writes the
final paired HDF5 dataset.

Phase 1 can be done with the existing vidar-robotwin collect_data.py, or
by running this script's ``--collect-from-sapien`` mode (requires RoboTwin env).

Usage::

    # Phase 2: generate goal frames and build HDF5
    python tools/collect_gc_idm_data.py \
        --expert_dir data/gc_idm_raw/put_object_cabinet \
        --output_h5 data/gc_idm_train/put_object_cabinet.h5 \
        --ckpt_dir ckpts/Wan2.2-TI2V-5B \
        --pt_dir ckpts/vidar_ckpts/vidar_merged_lora.pt

Each episode in expert_dir should have:
    episode_NNN/
        observations/     # 000.png, 001.png, ... (736×640 composites)
        actions.json      # [[14 floats], [14 floats], ...]
        prompt.txt        # text prompt for video generation
"""

import argparse
import json
import logging
import os

import cv2
import h5py
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_expert_episode(ep_dir):
    """Load observations and actions from an expert episode directory."""
    obs_dir = os.path.join(ep_dir, "observations")
    actions_path = os.path.join(ep_dir, "actions.json")
    prompt_path = os.path.join(ep_dir, "prompt.txt")

    # Load observations as (T, 3, H, W) uint8 array
    obs_files = sorted(
        [f for f in os.listdir(obs_dir) if f.endswith(".png") or f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    observations = []
    for f in obs_files:
        img = cv2.imread(os.path.join(obs_dir, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        observations.append(img.transpose(2, 0, 1))  # (3, H, W)
    observations = np.stack(observations)  # (T, 3, H, W)

    # Load actions
    with open(actions_path) as f:
        actions = np.array(json.load(f), dtype=np.float32)  # (T, 14)

    # Load prompt
    prompt = ""
    if os.path.exists(prompt_path):
        with open(prompt_path) as f:
            prompt = f.read().strip()

    return observations, actions, prompt


def generate_goal_frames(
    first_obs,
    prompt,
    num_frames,
    ckpt_dir,
    pt_dir,
    nft_lora_path=None,
    lora_alpha=1.0,
    device_id=0,
    seed=42,
):
    """Generate a video from the first observation frame using Wan2.2.

    Returns:
        goal_frames: (T, 3, H, W) uint8 numpy array
    """
    from PIL import Image
    from fastvideo.infer_nft import load_model

    model, cfg, wan_mod, save_fn, SIZE_CONFIGS, MAX_AREA_CONFIGS = load_model(
        ckpt_dir=ckpt_dir,
        pt_dir=pt_dir,
        nft_lora_path=nft_lora_path,
        lora_alpha=lora_alpha,
        device_id=device_id,
        t5_cpu=False,
        offload_model=True,
    )

    size_key = "640*736"
    # first_obs is (3, H, W) uint8 → PIL Image
    img = Image.fromarray(first_obs.transpose(1, 2, 0))
    img = img.resize(SIZE_CONFIGS[size_key])

    video_tensor = model.generate(
        input_prompt=prompt,
        img=img,
        size=SIZE_CONFIGS[size_key],
        max_area=MAX_AREA_CONFIGS[size_key],
        frame_num=num_frames,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=20,
        guide_scale=5.0,
        seed=seed,
        offload_model=True,
    )

    # (C, T, H, W) in [-1, 1] → (T, C, H, W) uint8
    video = video_tensor.clamp(-1, 1).add(1).mul(127.5).to(torch.uint8)
    video = video.permute(1, 0, 2, 3).cpu().numpy()  # (T, C, H, W)
    return video


def build_h5(expert_dir, output_h5, ckpt_dir, pt_dir, nft_lora_path, lora_alpha,
             device_id, max_episodes, seed):
    """Build the HDF5 training dataset from expert episodes + generated videos."""
    ep_dirs = sorted([
        d for d in os.listdir(expert_dir)
        if os.path.isdir(os.path.join(expert_dir, d)) and d.startswith("episode_")
    ])
    if max_episodes > 0:
        ep_dirs = ep_dirs[:max_episodes]

    os.makedirs(os.path.dirname(output_h5), exist_ok=True)

    with h5py.File(output_h5, "w") as hf:
        for i, ep_name in enumerate(ep_dirs):
            ep_dir = os.path.join(expert_dir, ep_name)
            logger.info(f"Processing {ep_name} ({i+1}/{len(ep_dirs)})")

            observations, actions, prompt = load_expert_episode(ep_dir)
            T = observations.shape[0]

            # Generate goal frames (121 frames to match observations)
            num_frames = min(121, T)
            # Frame count must be 4n+1 for Wan2.2
            num_frames = ((num_frames - 1) // 4) * 4 + 1

            goal_frames = generate_goal_frames(
                first_obs=observations[0],
                prompt=prompt,
                num_frames=num_frames,
                ckpt_dir=ckpt_dir,
                pt_dir=pt_dir,
                nft_lora_path=nft_lora_path,
                lora_alpha=lora_alpha,
                device_id=device_id,
                seed=seed,
            )

            # Align lengths
            n = min(T, goal_frames.shape[0])
            observations = observations[:n]
            actions = actions[:n]
            goal_frames = goal_frames[:n]

            # Write to HDF5
            grp = hf.create_group(ep_name)
            grp.create_dataset("observations", data=observations, compression="gzip")
            grp.create_dataset("goal_frames", data=goal_frames, compression="gzip")
            grp.create_dataset("actions", data=actions)

            logger.info(f"  {ep_name}: {n} timesteps, obs={observations.shape}, "
                        f"goals={goal_frames.shape}, actions={actions.shape}")

    logger.info(f"Dataset saved to {output_h5}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect GC-IDM training data (Phase 2: generate goal frames)")
    parser.add_argument("--expert_dir", type=str, required=True,
                        help="Directory containing expert episodes")
    parser.add_argument("--output_h5", type=str, required=True,
                        help="Output HDF5 dataset path")
    parser.add_argument("--ckpt_dir", type=str, default="ckpts/Wan2.2-TI2V-5B")
    parser.add_argument("--pt_dir", type=str, default="ckpts/vidar_ckpts/vidar_merged_lora.pt")
    parser.add_argument("--nft_lora_path", type=str, default="")
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="Max episodes to process (0 = all)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    build_h5(
        expert_dir=args.expert_dir,
        output_h5=args.output_h5,
        ckpt_dir=args.ckpt_dir,
        pt_dir=args.pt_dir,
        nft_lora_path=args.nft_lora_path if args.nft_lora_path else None,
        lora_alpha=args.lora_alpha,
        device_id=args.device,
        max_episodes=args.max_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
