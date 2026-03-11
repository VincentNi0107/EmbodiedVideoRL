"""Base reward scorer interface and utility functions."""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from PIL import Image


class RewardScorer:
    """Abstract base class for all reward scorers."""

    def score(
        self, prompt: str, first_frame: Image.Image,
        video_path: Optional[str] = None,
        frames_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError


class NoRewardScorer(RewardScorer):
    def score(self, prompt, first_frame, video_path=None, frames_dir=None):
        return {"reward": 0.0}


def build_sam3_predictor(device_id: int = 0):
    """Build a Sam3VideoPredictor safe for use inside DDP (torchrun).

    SAM3's base class ``Sam3VideoBase.__init__`` reads ``RANK`` and
    ``WORLD_SIZE`` from env-vars.  Under ``torchrun`` these are set by DDP,
    causing SAM3 to enter multi-GPU mode and issue gloo broadcast /
    all_gather calls that conflict with the training process group.

    This helper temporarily masks those env-vars so SAM3 always initialises
    in single-GPU mode (``rank=0, world_size=1``).
    """
    prev_device = torch.cuda.current_device()
    torch.cuda.set_device(device_id)

    # Temporarily mask DDP env vars so SAM3 sees single-GPU mode
    env_keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    saved = {k: os.environ.pop(k, None) for k in env_keys}
    try:
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor
        predictor = Sam3VideoPredictor()
    finally:
        # Restore env vars for DDP
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        torch.cuda.set_device(prev_device)
    return predictor


def video_first_frame_pil(video_tensor: torch.Tensor) -> Image.Image:
    """Extract the first frame from a video tensor as a PIL Image."""
    frame = video_tensor[:, 0].detach().float().clamp(-1, 1)
    frame = ((frame + 1.0) * 127.5).to(torch.uint8).cpu()
    return Image.fromarray(frame.permute(1, 2, 0).numpy())


def _moving_average(values: List[float], window: int) -> np.ndarray:
    """Compute centred moving average with edge padding."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) <= window:
        return np.full_like(arr, arr.mean())
    kernel = np.ones(window) / window
    # pad edges to keep same length
    pad = window // 2
    padded = np.concatenate([np.full(pad, arr[0]), arr, np.full(pad, arr[-1])])
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def save_reward_curve(
    steps: List[int],
    mean_rewards: List[float],
    save_path: str,
):
    """Plot and save the mean reward curve over training steps."""
    # Deduplicate: if a step appears multiple times (from resume), keep the last entry
    seen: Dict[int, float] = {}
    for s, r in zip(steps, mean_rewards):
        seen[s] = r
    steps_dedup = sorted(seen.keys())
    rewards_dedup = [seen[s] for s in steps_dedup]

    fig, ax = plt.subplots(figsize=(10, 5))
    # Raw data (semi-transparent)
    ax.plot(steps_dedup, rewards_dedup, marker="o", markersize=3, linewidth=0.8,
            color="#2196F3", alpha=0.35, label="per-step reward")
    # Smoothed moving average
    window = max(10, len(rewards_dedup) // 10)
    smoothed = _moving_average(rewards_dedup, window)
    ax.plot(steps_dedup, smoothed, linewidth=2.5, color="#F44336",
            label=f"moving avg (w={window})")
    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Mean Reward", fontsize=13)
    ax.set_title("NFT Training — Mean Reward per Step", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    y_lo = min(0.0, min(mean_rewards) - 0.05)
    y_hi = max(1.0, max(mean_rewards) + 0.05)
    ax.set_ylim(y_lo, y_hi)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
