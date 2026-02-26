"""VideoAlign reward scorer."""

from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image

from fastvideo.reward.base import RewardScorer


class VideoAlignScorer(RewardScorer):
    _KEY_MAP = {"vq": "VQ", "mq": "MQ", "ta": "TA", "overall": "Overall"}

    def __init__(self, device, ckpt_dir, score_key="overall", use_norm=True):
        from fastvideo.models.videoalign.inference import VideoVLMRewardInference
        self._score_key = score_key.lower()
        if self._score_key not in self._KEY_MAP:
            raise ValueError(f"Unsupported score_key: {score_key}")
        self._inferencer = VideoVLMRewardInference(
            ckpt_dir, device=f"cuda:{device.index}", dtype=torch.bfloat16,
        )
        self._use_norm = use_norm

    @torch.no_grad()
    def score(self, prompt, first_frame, video_path=None):
        if video_path is None:
            raise ValueError("video_path required for VideoAlign")
        rw = self._inferencer.reward(
            [str(Path(video_path).resolve())], [prompt],
            use_norm=self._use_norm,
        )
        r = rw[0]
        sel_key = self._KEY_MAP[self._score_key]
        return {
            "reward": float(r[sel_key]),
            "VQ": float(r["VQ"]), "MQ": float(r["MQ"]),
            "TA": float(r["TA"]), "Overall": float(r["Overall"]),
        }
