"""Reward scorers for video generation training."""

from fastvideo.reward.base import (
    RewardScorer,
    NoRewardScorer,
    save_reward_curve,
    video_first_frame_pil,
)
from fastvideo.reward.gpt import GPTRewardScorer
from fastvideo.reward.videoalign import VideoAlignScorer
from fastvideo.reward.hallucination import HallucinationRewardScorer
from fastvideo.reward.hallucination_bottles import BottleHallucinationRewardScorer
from fastvideo.reward.hallucination_bowls import BowlStackRewardScorer
from fastvideo.reward.hallucination_blocks_size import BlockSizeRankingRewardScorer
from fastvideo.reward.flow_aepe import FlowAEPERewardScorer
from fastvideo.reward.builder import build_reward_scorer, add_reward_args

__all__ = [
    "RewardScorer",
    "NoRewardScorer",
    "GPTRewardScorer",
    "VideoAlignScorer",
    "HallucinationRewardScorer",
    "BottleHallucinationRewardScorer",
    "BowlStackRewardScorer",
    "BlockSizeRankingRewardScorer",
    "FlowAEPERewardScorer",
    "build_reward_scorer",
    "add_reward_args",
    "save_reward_curve",
    "video_first_frame_pil",
]
