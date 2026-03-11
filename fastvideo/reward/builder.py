"""Reward scorer factory and CLI argument helpers."""

import argparse
from pathlib import Path
from typing import Any

from fastvideo.reward.base import NoRewardScorer
from fastvideo.reward.gpt import GPTRewardScorer
from fastvideo.reward.videoalign import VideoAlignScorer
from fastvideo.reward.hallucination import HallucinationRewardScorer
from fastvideo.reward.hallucination_bottles import BottleHallucinationRewardScorer
from fastvideo.reward.hallucination_bowls import BowlStackRewardScorer
from fastvideo.reward.hallucination_blocks_size import BlockSizeRankingRewardScorer
from fastvideo.reward.flow_aepe import FlowAEPERewardScorer


def build_reward_scorer(args, device):
    """Build the appropriate RewardScorer from parsed CLI args."""
    if args.reward_backend == "none":
        return NoRewardScorer()
    if args.reward_backend == "gpt":
        return GPTRewardScorer(
            api_base=args.gpt_api_base,
            api_key=args.gpt_api_key,
            model=args.gpt_model,
            temperature=args.gpt_temperature,
        )
    if args.reward_backend == "videoalign":
        return VideoAlignScorer(
            device=device, ckpt_dir=args.videoalign_ckpt_dir,
            score_key=args.videoalign_score_key,
            use_norm=args.videoalign_use_norm,
        )
    if args.reward_backend == "hallucination":
        prompts = args.hallucination_prompts
        if args.hallucination_expected_counts is not None:
            if len(args.hallucination_expected_counts) != len(prompts):
                raise ValueError(
                    f"--hallucination_expected_counts length ({len(args.hallucination_expected_counts)}) "
                    f"must match --hallucination_prompts length ({len(prompts)})"
                )
            expected_counts = dict(zip(prompts, args.hallucination_expected_counts))
        else:
            expected_counts = {p: 1 for p in prompts}
        return HallucinationRewardScorer(
            prompts=prompts,
            expected_counts=expected_counts,
            crop_top_ratio=args.hallucination_crop_top_ratio,
            occlusion_gap_max=args.occlusion_gap_max,
            occlusion_pos_thr=args.occlusion_pos_thr,
            duplication_spike_max=args.duplication_spike_max,
            device_id=device.index if device.index is not None else 0,
        )
    if args.reward_backend == "hallucination_bottles":
        return BottleHallucinationRewardScorer(
            prompt=args.bottle_hall_prompt,
            crop_top_ratio=args.hallucination_crop_top_ratio,
            cx_cutoff=args.bottle_cx_cutoff,
            spike_max=args.bottle_spike_max,
            filter_max_gap=args.bottle_filter_max_gap,
            device_id=device.index if device.index is not None else 0,
        )
    if args.reward_backend == "hallucination_bowls":
        return BowlStackRewardScorer(
            prompt=args.bowl_stack_prompt,
            initial_count=args.bowl_initial_count,
            crop_top_ratio=args.hallucination_crop_top_ratio,
            convergence_thr=args.bowl_convergence_thr,
            check_window_frac=args.bowl_check_window_frac,
            gap_max=args.bowl_gap_max,
            reappear_max=args.bowl_reappear_max,
            reappear_pos_thr=args.bowl_reappear_pos_thr,
            device_id=device.index if device.index is not None else 0,
        )
    if args.reward_backend == "hallucination_blocks_size":
        return BlockSizeRankingRewardScorer(
            prompt=args.block_size_prompt,
            expected_count=args.block_size_expected_count,
            crop_top_ratio=args.hallucination_crop_top_ratio,
            check_window_frac=args.block_size_check_window_frac,
            min_visible_frac=args.block_size_min_visible_frac,
            ordering_margin=args.block_size_ordering_margin,
            center_margin=args.block_size_center_margin,
            gap_max=args.block_size_gap_max,
            occlusion_gap_max=args.block_size_occlusion_gap_max,
            occlusion_pos_thr=args.block_size_occlusion_pos_thr,
            device_id=device.index if device.index is not None else 0,
        )
    if args.reward_backend == "flow_aepe":
        return FlowAEPERewardScorer(
            cfg_path=args.flow_aepe_cfg,
            ckpt_path=args.flow_aepe_ckpt,
            epe_threshold=args.flow_aepe_threshold,
            crop_top_ratio=args.hallucination_crop_top_ratio,
            frame_step=args.flow_aepe_frame_step,
            device_id=device.index if device.index is not None else 0,
        )
    raise ValueError(f"Unsupported reward_backend: {args.reward_backend}")


def _str2bool(v):
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in ("yes", "true", "t", "y", "1"):
        return True
    if val in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def add_reward_args(p: argparse.ArgumentParser) -> None:
    """Add all reward-related CLI arguments to an argument parser."""
    # Reward backend selection
    p.add_argument("--reward_backend", type=str, default="videoalign",
                    choices=["videoalign", "gpt", "hallucination", "hallucination_bottles",
                             "hallucination_bowls", "hallucination_blocks_size",
                             "flow_aepe", "none"])

    # GPT / Gemini
    p.add_argument("--gpt_api_base", type=str,
                    default="http://35.220.164.252:3888/v1/")
    p.add_argument("--gpt_api_key", type=str,
                    default=None,
                    help="API key for GPT/Gemini reward. Defaults to $GPT_API_KEY env var.")
    p.add_argument("--gpt_model", type=str, default="gpt-4o")
    p.add_argument("--gpt_temperature", type=float, default=0.2)

    # VideoAlign
    p.add_argument("--videoalign_ckpt_dir", type=str, default="./videoalign_ckpt")
    p.add_argument("--videoalign_score_key", type=str, default="overall",
                    choices=["vq", "mq", "ta", "overall"])
    p.add_argument("--videoalign_use_norm", type=_str2bool, default=True)

    # Hallucination reward (SAM3)
    p.add_argument("--hallucination_prompts", type=str, nargs="+",
                    default=["red block", "green block", "blue block"],
                    help="Object text prompts for SAM3 hallucination tracking")
    p.add_argument("--hallucination_expected_counts", type=int, nargs="+",
                    default=None,
                    help="Expected count for each prompt (same order). Default: 1 each.")
    p.add_argument("--hallucination_crop_top_ratio", type=float, default=2/3,
                    help="Fraction of frame height to keep from top (crops wrist cameras)")
    p.add_argument("--occlusion_gap_max", type=int, default=5,
                    help="Max consecutive absent frames to suppress as occlusion")
    p.add_argument("--occlusion_pos_thr", type=float, default=0.15,
                    help="Max normalised L-inf centre shift for occlusion suppression")
    p.add_argument("--duplication_spike_max", type=int, default=0,
                    help="Max consecutive frames of count > expected to suppress as SAM3 noise (0=disabled)")

    # Bottle hallucination reward (SAM3 trajectory-based)
    p.add_argument("--bottle_hall_prompt", type=str, default="bottle",
                    help="SAM3 text prompt for bottle tracking (default: 'bottle')")
    p.add_argument("--bottle_cx_cutoff", type=float, default=0.26,
                    help="Truncate trajectory when cx < this (bottle reached dustbin)")
    p.add_argument("--bottle_spike_max", type=int, default=3,
                    help="Max frames for brief count-spike suppression in monotonic check")
    p.add_argument("--bottle_filter_max_gap", type=int, default=5,
                    help="Split trajectory at gaps > this many frames (SAM3 error filtering)")

    # Bowl stacking reward (SAM3 convergence-based)
    p.add_argument("--bowl_stack_prompt", type=str, default="bowl",
                    help="SAM3 text prompt for bowl tracking")
    p.add_argument("--bowl_initial_count", type=int, default=3,
                    help="Expected number of bowls at start")
    p.add_argument("--bowl_convergence_thr", type=float, default=0.30,
                    help="Max pairwise L-inf distance for convergence check")
    p.add_argument("--bowl_check_window_frac", type=float, default=0.20,
                    help="Fraction of final frames for convergence check")
    p.add_argument("--bowl_gap_max", type=int, default=5,
                    help="Max frames for count-spike suppression")
    p.add_argument("--bowl_reappear_max", type=int, default=10,
                    help="Max disappearance gap before hallucination")
    p.add_argument("--bowl_reappear_pos_thr", type=float, default=0.15,
                    help="Position shift threshold for reappearance suppression")

    # Block size ranking reward (SAM3 + spatial ordering)
    p.add_argument("--block_size_prompt", type=str, default="block",
                    help="SAM3 text prompt for block tracking")
    p.add_argument("--block_size_expected_count", type=int, default=3,
                    help="Expected number of blocks")
    p.add_argument("--block_size_check_window_frac", type=float, default=0.20,
                    help="Fraction of final frames for ordering check")
    p.add_argument("--block_size_min_visible_frac", type=float, default=0.50,
                    help="Min fraction of check window with all blocks visible")
    p.add_argument("--block_size_ordering_margin", type=float, default=0.03,
                    help="Min cx difference between adjacent blocks for ordering")
    p.add_argument("--block_size_center_margin", type=float, default=0.15,
                    help="Max deviation of mean cx from 0.5 for center check")
    p.add_argument("--block_size_gap_max", type=int, default=5,
                    help="Max frames for hallucination spike suppression")
    p.add_argument("--block_size_occlusion_gap_max", type=int, default=5,
                    help="Max frames for occlusion suppression")
    p.add_argument("--block_size_occlusion_pos_thr", type=float, default=0.15,
                    help="Position threshold for occlusion suppression")

    # Debug video rendering (shared across all SAM3-based backends)
    p.add_argument("--skip_reward_debug_video", type=_str2bool, default=False,
                    help="Skip rendering annotated debug videos during reward scoring. "
                         "Saves ~10-15s per video per step. Reward values are unaffected.")

    # Flow AEPE reward (SEA-RAFT forward-backward consistency)
    p.add_argument("--flow_aepe_cfg", type=str,
                    default=str(Path("/gpfs/projects/p33048/WorldArena/video_quality/WorldArena"
                                     "/third_party/SEA-RAFT/config/eval/spring-M.json")),
                    help="SEA-RAFT config JSON path")
    p.add_argument("--flow_aepe_ckpt", type=str,
                    default=str(Path("/gpfs/projects/p33048/WorldArena/video_quality/WorldArena"
                                     "/third_party/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth")),
                    help="SEA-RAFT checkpoint path")
    p.add_argument("--flow_aepe_threshold", type=float, default=0.5,
                    help="Flow score threshold for binary reward (score >= thr → reward=1)")
    p.add_argument("--flow_aepe_frame_step", type=int, default=1,
                    help="Subsample frames (1=every frame, 2=every other, etc.)")
