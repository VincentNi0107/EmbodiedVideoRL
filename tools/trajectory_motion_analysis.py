"""Batch trajectory motion analysis for stack_bowls_three videos.

Uses SAM3 to track "bowl" across all video frames, then computes per-object
motion metrics (trajectory length, average speed, max speed, speed jump) to
quantify motion intensity and detect trajectory anomalies.

Usage (on a GPU node):
    conda run -n wanx python tools/trajectory_motion_analysis.py \
        --video-root data/outputs/nft_stack_bowls_three/ng4_s42_tl0.0_kl0.001/videos \
        --reward-debug-root data/outputs/nft_stack_bowls_three/ng4_s42_tl0.0_kl0.001/reward_debug \
        --out-dir data/outputs/nft_stack_bowls_three/ng4_s42_tl0.0_kl0.001/trajectory_analysis
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from glob import glob
from pathlib import Path

import cv2
import numpy as np


def extract_trajectories(predictor, video_path: str, crop_top_ratio: float = 2/3,
                         prompt: str = "bowl"):
    """Run SAM3 tracking and return per-object trajectories.

    Returns:
        objects: {obj_id_str: [{frame, cx, cy, w, h, prob}, ...]}
        total_frames: int
        fps: float
    """
    from fastvideo.reward.sam3_utils import extract_frames_to_jpeg, track_prompt

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    crop_h = None
    if crop_top_ratio < 1.0:
        crop_h = (int(h_full * crop_top_ratio)) // 16 * 16

    jpeg_dir = extract_frames_to_jpeg(video_path, crop_h=crop_h)

    try:
        frame_results = track_prompt(predictor, jpeg_dir, prompt)
    finally:
        shutil.rmtree(jpeg_dir, ignore_errors=True)

    # Build per-object trajectories
    objects = {}
    for fi, fr in enumerate(frame_results):
        for i, obj_id in enumerate(fr["obj_ids"]):
            oid = str(obj_id)
            if oid not in objects:
                objects[oid] = []
            bx, by, bw, bh = fr["boxes_xywh"][i]
            objects[oid].append({
                "frame": fi,
                "cx": float(bx + bw / 2),
                "cy": float(by + bh / 2),
                "w": float(bw),
                "h": float(bh),
                "prob": float(fr["probs"][i]),
            })

    return objects, total_frames, fps


def compute_motion_metrics(traj):
    """Compute motion metrics for a single object trajectory.

    Args:
        traj: list of dicts with keys: frame, cx, cy

    Returns:
        dict with: traj_length, avg_speed, max_speed, max_speed_jump, num_points,
                   speeds (list), speed_jumps (list)
    """
    if len(traj) < 2:
        return {
            "traj_length": 0.0,
            "avg_speed": 0.0,
            "max_speed": 0.0,
            "max_speed_jump": 0.0,
            "num_points": len(traj),
            "speeds": [],
            "speed_jumps": [],
        }

    # Compute per-step velocity vectors and speeds
    velocities = []  # (vx, vy) per step
    speeds = []
    total_length = 0.0
    for i in range(1, len(traj)):
        dx = traj[i]["cx"] - traj[i - 1]["cx"]
        dy = traj[i]["cy"] - traj[i - 1]["cy"]
        dist = math.sqrt(dx * dx + dy * dy)
        # Frame gap (SAM3 may skip frames)
        frame_gap = max(traj[i]["frame"] - traj[i - 1]["frame"], 1)
        vx = dx / frame_gap
        vy = dy / frame_gap
        velocities.append((vx, vy))
        speeds.append(dist / frame_gap)
        total_length += dist

    # Compute speed jumps as vector difference norm (captures direction changes)
    speed_jumps = []
    for i in range(1, len(velocities)):
        dvx = velocities[i][0] - velocities[i - 1][0]
        dvy = velocities[i][1] - velocities[i - 1][1]
        speed_jumps.append(math.sqrt(dvx * dvx + dvy * dvy))

    max_speed = max(speeds) if speeds else 0.0
    avg_speed = total_length / (traj[-1]["frame"] - traj[0]["frame"]) if traj[-1]["frame"] != traj[0]["frame"] else 0.0
    max_speed_jump = max(speed_jumps) if speed_jumps else 0.0

    return {
        "traj_length": total_length,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "max_speed_jump": max_speed_jump,
        "num_points": len(traj),
        "speeds": speeds,
        "speed_jumps": speed_jumps,
    }


def lookup_reward_label(reward_debug_root: str, step_dir: str, video_stem: str) -> str:
    """Look up CLEAN/FAIL label from reward_debug directory."""
    debug_step_dir = os.path.join(reward_debug_root, step_dir)
    if not os.path.isdir(debug_step_dir):
        return "UNKNOWN"

    for f in os.listdir(debug_step_dir):
        if f.startswith(video_stem) and f.endswith("_CLEAN.mp4"):
            return "CLEAN"
        if f.startswith(video_stem) and f.endswith("_FAIL.mp4"):
            return "FAIL"
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(description="Trajectory motion analysis for stack_bowls_three")
    parser.add_argument("--video-root", type=str, required=True,
                        help="Root directory containing step*/video.mp4 folders")
    parser.add_argument("--reward-debug-root", type=str, required=True,
                        help="Root directory containing reward_debug step* folders with CLEAN/FAIL labels")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--prompt", type=str, default="bowl",
                        help="SAM3 text prompt for tracking")
    parser.add_argument("--crop-top-ratio", type=float, default=2/3,
                        help="Fraction of frame height to keep from top")
    parser.add_argument("--pattern", type=str, default="*",
                        help="Glob pattern for step directories (e.g. 'step0001*')")
    parser.add_argument("--max-videos", type=int, default=0,
                        help="Max videos to process (0=all)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Discover all videos
    step_dirs = sorted(glob(os.path.join(args.video_root, args.pattern)))
    step_dirs = [d for d in step_dirs if os.path.isdir(d)]

    all_videos = []
    for sd in step_dirs:
        step_name = os.path.basename(sd)
        for mp4 in sorted(glob(os.path.join(sd, "*.mp4"))):
            all_videos.append((step_name, mp4))

    if args.max_videos > 0:
        all_videos = all_videos[:args.max_videos]

    print(f"Found {len(all_videos)} videos to process")

    # Build SAM3 predictor
    print("Loading SAM3 predictor...")
    import torch
    from fastvideo.reward.base import build_sam3_predictor
    predictor = build_sam3_predictor(device_id=0)
    print("SAM3 ready.")

    results = []
    for idx, (step_name, video_path) in enumerate(all_videos):
        video_stem = Path(video_path).stem
        label = lookup_reward_label(args.reward_debug_root, step_name, video_stem)

        t0 = time.time()
        print(f"\n[{idx+1}/{len(all_videos)}] {step_name}/{video_stem} ({label})")

        try:
            objects, total_frames, fps = extract_trajectories(
                predictor, video_path,
                crop_top_ratio=args.crop_top_ratio,
                prompt=args.prompt,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "step": step_name,
                "video_stem": video_stem,
                "label": label,
                "error": str(e),
            })
            continue

        elapsed = time.time() - t0
        print(f"  SAM3 tracking: {elapsed:.1f}s, {len(objects)} objects, {total_frames} frames")

        # Filter real objects (first appearance <= frame 5)
        real_objects = {oid: traj for oid, traj in objects.items()
                        if traj and traj[0]["frame"] <= 5}

        # Compute per-object metrics
        obj_metrics = {}
        for oid, traj in real_objects.items():
            obj_metrics[oid] = compute_motion_metrics(traj)

        # Aggregate metrics across all real objects
        if obj_metrics:
            all_traj_lengths = [m["traj_length"] for m in obj_metrics.values()]
            all_avg_speeds = [m["avg_speed"] for m in obj_metrics.values()]
            all_max_speeds = [m["max_speed"] for m in obj_metrics.values()]
            all_max_jumps = [m["max_speed_jump"] for m in obj_metrics.values()]

            agg = {
                "total_traj_length": sum(all_traj_lengths),
                "mean_traj_length": np.mean(all_traj_lengths),
                "mean_avg_speed": np.mean(all_avg_speeds),
                "max_max_speed": max(all_max_speeds),
                "mean_max_speed": np.mean(all_max_speeds),
                "max_speed_jump": max(all_max_jumps),
                "mean_speed_jump": np.mean(all_max_jumps),
            }
        else:
            agg = {
                "total_traj_length": 0.0,
                "mean_traj_length": 0.0,
                "mean_avg_speed": 0.0,
                "max_max_speed": 0.0,
                "mean_max_speed": 0.0,
                "max_speed_jump": 0.0,
                "mean_speed_jump": 0.0,
            }

        # Round for display
        for k in agg:
            agg[k] = round(float(agg[k]), 4)

        print(f"  Objects: {len(real_objects)} real, {len(objects) - len(real_objects)} spurious")
        print(f"  TrajLen={agg['total_traj_length']:.3f}  AvgSpd={agg['mean_avg_speed']:.4f}  "
              f"MaxSpd={agg['max_max_speed']:.4f}  MaxJump={agg['max_speed_jump']:.4f}")

        # Build filename with metrics
        # Format: {video_stem}_{label}_tl{total_traj_length}_as{avg_speed}_ms{max_speed}_mj{max_jump}.json
        safe_tl = f"{agg['total_traj_length']:.3f}"
        safe_as = f"{agg['mean_avg_speed']:.4f}"
        safe_ms = f"{agg['max_max_speed']:.4f}"
        safe_mj = f"{agg['max_speed_jump']:.4f}"
        out_name = f"{video_stem}_{label}_tl{safe_tl}_as{safe_as}_ms{safe_ms}_mj{safe_mj}.json"
        out_path = os.path.join(args.out_dir, step_name, out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Save per-video detail
        detail = {
            "video_path": video_path,
            "video_stem": video_stem,
            "step": step_name,
            "label": label,
            "total_frames": total_frames,
            "fps": fps,
            "num_real_objects": len(real_objects),
            "num_spurious_objects": len(objects) - len(real_objects),
            "aggregate_metrics": agg,
            "per_object_metrics": {
                oid: {k: v for k, v in m.items() if k not in ("speeds", "speed_jumps")}
                for oid, m in obj_metrics.items()
            },
            "per_object_trajectories": {
                oid: traj for oid, traj in real_objects.items()
            },
        }
        with open(out_path, "w") as f:
            json.dump(detail, f, indent=2)

        # Collect for summary
        results.append({
            "step": step_name,
            "video_stem": video_stem,
            "label": label,
            **agg,
        })

    # Save summary
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary statistics by CLEAN vs FAIL
    clean = [r for r in results if r.get("label") == "CLEAN" and "error" not in r]
    fail = [r for r in results if r.get("label") == "FAIL" and "error" not in r]

    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(results)} videos ({len(clean)} CLEAN, {len(fail)} FAIL)")
    print(f"{'='*70}")

    for group_name, group in [("CLEAN", clean), ("FAIL", fail)]:
        if not group:
            continue
        tls = [r["total_traj_length"] for r in group]
        avs = [r["mean_avg_speed"] for r in group]
        mss = [r["max_max_speed"] for r in group]
        mjs = [r["max_speed_jump"] for r in group]
        print(f"\n{group_name} (n={len(group)}):")
        print(f"  TotalTrajLen:  mean={np.mean(tls):.4f}  std={np.std(tls):.4f}  "
              f"min={np.min(tls):.4f}  max={np.max(tls):.4f}")
        print(f"  MeanAvgSpeed:  mean={np.mean(avs):.5f}  std={np.std(avs):.5f}  "
              f"min={np.min(avs):.5f}  max={np.max(avs):.5f}")
        print(f"  MaxSpeed:      mean={np.mean(mss):.5f}  std={np.std(mss):.5f}  "
              f"min={np.min(mss):.5f}  max={np.max(mss):.5f}")
        print(f"  MaxSpeedJump:  mean={np.mean(mjs):.5f}  std={np.std(mjs):.5f}  "
              f"min={np.min(mjs):.5f}  max={np.max(mjs):.5f}")

    # Also save a ranked list by max_speed_jump (突变最严重的排前面)
    ranked = sorted(
        [r for r in results if "error" not in r],
        key=lambda r: r["max_speed_jump"],
        reverse=True,
    )
    ranked_path = os.path.join(args.out_dir, "ranked_by_speed_jump.json")
    with open(ranked_path, "w") as f:
        json.dump(ranked, f, indent=2)

    # And ranked by max_max_speed
    ranked_speed = sorted(
        [r for r in results if "error" not in r],
        key=lambda r: r["max_max_speed"],
        reverse=True,
    )
    ranked_speed_path = os.path.join(args.out_dir, "ranked_by_max_speed.json")
    with open(ranked_speed_path, "w") as f:
        json.dump(ranked_speed, f, indent=2)

    print(f"\nResults saved to: {args.out_dir}")
    print(f"  summary.json, ranked_by_speed_jump.json, ranked_by_max_speed.json")
    print(f"  + per-video JSON files in step*/ subdirs")

    predictor.shutdown()


if __name__ == "__main__":
    main()
