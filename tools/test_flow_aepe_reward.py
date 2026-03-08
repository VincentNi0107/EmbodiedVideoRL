#!/usr/bin/env python
"""Test flow-AEPE reward on blocks_ranking_rgb rollout videos.

Usage (GPU recommended):
    python tools/test_flow_aepe_reward.py \
        --input-root data/outputs/rollout_robotwin_121 \
        --out-dir    data/outputs/flow_aepe_blocks_ranking_rgb \
        --pattern    "robotwin_blocks_ranking_rgb_*"

    # Faster: subsample frames
    python tools/test_flow_aepe_reward.py --frame-step 2

    # CPU (slow but works):
    CUDA_VISIBLE_DEVICES="" python tools/test_flow_aepe_reward.py
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastvideo.reward.flow_aepe import FlowAEPERewardScorer


def main():
    parser = argparse.ArgumentParser(description="Test flow-AEPE reward on rollout videos")
    parser.add_argument("--input-root", type=str,
                        default="data/outputs/rollout_robotwin_121",
                        help="Root directory containing scene folders")
    parser.add_argument("--out-dir", type=str,
                        default="data/outputs/flow_aepe_blocks_ranking_rgb",
                        help="Output directory for results")
    parser.add_argument("--pattern", type=str,
                        default="robotwin_blocks_ranking_rgb_*",
                        help="Glob pattern to match scene directories")
    parser.add_argument("--crop-top-ratio", type=float, default=2/3,
                        help="Crop top ratio (2/3 removes wrist cameras)")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Subsample frames (1=all, 2=every other, etc.)")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device ID (ignored if CUDA unavailable)")
    parser.add_argument("--cfg", type=str, default=None,
                        help="SEA-RAFT config path (default: auto)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="SEA-RAFT checkpoint path (default: auto)")
    args = parser.parse_args()

    # Resolve paths
    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find scene directories
    scene_dirs = sorted(glob.glob(str(input_root / args.pattern)))
    if not scene_dirs:
        print(f"No scene directories found matching {input_root / args.pattern}")
        sys.exit(1)

    # Collect all video files
    videos = []
    for sd in scene_dirs:
        scene_name = os.path.basename(sd)
        for vf in sorted(glob.glob(os.path.join(sd, "*.mp4"))):
            videos.append((scene_name, vf))

    print(f"Found {len(videos)} videos across {len(scene_dirs)} scenes")
    print(f"Output: {out_dir}")
    print(f"Frame step: {args.frame_step}, crop_top_ratio: {args.crop_top_ratio}")

    # Initialize scorer
    kwargs = {}
    if args.cfg:
        kwargs["cfg_path"] = args.cfg
    if args.ckpt:
        kwargs["ckpt_path"] = args.ckpt

    scorer = FlowAEPERewardScorer(
        crop_top_ratio=args.crop_top_ratio,
        frame_step=args.frame_step,
        device_id=args.device,
        epe_threshold=0.5,  # not used for analysis; we inspect raw scores
        **kwargs,
    )

    # Process videos
    all_results = []
    scene_results = {}  # scene_name -> list of dicts
    t0 = time.time()

    for idx, (scene_name, video_path) in enumerate(videos):
        video_stem = Path(video_path).stem
        t_start = time.time()

        result = scorer.score_continuous(video_path)
        elapsed = time.time() - t_start

        entry = {
            "scene": scene_name,
            "video": video_stem,
            "video_path": video_path,
            "score": result["score"],
            "avg_epe": result["avg_epe"],
            "dynamic_degree": result["dynamic_degree"],
            "num_pairs": len(result["per_pair_epe"]),
            "elapsed_sec": round(elapsed, 2),
        }
        all_results.append(entry)

        if scene_name not in scene_results:
            scene_results[scene_name] = []
        scene_results[scene_name].append(entry)

        print(f"[{idx+1}/{len(videos)}] {video_stem}: "
              f"score={result['score']:.4f}  avg_epe={result['avg_epe']:.4f}  "
              f"dynamic={result['dynamic_degree']:.4f}  ({elapsed:.1f}s)")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s ({total_time/len(videos):.1f}s per video)")

    # Compute summary statistics
    scores = [r["score"] for r in all_results]
    epes = [r["avg_epe"] for r in all_results]

    summary = {
        "num_videos": len(all_results),
        "num_scenes": len(scene_results),
        "frame_step": args.frame_step,
        "crop_top_ratio": args.crop_top_ratio,
        "total_time_sec": round(total_time, 2),
        "score_mean": round(float(sum(scores) / len(scores)), 6),
        "score_std": round(float((sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5), 6),
        "score_min": round(min(scores), 6),
        "score_max": round(max(scores), 6),
        "avg_epe_mean": round(float(sum(epes) / len(epes)), 6),
        "avg_epe_std": round(float((sum((e - sum(epes)/len(epes))**2 for e in epes) / len(epes))**0.5), 6),
        "avg_epe_min": round(min(epes), 6),
        "avg_epe_max": round(max(epes), 6),
    }

    # Per-scene summary
    scene_summaries = {}
    for scene_name, entries in sorted(scene_results.items()):
        sc = [e["score"] for e in entries]
        ep = [e["avg_epe"] for e in entries]
        scene_summaries[scene_name] = {
            "num_videos": len(entries),
            "score_mean": round(float(sum(sc) / len(sc)), 6),
            "score_min": round(min(sc), 6),
            "score_max": round(max(sc), 6),
            "avg_epe_mean": round(float(sum(ep) / len(ep)), 6),
        }

    # Save results
    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"summary": summary, "per_scene": scene_summaries}, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Videos:        {summary['num_videos']}")
    print(f"  Score (1/EPE): mean={summary['score_mean']:.4f}  "
          f"std={summary['score_std']:.4f}  "
          f"range=[{summary['score_min']:.4f}, {summary['score_max']:.4f}]")
    print(f"  Avg EPE:       mean={summary['avg_epe_mean']:.4f}  "
          f"std={summary['avg_epe_std']:.4f}  "
          f"range=[{summary['avg_epe_min']:.4f}, {summary['avg_epe_max']:.4f}]")
    print()
    print("Per-scene avg_epe:")
    for scene_name, ss in sorted(scene_summaries.items()):
        print(f"  {scene_name}: avg_epe_mean={ss['avg_epe_mean']:.4f}  "
              f"score_mean={ss['score_mean']:.4f}")

    # Also save a sorted ranking (worst EPE first = most likely hallucination)
    ranked = sorted(all_results, key=lambda r: r["avg_epe"], reverse=True)
    with open(out_dir / "ranked_by_epe.json", "w") as f:
        json.dump(ranked, f, indent=2)

    print(f"\nTop 10 worst EPE (most likely hallucination):")
    for i, r in enumerate(ranked[:10]):
        print(f"  {i+1}. {r['video']}: avg_epe={r['avg_epe']:.4f}  score={r['score']:.4f}")

    print(f"\nResults saved to {out_dir}/")

    scorer.shutdown()


if __name__ == "__main__":
    main()
