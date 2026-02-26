"""
Batch hallucination detection for all blocks_ranking_rgb videos.
(Improved version — with occlusion suppression.)

Builds SAM3 predictor once, then processes each video sequentially.
Output videos are named with hallucination status prefix:
  hallucinated_<stem>.mp4   or   clean_<stem>.mp4

All videos are placed flat in a single episode-level output directory:
  <out_root>/<episode_dir>/<prefix>_<stem>.mp4

Brief disappearances (≤ occlusion-gap-max frames) where the object
reappears at roughly the same position are NOT counted as hallucinations.
"""

import argparse
import glob
import os
import sys
import time

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(__file__))

from detect_hallucination import process_video


def main():
    parser = argparse.ArgumentParser(
        description="Batch hallucination detection on blocks_ranking_rgb videos (improved)")
    parser.add_argument("--input-root", type=str,
                        default="/gpfs/projects/p33048/DanceGRPO/data/outputs/rollout_robotwin_121",
                        help="Root directory containing episode subdirectories")
    parser.add_argument("--out-root", type=str,
                        default=None,
                        help="Output root (default: <input_root>_hallucination)")
    parser.add_argument("--pattern", type=str,
                        default="robotwin_blocks_ranking_rgb_*",
                        help="Glob pattern for episode directories")
    parser.add_argument("--prompts", nargs="+",
                        default=["red block", "green block", "blue block"],
                        help="Object text prompts for SAM3")
    parser.add_argument("--expected-counts", nargs="+", type=int,
                        default=None,
                        help="Expected count for each prompt")
    parser.add_argument("--crop-top-ratio", type=float, default=2 / 3,
                        help="Fraction of frame height to keep from top")
    parser.add_argument("--occlusion-gap-max", type=int, default=5,
                        help="Max gap frames to suppress as occlusion (default 5)")
    parser.add_argument("--occlusion-pos-thr", type=float, default=0.15,
                        help="Max normalised centre-shift to suppress as occlusion (default 0.15)")
    args = parser.parse_args()

    out_root = args.out_root or (args.input_root.rstrip("/") + "_hallucination")

    # Build expected counts
    if args.expected_counts is not None:
        if len(args.expected_counts) != len(args.prompts):
            parser.error("--expected-counts must match --prompts length")
        expected = dict(zip(args.prompts, args.expected_counts))
    else:
        expected = {p: 1 for p in args.prompts}

    # Collect all mp4 files matching pattern
    search = os.path.join(args.input_root, args.pattern, "*.mp4")
    videos = sorted(glob.glob(search))
    print(f"Found {len(videos)} videos matching {search}")
    if not videos:
        print("Nothing to do.")
        return

    os.makedirs(out_root, exist_ok=True)

    # Build SAM3 predictor once
    print("Loading SAM3 video predictor (shared across all videos)...")
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor()

    results_summary = []
    t_start = time.time()

    for vi, video_path in enumerate(videos):
        stem = os.path.splitext(os.path.basename(video_path))[0]
        episode_dir = os.path.basename(os.path.dirname(video_path))
        out_dir = os.path.join(out_root, episode_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Use a temp name first, rename after we know the result
        tmp_video = os.path.join(out_dir, f"_tmp_{stem}.mp4")

        # Skip if already processed (either prefix exists)
        hall_path = os.path.join(out_dir, f"hallucinated_{stem}.mp4")
        clean_path = os.path.join(out_dir, f"clean_{stem}.mp4")
        if os.path.exists(hall_path) or os.path.exists(clean_path):
            print(f"[{vi+1}/{len(videos)}] SKIP {stem}")
            continue

        print(f"\n{'='*60}")
        print(f"[{vi+1}/{len(videos)}] Processing: {stem}")
        print(f"{'='*60}")

        try:
            summary = process_video(
                input_path=video_path,
                output_video=tmp_video,
                output_csv=None,
                output_json=None,
                prompts=args.prompts,
                expected_counts=expected,
                crop_top_ratio=args.crop_top_ratio,
                predictor=predictor,
                occlusion_gap_max=args.occlusion_gap_max,
                occlusion_pos_thr=args.occlusion_pos_thr,
            )

            # Rename based on hallucination status
            n_hall = summary["total_hallucination_frames"]
            has_hallucination = n_hall > 0
            prefix = "hallucinated" if has_hallucination else "clean"
            final_path = os.path.join(out_dir, f"{prefix}_{stem}.mp4")
            os.rename(tmp_video, final_path)

            results_summary.append({
                "video": stem,
                "hallucination": has_hallucination,
                "hall_frames": n_hall,
                "total_frames": summary["frame_count"],
                "output": final_path,
            })
            print(f"  -> {os.path.basename(final_path)}")

        except Exception as e:
            print(f"  ERROR: {e}")
            # Clean up temp file if it exists
            if os.path.exists(tmp_video):
                os.remove(tmp_video)
            results_summary.append({
                "video": stem,
                "hallucination": None,
                "error": str(e),
            })

    predictor.shutdown()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE  ({elapsed:.0f}s = {elapsed/60:.1f}min)")
    print(f"{'='*60}")
    print(f"Total: {len(videos)} videos")

    n_processed = sum(1 for r in results_summary if r.get("hallucination") is not None)
    n_hall = sum(1 for r in results_summary if r.get("hallucination") is True)
    n_clean = sum(1 for r in results_summary if r.get("hallucination") is False)
    n_err = sum(1 for r in results_summary if r.get("hallucination") is None)

    print(f"Processed: {n_processed}  |  Hallucinated: {n_hall}  |  Clean: {n_clean}  |  Errors: {n_err}")
    print(f"Output root: {out_root}")

    # Print per-video summary
    for r in results_summary:
        if r.get("hallucination") is None:
            status = "ERROR"
        elif r["hallucination"]:
            status = f"HALL ({r['hall_frames']}/{r['total_frames']})"
        else:
            status = "CLEAN"
        print(f"  {status:20s}  {r['video']}")


if __name__ == "__main__":
    main()
