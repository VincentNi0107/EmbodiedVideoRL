"""Batch validation of BowlStackRewardScorer on rollout videos.

Processes all robotwin_stack_bowls_three_* scene directories under --input-root.
Saves annotated videos tagged _CLEAN / _FAIL and a grand summary JSON.

Usage:
    python validate_bowl_stack_reward.py \
        --input-root data/outputs/rollout_robotwin_121 \
        --out-root   data/outputs/bowl_stack_validation
"""

import argparse
import json
import os
import sys
import glob
import time

sys.path.insert(0, os.path.dirname(__file__))

# Set LOCAL_RANK for main_print (normally set by torchrun)
os.environ.setdefault("LOCAL_RANK", "0")


def main():
    parser = argparse.ArgumentParser(description="Batch bowl stack reward validation")
    parser.add_argument("--input-root", type=str,
                        default="data/outputs/rollout_robotwin_121",
                        help="Root directory containing scene folders")
    parser.add_argument("--out-root", type=str,
                        default="data/outputs/bowl_stack_validation",
                        help="Output directory for annotated videos and summaries")
    parser.add_argument("--pattern", type=str,
                        default="robotwin_stack_bowls_three_*",
                        help="Glob pattern for scene directories")
    # Scorer params
    parser.add_argument("--prompt", type=str, default="bowl")
    parser.add_argument("--initial-count", type=int, default=3)
    parser.add_argument("--crop-top-ratio", type=float, default=2/3)
    parser.add_argument("--convergence-thr", type=float, default=0.30)
    parser.add_argument("--check-window-frac", type=float, default=0.20)
    parser.add_argument("--gap-max", type=int, default=5)
    parser.add_argument("--reappear-max", type=int, default=10)
    parser.add_argument("--reappear-pos-thr", type=float, default=0.15)
    args = parser.parse_args()

    # Find scene directories
    scene_dirs = sorted(glob.glob(os.path.join(args.input_root, args.pattern)))
    if not scene_dirs:
        print(f"No scene directories found matching {args.pattern} in {args.input_root}")
        return

    print(f"Found {len(scene_dirs)} scene directories")

    # Load SAM3 predictor once
    import torch
    from fastvideo.reward.hallucination_bowls import BowlStackRewardScorer

    scorer = BowlStackRewardScorer(
        prompt=args.prompt,
        initial_count=args.initial_count,
        crop_top_ratio=args.crop_top_ratio,
        convergence_thr=args.convergence_thr,
        check_window_frac=args.check_window_frac,
        gap_max=args.gap_max,
        reappear_max=args.reappear_max,
        reappear_pos_thr=args.reappear_pos_thr,
        device_id=0,
    )

    os.makedirs(args.out_root, exist_ok=True)

    grand_results = []
    total_clean = 0
    total_fail = 0
    total_error = 0

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        scene_out = os.path.join(args.out_root, scene_name)
        os.makedirs(scene_out, exist_ok=True)

        videos = sorted(glob.glob(os.path.join(scene_dir, "*.mp4")))
        print(f"\n{'='*60}")
        print(f"Scene: {scene_name}  ({len(videos)} videos)")
        print(f"{'='*60}")

        scene_results = []
        for vpath in videos:
            vstem = os.path.splitext(os.path.basename(vpath))[0]
            t0 = time.time()

            result = scorer.score(
                prompt="",
                first_frame=None,
                video_path=os.path.abspath(vpath),
                debug_save_path=os.path.join(scene_out, f"{vstem}_debug.mp4"),
            )

            elapsed = time.time() - t0
            tag = "CLEAN" if result.get("pass", False) else "FAIL"
            reward = result.get("reward", 0.0)
            resp = result.get("_response_text", "")

            if reward == 1.0:
                total_clean += 1
            elif "ERROR" in resp:
                total_error += 1
            else:
                total_fail += 1

            print(f"  [{tag}] {vstem}  ({elapsed:.1f}s)  {resp}")

            scene_results.append({
                "video": vstem,
                "reward": reward,
                "pass": result.get("pass", False),
                "fail_reasons": result.get("fail_reasons", []),
                "num_real_objects": result.get("num_real_objects", -1),
                "convergence_max_dist": result.get("convergence_max_dist", -1),
                "response": resp,
            })

        # Save per-scene summary
        scene_summary_path = os.path.join(scene_out, f"{scene_name}_analysis.json")
        with open(scene_summary_path, "w") as f:
            json.dump(scene_results, f, indent=2)

        grand_results.append({
            "scene": scene_name,
            "total": len(videos),
            "clean": sum(1 for r in scene_results if r["pass"]),
            "fail": sum(1 for r in scene_results if not r["pass"]),
            "videos": scene_results,
        })

    # Grand summary
    total = total_clean + total_fail + total_error
    print(f"\n{'='*60}")
    print(f"GRAND SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {total}  CLEAN: {total_clean}  FAIL: {total_fail}  ERROR: {total_error}")
    print(f"Clean rate: {total_clean/total*100:.1f}%" if total > 0 else "N/A")

    grand_summary = {
        "total": total,
        "clean": total_clean,
        "fail": total_fail,
        "error": total_error,
        "clean_rate": total_clean / total if total > 0 else 0,
        "scenes": grand_results,
    }
    grand_path = os.path.join(args.out_root, "grand_summary.json")
    with open(grand_path, "w") as f:
        json.dump(grand_summary, f, indent=2)
    print(f"Summary saved to {grand_path}")

    scorer.shutdown()


if __name__ == "__main__":
    main()
