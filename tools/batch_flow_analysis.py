"""
Batch Farneback optical flow analysis for NFT training videos.

Computes per-video flow metrics (mean_mag, max_mag, std_mag, max_jerk,
anomaly_ratio), cross-references CLEAN/FAIL labels from reward_debug,
saves 2-panel output videos (original + magnitude heatmap), and produces
a summary JSON sorted by max_jerk.

No GPU required (Farneback is CPU-only).

Usage:
    python tools/batch_flow_analysis.py
    python tools/batch_flow_analysis.py --skip-video   # metrics-only (faster)
"""

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch

from fastvideo.models.wan.utils.utils import save_video as _wan_save_video


# ---------------------------------------------------------------------------
# Reused from tools/detect_flow_anomalies.py
# ---------------------------------------------------------------------------

def save_video_libx264(frames_bgr: list, output_path: str, fps: float) -> None:
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    tensor = torch.from_numpy(np.stack(rgb, axis=0))
    tensor = tensor.float() / 127.5 - 1.0
    tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)
    _wan_save_video(tensor, save_file=output_path, fps=fps,
                    normalize=True, value_range=(-1, 1))


def compute_flow_farneback(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )


def flow_to_magnitude_heatmap(mag: np.ndarray, mag_clip: float | None = None) -> np.ndarray:
    if mag_clip is None:
        mag_clip = float(np.percentile(mag, 99.0)) if mag.size > 0 else 1.0
    norm = np.clip(mag / max(mag_clip, 1e-6), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    cv2.putText(heatmap, f"mag (0~{mag_clip:.1f}px)", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return heatmap


def robust_stats(x: np.ndarray):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, mad


# ---------------------------------------------------------------------------
# Label lookup
# ---------------------------------------------------------------------------

def lookup_label(stem: str, labels_dir: str) -> str:
    """Check reward_debug dir for {stem}_CLEAN.mp4 or {stem}_FAIL.mp4."""
    for suffix, label in [("_CLEAN.mp4", "CLEAN"), ("_FAIL.mp4", "FAIL")]:
        if os.path.exists(os.path.join(labels_dir, stem + suffix)):
            return label
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Per-video analysis
# ---------------------------------------------------------------------------

def analyze_video(video_path: str, labels_dir: str, out_dir: str,
                  crop_top_ratio: float, abs_mag_min: float,
                  skip_video: bool) -> dict | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    crop_h = int(h_full * crop_top_ratio) // 16 * 16

    stem = os.path.splitext(os.path.basename(video_path))[0]
    label = lookup_label(stem, labels_dir)

    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        return None
    frame0 = frame0[:crop_h, :]
    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    per_frame_mean = []
    per_frame_max = []
    anomaly_ratios = []
    output_frames = []

    if not skip_video:
        # First frame: no flow yet, show blank heatmap
        blank_heat = np.zeros_like(frame0)
        panel = np.concatenate([frame0, blank_heat], axis=1)
        pw = panel.shape[1] // 16 * 16
        ph = panel.shape[0] // 16 * 16
        output_frames.append(panel[:ph, :pw])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame = frame[:crop_h, :]
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = compute_flow_farneback(prev_gray, curr_gray)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        fm = float(np.mean(mag))
        fmx = float(np.max(mag))
        per_frame_mean.append(fm)
        per_frame_max.append(fmx)

        # Anomaly ratio: fraction of pixels with mag > (median + 3*MAD) AND > abs_mag_min
        med, mad = robust_stats(mag)
        high_mag = (mag > (med + 3.0 * mad)) & (mag > abs_mag_min)
        anomaly_ratios.append(float(np.mean(high_mag)))

        if not skip_video:
            heatmap = flow_to_magnitude_heatmap(mag)
            # Text overlay on original
            annotated = frame.copy()
            cv2.putText(annotated, f"f={frame_idx} mean={fm:.1f} max={fmx:.1f}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            panel = np.concatenate([annotated, heatmap], axis=1)
            pw = panel.shape[1] // 16 * 16
            ph = panel.shape[0] // 16 * 16
            output_frames.append(panel[:ph, :pw])

        prev_gray = curr_gray

    cap.release()

    if len(per_frame_mean) == 0:
        return None

    mean_mag = float(np.mean(per_frame_mean))
    max_mag = float(np.max(per_frame_max))
    std_mag = float(np.std(per_frame_mean))
    anomaly_ratio = float(np.mean(anomaly_ratios))

    # Jerk: max frame-to-frame change in mean magnitude
    arr = np.array(per_frame_mean)
    jerks = np.abs(np.diff(arr))
    max_jerk = float(np.max(jerks)) if len(jerks) > 0 else 0.0

    # Build output filename with metrics + label
    out_name = (f"{stem}_{label}"
                f"_meanmag{mean_mag:.1f}"
                f"_maxmag{max_mag:.1f}"
                f"_jerk{max_jerk:.1f}.mp4")
    out_path = os.path.join(out_dir, out_name)

    if not skip_video and output_frames:
        os.makedirs(out_dir, exist_ok=True)
        save_video_libx264(output_frames, out_path, fps)

    return {
        "stem": stem,
        "label": label,
        "mean_mag": round(mean_mag, 3),
        "max_mag": round(max_mag, 3),
        "std_mag": round(std_mag, 3),
        "max_jerk": round(max_jerk, 3),
        "anomaly_ratio": round(anomaly_ratio, 4),
        "output_video": out_path if not skip_video else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Batch Farneback flow analysis")
    p.add_argument("--videos-root",
                   default="data/outputs/nft_stack_bowls_three/ng4_s42_tl0.0_kl0.001/videos")
    p.add_argument("--labels-root",
                   default="data/outputs/nft_stack_bowls_three/ng4_s42_tl0.0_kl0.001/reward_debug")
    p.add_argument("--out-dir",
                   default="data/outputs/nft_stack_bowls_three/ng4_s42_tl0.0_kl0.001/flow_analysis")
    p.add_argument("--crop-top-ratio", type=float, default=0.6667)
    p.add_argument("--anomaly-threshold", type=float, default=1.5)
    p.add_argument("--skip-video", action="store_true",
                   help="Skip output video generation (metrics-only, faster)")
    args = p.parse_args()

    # Discover all videos
    pattern = os.path.join(args.videos_root, "step*", "*.mp4")
    video_paths = sorted(glob.glob(pattern))
    if not video_paths:
        print(f"No videos found matching {pattern}")
        return

    print(f"Found {len(video_paths)} videos")
    if args.skip_video:
        print("(--skip-video: metrics only, no output videos)")

    all_results = []
    t0 = time.time()

    for i, vp in enumerate(video_paths):
        step_dir = os.path.basename(os.path.dirname(vp))
        labels_dir = os.path.join(args.labels_root, step_dir)
        vid_out_dir = os.path.join(args.out_dir, step_dir)

        t1 = time.time()
        try:
            result = analyze_video(vp, labels_dir, vid_out_dir,
                                   args.crop_top_ratio, args.anomaly_threshold,
                                   args.skip_video)
        except Exception as e:
            print(f"  ERROR processing {vp}: {e}")
            continue

        if result is None:
            print(f"  SKIP {vp} (empty or unreadable)")
            continue

        result["step_dir"] = step_dir
        all_results.append(result)
        dt = time.time() - t1
        r = result
        print(f"[{i+1:3d}/{len(video_paths)}] {r['stem']}  "
              f"{r['label']:5s}  mean={r['mean_mag']:.1f}  max={r['max_mag']:.1f}  "
              f"jerk={r['max_jerk']:.1f}  ({dt:.1f}s)")

    total_time = time.time() - t0
    print(f"\nTotal: {total_time:.0f}s ({total_time/60:.1f}min), "
          f"avg {total_time/max(len(all_results),1):.1f}s/video")

    # Aggregate stats
    def agg(vals):
        a = np.array(vals)
        return {"mean": round(float(np.mean(a)), 3),
                "std": round(float(np.std(a)), 3),
                "min": round(float(np.min(a)), 3),
                "max": round(float(np.max(a)), 3)}

    metrics = ["mean_mag", "max_mag", "std_mag", "max_jerk", "anomaly_ratio"]
    aggregate = {m: agg([r[m] for r in all_results]) for m in metrics}

    by_label = {}
    for lab in ["CLEAN", "FAIL", "UNKNOWN"]:
        subset = [r for r in all_results if r["label"] == lab]
        if not subset:
            continue
        by_label[lab] = {"count": len(subset)}
        for m in metrics:
            vals = [r[m] for r in subset]
            by_label[lab][m] = {"mean": round(float(np.mean(vals)), 3),
                                "std": round(float(np.std(vals)), 3)}

    # Sort by max_jerk descending
    ranked = sorted(all_results, key=lambda r: r["max_jerk"], reverse=True)

    summary = {
        "config": {
            "videos_root": args.videos_root,
            "labels_root": args.labels_root,
            "crop_top_ratio": args.crop_top_ratio,
            "anomaly_threshold": args.anomaly_threshold,
            "total_videos": len(all_results),
        },
        "aggregate": aggregate,
        "by_label": by_label,
        "videos_ranked_by_jerk": ranked,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 70)
    print("CLEAN vs FAIL comparison")
    print("=" * 70)
    header = f"{'':>12s}  {'mean_mag':>8s}  {'max_mag':>8s}  {'std_mag':>8s}  {'max_jerk':>8s}  {'anom_ratio':>10s}"
    print(header)
    for lab in ["CLEAN", "FAIL"]:
        if lab not in by_label:
            continue
        bl = by_label[lab]
        print(f"{lab+'('+str(bl['count'])+')':>12s}  "
              f"{bl['mean_mag']['mean']:8.2f}  "
              f"{bl['max_mag']['mean']:8.2f}  "
              f"{bl['std_mag']['mean']:8.2f}  "
              f"{bl['max_jerk']['mean']:8.2f}  "
              f"{bl['anomaly_ratio']['mean']:10.4f}")

    print(f"\nTop 10 by max_jerk:")
    for i, r in enumerate(ranked[:10]):
        print(f"  {i+1:2d}. {r['stem']}  [{r['label']}]  "
              f"jerk={r['max_jerk']:.2f}  max={r['max_mag']:.1f}  mean={r['mean_mag']:.1f}")

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
