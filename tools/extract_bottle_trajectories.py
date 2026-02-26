"""
Extract individual bottle trajectories from put_bottles_dustbin videos using SAM3.

For each video, tracks "bottle" with SAM3, then extracts per-object trajectories
(using obj_ids for consistent tracking across frames). Saves:
  - Per-video JSON with full trajectory data (frame-by-frame bbox centers per bottle)
  - Per-video annotated tracking MP4 with color-coded bounding boxes and trajectory trails
  - Summary JSON with leftmost x-coordinate reached by each bottle across all videos
"""

import argparse
import json
import os
import sys
import tempfile
import shutil

import cv2
import numpy as np
import torch

# Reuse helpers from detect_hallucination_bottles
sys.path.insert(0, os.path.dirname(__file__))
from detect_hallucination_bottles import extract_frames_to_jpeg, track_prompt, save_video_libx264

# Distinct colors per obj_id (BGR)
OBJ_COLORS_BGR = [
    (0, 0, 255),     # red     - obj 0
    (0, 200, 0),     # green   - obj 1
    (255, 150, 0),   # blue    - obj 2
    (0, 200, 255),   # yellow  - obj 3
    (255, 0, 200),   # magenta - obj 4
]


def draw_tracking_frame(
    frame_bgr: np.ndarray,
    frame_result: dict,
    frame_idx: int,
    total_frames: int,
    trail_history: dict,  # obj_id -> list of (cx_px, cy_px)
) -> np.ndarray:
    """Draw bounding boxes, obj_id labels, and trajectory trails on a frame."""
    ann = frame_bgr.copy()
    h, w = ann.shape[:2]

    for i, obj_id in enumerate(frame_result["obj_ids"]):
        color = OBJ_COLORS_BGR[obj_id % len(OBJ_COLORS_BGR)]
        bx, by, bw, bh = frame_result["boxes_xywh"][i]
        x0, y0 = int(bx * w), int(by * h)
        x1, y1 = int((bx + bw) * w), int((by + bh) * h)
        cx_px, cy_px = (x0 + x1) // 2, (y0 + y1) // 2
        prob = frame_result["probs"][i]

        # Bounding box
        cv2.rectangle(ann, (x0, y0), (x1, y1), color, 2)

        # Label: obj_id + probability
        label = f"id={obj_id} {prob:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(ann, (x0, max(y0 - th - 8, 0)), (x0 + tw + 4, y0), color, -1)
        cv2.putText(ann, label, (x0 + 2, max(y0 - 4, th + 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Center dot
        cv2.circle(ann, (cx_px, cy_px), 3, color, -1)

        # Accumulate trail
        oid_str = str(obj_id)
        if oid_str not in trail_history:
            trail_history[oid_str] = []
        trail_history[oid_str].append((cx_px, cy_px))

    # Draw trajectory trails for all known objects
    for oid_str, trail in trail_history.items():
        if len(trail) < 2:
            continue
        color = OBJ_COLORS_BGR[int(oid_str) % len(OBJ_COLORS_BGR)]
        for k in range(1, len(trail)):
            # Fade older points
            alpha = 0.3 + 0.7 * (k / len(trail))
            c = tuple(int(v * alpha) for v in color)
            cv2.line(ann, trail[k - 1], trail[k], c, 1, cv2.LINE_AA)

    # Frame counter
    cv2.putText(ann, f"frame {frame_idx}/{total_frames}",
                (w - 190, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    return ann


def extract_trajectories(
    input_path: str,
    output_video: str | None = None,
    prompt: str = "bottle",
    crop_top_ratio: float = 2 / 3,
    predictor=None,
) -> dict:
    """Extract per-object trajectories from a single video.

    Args:
        output_video: if provided, save annotated tracking video to this path.

    Returns dict with:
        video: input path
        frame_count: total frames
        frame_size: (w, h) after crop
        objects: {obj_id: [{frame, cx, cy, x, y, w, h, prob}, ...]}
    """
    from sam3.model_builder import build_sam3_video_predictor

    own_predictor = predictor is None

    # Read video metadata
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    crop_h = None
    if crop_top_ratio < 1.0:
        crop_h = (int(h_full * crop_top_ratio)) // 16 * 16
    h = crop_h if crop_h else h_full

    print(f"Video: {input_path}")
    print(f"  {total_frames} frames, {vid_w}x{h} @ {fps:.1f} fps")

    # Extract frames
    jpeg_dir = extract_frames_to_jpeg(input_path, crop_h=crop_h)

    # Build predictor if needed
    if own_predictor:
        print("Loading SAM3 video predictor...")
        predictor = build_sam3_video_predictor()

    # Track bottles
    print(f"  Tracking '{prompt}'...")
    frame_results = track_prompt(predictor, jpeg_dir, prompt)
    print(f"  Done. {len(frame_results)} frames tracked.")

    if own_predictor:
        predictor.shutdown()

    # Build per-object trajectories
    objects = {}  # obj_id -> list of per-frame records

    for fi, fr in enumerate(frame_results):
        for i, obj_id in enumerate(fr["obj_ids"]):
            obj_id_str = str(obj_id)
            if obj_id_str not in objects:
                objects[obj_id_str] = []

            bx, by, bw, bh = fr["boxes_xywh"][i]
            cx = float(bx + bw / 2)
            cy = float(by + bh / 2)
            prob = float(fr["probs"][i])

            objects[obj_id_str].append({
                "frame": fi,
                "cx": cx,
                "cy": cy,
                "x": float(bx),
                "y": float(by),
                "w": float(bw),
                "h": float(bh),
                "prob": prob,
            })

    # Render annotated tracking video
    if output_video:
        print("  Rendering tracking video...")
        trail_history = {}
        out_frames = []
        for fi in range(total_frames):
            frame_bgr = cv2.imread(os.path.join(jpeg_dir, f"{fi:06d}.jpg"))
            ann = draw_tracking_frame(
                frame_bgr, frame_results[fi], fi, total_frames, trail_history,
            )
            out_frames.append(ann)
        os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
        save_video_libx264(out_frames, output_video, fps)
        print(f"  Saved tracking video: {output_video}")

    # Clean up temp frames
    shutil.rmtree(jpeg_dir, ignore_errors=True)

    result = {
        "video": input_path,
        "frame_count": total_frames,
        "frame_size": [vid_w, h],
        "fps": fps,
        "prompt": prompt,
        "crop_top_ratio": crop_top_ratio,
        "num_objects_tracked": len(objects),
        "objects": objects,
    }

    return result


def analyze_leftmost(all_results: list[dict]) -> dict:
    """Analyze trajectories to find leftmost x-coordinate reached by each bottle.

    Returns per-video analysis with leftmost cx for each tracked object.
    """
    analysis = {}
    for res in all_results:
        video_name = os.path.basename(res["video"])
        vid_analysis = {
            "video": video_name,
            "num_objects": res["num_objects_tracked"],
            "objects": {},
        }
        for obj_id, traj in res["objects"].items():
            if not traj:
                continue
            # Find the frame where this object has the smallest cx (leftmost)
            min_entry = min(traj, key=lambda t: t["cx"])
            # Also compute some stats
            cx_values = [t["cx"] for t in traj]
            vid_analysis["objects"][obj_id] = {
                "num_frames_visible": len(traj),
                "first_frame": traj[0]["frame"],
                "last_frame": traj[-1]["frame"],
                "leftmost_cx": min_entry["cx"],
                "leftmost_cy": min_entry["cy"],
                "leftmost_frame": min_entry["frame"],
                "rightmost_cx": max(cx_values),
                "mean_cx": float(np.mean(cx_values)),
                "initial_cx": traj[0]["cx"],
                "initial_cy": traj[0]["cy"],
            }
        analysis[video_name] = vid_analysis
    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Extract bottle trajectories from put_bottles_dustbin videos")
    parser.add_argument("--input-dir", required=True, type=str,
                        help="Directory containing video files")
    parser.add_argument("--out-dir", default=None, type=str,
                        help="Output directory (default: <input-dir>_trajectories)")
    parser.add_argument("--prompt", default="bottle", type=str)
    parser.add_argument("--crop-top-ratio", type=float, default=2 / 3)
    parser.add_argument("--pattern", default=None, type=str,
                        help="Only process videos matching this pattern")
    args = parser.parse_args()

    out_dir = args.out_dir or args.input_dir + "_trajectories"
    os.makedirs(out_dir, exist_ok=True)

    # Find videos
    videos = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".mp4")
        and (args.pattern is None or args.pattern in f)
    ])
    print(f"Found {len(videos)} videos in {args.input_dir}")

    # Load SAM3 predictor once, reuse for all videos
    from sam3.model_builder import build_sam3_video_predictor
    print("Loading SAM3 video predictor...")
    predictor = build_sam3_video_predictor()

    all_results = []
    for vi, vpath in enumerate(videos):
        print(f"\n{'='*60}")
        print(f"[{vi+1}/{len(videos)}] Processing: {os.path.basename(vpath)}")
        print(f"{'='*60}")

        stem = os.path.splitext(os.path.basename(vpath))[0]
        out_video = os.path.join(out_dir, f"{stem}_tracking.mp4")

        result = extract_trajectories(
            input_path=vpath,
            output_video=out_video,
            prompt=args.prompt,
            crop_top_ratio=args.crop_top_ratio,
            predictor=predictor,
        )
        all_results.append(result)

        # Save per-video trajectory JSON
        out_json = os.path.join(out_dir, f"{stem}_trajectory.json")
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_json}")

    predictor.shutdown()

    # Analyze leftmost positions
    analysis = analyze_leftmost(all_results)

    # Save analysis
    analysis_path = os.path.join(out_dir, "leftmost_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved leftmost analysis: {analysis_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("LEFTMOST X-COORDINATE ANALYSIS")
    print(f"{'='*60}")
    for video_name, va in analysis.items():
        print(f"\n{video_name}:")
        print(f"  Objects tracked: {va['num_objects']}")
        for obj_id, info in sorted(va["objects"].items(), key=lambda x: x[1]["initial_cx"]):
            print(f"  Bottle obj_id={obj_id}:")
            print(f"    Initial position:  cx={info['initial_cx']:.4f}, cy={info['initial_cy']:.4f}")
            print(f"    Leftmost reached:  cx={info['leftmost_cx']:.4f} (frame {info['leftmost_frame']})")
            print(f"    Visible frames:    {info['first_frame']}-{info['last_frame']} ({info['num_frames_visible']} frames)")


if __name__ == "__main__":
    main()
