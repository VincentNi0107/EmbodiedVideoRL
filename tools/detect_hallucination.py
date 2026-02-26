"""
Detect object hallucinations in generated manipulation videos using SAM3.
(Original version — one session per prompt, no session reuse.)

Uses SAM3 video predictor (with cross-frame tracking) to track each prompted
object throughout the video. For each frame, counts instances of each object
and flags hallucination events: disappear, reappear, duplicate.

Outputs:
- An annotated mp4 with segmentation masks and per-frame object counts.
- A CSV with per-frame detection details.
- A JSON summary with hallucination events.
"""

import argparse
import csv
import json
import os
import sys
import tempfile

import cv2
import numpy as np
import torch

# wan.utils.utils.save_video for libx264 output (VSCode-compatible)
from fastvideo.models.wan.utils.utils import save_video as _wan_save_video


# ──────────────────────────────────────────────────────────────────────
# Video I/O helpers
# ──────────────────────────────────────────────────────────────────────

def save_video_libx264(frames_bgr: list, output_path: str, fps: float) -> None:
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    tensor = torch.from_numpy(np.stack(rgb, axis=0))
    tensor = tensor.float() / 127.5 - 1.0
    tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)
    _wan_save_video(tensor, save_file=output_path, fps=fps,
                    normalize=True, value_range=(-1, 1))


def extract_frames_to_jpeg(video_path: str, crop_h: int | None = None) -> str:
    """Extract video frames to a temp JPEG directory (SAM3 prefers JPEG folders).

    Returns the temp directory path.
    """
    tmpdir = tempfile.mkdtemp(prefix="sam3_frames_")
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if crop_h is not None:
            frame = frame[:crop_h]
        cv2.imwrite(os.path.join(tmpdir, f"{idx:06d}.jpg"), frame)
        idx += 1
    cap.release()
    return tmpdir


# ──────────────────────────────────────────────────────────────────────
# SAM3 video tracker
# ──────────────────────────────────────────────────────────────────────

# Distinct colors (BGR) for prompts
PALETTE = [
    (60, 60, 220),    # red block -> reddish
    (60, 200, 60),    # green block -> greenish
    (220, 140, 40),   # blue block -> bluish
    (0, 200, 200),    # yellow
    (200, 0, 200),    # magenta
    (200, 200, 0),    # cyan
]


def track_prompt(predictor, video_resource: str, prompt: str):
    """Run SAM3 video predictor for a single text prompt.

    Creates a dedicated session (start_session + close_session) for each prompt.

    Returns:
        list[dict] indexed by frame_idx, each containing:
            obj_ids: np.ndarray
            probs: np.ndarray
            boxes_xywh: np.ndarray (N, 4) normalized
            masks: np.ndarray (N, H, W) bool
            num_tracked: int
    """
    resp = predictor.handle_request(dict(
        type="start_session",
        resource_path=video_resource,
    ))
    session_id = resp["session_id"]

    # Add text prompt at frame 0
    predictor.handle_request(dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text=prompt,
    ))

    # Propagate through all frames
    frame_results = []
    for resp_frame in predictor.handle_stream_request(dict(
        type="propagate_in_video",
        session_id=session_id,
    )):
        outs = resp_frame["outputs"]
        frame_results.append({
            "obj_ids": outs["out_obj_ids"],
            "probs": outs["out_probs"],
            "boxes_xywh": outs["out_boxes_xywh"],
            "masks": outs["out_binary_masks"],
            "num_tracked": outs["frame_stats"]["num_obj_tracked"],
        })

    predictor.handle_request(dict(
        type="close_session",
        session_id=session_id,
    ))
    return frame_results


def draw_frame(frame_bgr: np.ndarray, prompts: list[str],
               all_results: dict, frame_idx: int,
               expected_counts: dict, total_frames: int) -> np.ndarray:
    """Draw masks, boxes, and summary text for one frame."""
    ann = frame_bgr.copy()
    h, w = ann.shape[:2]

    for pidx, prompt in enumerate(prompts):
        color = PALETTE[pidx % len(PALETTE)]
        res = all_results[prompt][frame_idx]
        n = len(res["obj_ids"])

        # Draw masks
        for i in range(n):
            mask = res["masks"][i]
            overlay = ann.copy()
            overlay[mask] = color
            ann = cv2.addWeighted(overlay, 0.35, ann, 0.65, 0)

        # Draw boxes (out_boxes_xywh = top-left x,y + width,height, normalized)
        for i in range(n):
            bx, by, bw, bh = res["boxes_xywh"][i]
            x0 = int(bx * w)
            y0 = int(by * h)
            x1 = int((bx + bw) * w)
            y1 = int((by + bh) * h)
            cv2.rectangle(ann, (x0, y0), (x1, y1), color, 2)
            label = f"{prompt} {res['probs'][i]:.2f}"
            cv2.putText(ann, label, (x0, max(y0 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Summary text at top
    y_text = 22
    for pidx, prompt in enumerate(prompts):
        color = PALETTE[pidx % len(PALETTE)]
        res = all_results[prompt][frame_idx]
        count = len(res["obj_ids"])
        expected = expected_counts.get(prompt, 1)
        if count != expected:
            label = f"{prompt}: {count} (expect {expected}) !!!"
            text_color = (0, 0, 255)
        else:
            label = f"{prompt}: {count}"
            text_color = color
        cv2.putText(ann, label, (8, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)
        y_text += 22

    # Frame number
    cv2.putText(ann, f"frame {frame_idx}/{total_frames}",
                (w - 180, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)
    return ann


# ──────────────────────────────────────────────────────────────────────
# Main processing
# ──────────────────────────────────────────────────────────────────────

def process_video(
    input_path: str,
    output_video: str,
    output_csv: str | None,
    output_json: str | None,
    prompts: list[str],
    expected_counts: dict[str, int],
    crop_top_ratio: float,
    predictor=None,
    occlusion_gap_max: int = 5,
    occlusion_pos_thr: float = 0.15,
    quiet: bool = False,
) -> dict:
    """Process a single video. Returns a summary dict.

    If *predictor* is provided it is reused (caller manages lifecycle).
    Otherwise a new one is built and shut down within this call.
    If *quiet* is True, suppress all print output and SAM3 progress bars / logs.
    """
    from sam3.model_builder import build_sam3_video_predictor
    import shutil
    import logging

    def _log(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    own_predictor = predictor is None

    # Read video info
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    crop_h = None
    if crop_top_ratio < 1.0:
        crop_h = (int(h_full * crop_top_ratio)) // 16 * 16
    h = crop_h if crop_h else h_full

    _log(f"Video: {total_frames} frames, {w}x{h} @ {fps:.1f} fps")
    _log(f"Prompts: {prompts}")
    _log(f"Expected counts: {expected_counts}")

    # Extract frames to JPEG dir (with optional crop)
    _log("Extracting frames...")
    jpeg_dir = extract_frames_to_jpeg(input_path, crop_h=crop_h)
    _log(f"  Frames saved to {jpeg_dir}")

    # Build predictor if not provided
    if own_predictor:
        _log("Loading SAM3 video predictor...")
        predictor = build_sam3_video_predictor()

    # Track each prompt independently (separate session per prompt)
    # In quiet mode, suppress SAM3 tqdm progress bars and logger output
    all_results = {}
    _quiet_loggers = []
    if quiet:
        os.environ["TQDM_DISABLE"] = "1"
        # Suppress all sam3.* loggers (they set propagate=False, so parent level won't help)
        for name in list(logging.Logger.manager.loggerDict):
            if name.startswith("sam3"):
                lg = logging.getLogger(name)
                _quiet_loggers.append((lg, lg.level))
                lg.setLevel(logging.ERROR)
    try:
        for prompt in prompts:
            _log(f"Tracking: '{prompt}'...")
            all_results[prompt] = track_prompt(predictor, jpeg_dir, prompt)
            _log(f"  Done. {len(all_results[prompt])} frames tracked.")
    finally:
        if quiet:
            os.environ.pop("TQDM_DISABLE", None)
            for lg, prev_level in _quiet_loggers:
                lg.setLevel(prev_level)

    if own_predictor:
        predictor.shutdown()

    # ── Post-process: suppress brief occlusion events ─────────────────────────
    # For each prompt, scan the per-frame count sequence and mark frames as
    # "occlusion_suppressed" (not a real hallucination) when:
    #   1. The object disappears for at most `occlusion_gap_max` consecutive frames
    #   2. After reappearing, its bounding-box centre is within `occlusion_iou_thr`
    #      (normalised L∞ distance) of where it was last seen before vanishing.
    # occlusion_gap_max / occlusion_pos_thr come from function parameters

    # suppressed[prompt][frame_idx] = True  →  treat as "okay" even if count wrong
    suppressed: dict[str, list[bool]] = {}
    for prompt in prompts:
        n_frames = len(all_results[prompt])
        expected = expected_counts.get(prompt, 1)
        sup = [False] * n_frames

        def centre(res):
            """Return mean box centre (cx, cy) normalised, or None if no boxes."""
            boxes = res["boxes_xywh"]
            if len(boxes) == 0:
                return None
            cx = float(np.mean([b[0] + b[2] / 2 for b in boxes]))
            cy = float(np.mean([b[1] + b[3] / 2 for b in boxes]))
            return (cx, cy)

        fi = 0
        while fi < n_frames:
            res_fi = all_results[prompt][fi]
            count_fi = len(res_fi["obj_ids"])
            # We only care about disappearance (count drops below expected)
            if count_fi < expected:
                gap_start = fi
                # Find where the count returns to expected
                gap_end = fi + 1
                while gap_end < n_frames and len(all_results[prompt][gap_end]["obj_ids"]) < expected:
                    gap_end += 1
                gap_len = gap_end - gap_start

                if gap_len <= occlusion_gap_max and gap_end < n_frames:
                    # Compare centre before gap vs centre after gap
                    c_before = centre(all_results[prompt][gap_start - 1]) if gap_start > 0 else None
                    c_after  = centre(all_results[prompt][gap_end])
                    if c_before is not None and c_after is not None:
                        dist = max(abs(c_after[0] - c_before[0]),
                                   abs(c_after[1] - c_before[1]))
                        if dist <= occlusion_pos_thr:
                            # Mark the gap frames as suppressed
                            for gfi in range(gap_start, gap_end):
                                sup[gfi] = True
                fi = gap_end  # resume scanning after gap
            else:
                fi += 1

        suppressed[prompt] = sup

    # Re-read frames for annotation (from the JPEG dir, already cropped)
    _log("Rendering annotated video...")
    out_frames = []
    rows = []
    hallucination_events = []

    for fi in range(total_frames):
        frame_bgr = cv2.imread(os.path.join(jpeg_dir, f"{fi:06d}.jpg"))
        ann = draw_frame(frame_bgr, prompts, all_results, fi,
                         expected_counts, total_frames)
        out_frames.append(ann)

        # Build per-frame row
        row = {"frame_idx": fi}
        is_hall = False
        for prompt in prompts:
            res = all_results[prompt][fi]
            count = len(res["obj_ids"])
            key = prompt.replace(" ", "_")
            row[f"{key}_count"] = count
            row[f"{key}_tracked"] = res["num_tracked"]
            row[f"{key}_probs"] = ",".join(f"{p:.3f}" for p in res["probs"])
            row[f"{key}_suppressed"] = suppressed[prompt][fi]

            expected = expected_counts.get(prompt, 1)
            if count != expected and not suppressed[prompt][fi]:
                is_hall = True
                hallucination_events.append({
                    "frame_idx": fi,
                    "prompt": prompt,
                    "expected": expected,
                    "detected": count,
                    "type": "disappeared" if count < expected else "duplicated",
                })
        row["hallucination"] = is_hall
        rows.append(row)

    # Save video
    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
    _log(f"Saving video ({len(out_frames)} frames)...")
    save_video_libx264(out_frames, output_video, fps)

    # Cleanup temp frames
    shutil.rmtree(jpeg_dir, ignore_errors=True)

    # Save CSV (optional)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        if rows:
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    # Build summary
    counts_over_time = {}
    for prompt in prompts:
        key = prompt.replace(" ", "_")
        vals = [r[f"{key}_count"] for r in rows]
        counts_over_time[prompt] = {
            "min": int(min(vals)),
            "max": int(max(vals)),
            "mean": round(sum(vals) / len(vals), 2) if vals else 0,
            "frames_with_zero": sum(1 for v in vals if v == 0),
            "frames_with_multiple": sum(1 for v in vals if v > 1),
            "frames_suppressed_occlusion": sum(1 for r in rows if r[f"{key}_suppressed"]),
        }

    summary = {
        "input_video": input_path,
        "output_video": output_video,
        "frame_count": len(rows),
        "fps": fps,
        "prompts": prompts,
        "expected_counts": expected_counts,
        "occlusion_suppression": {
            "gap_max_frames": occlusion_gap_max,
            "position_threshold": occlusion_pos_thr,
        },
        "counts_over_time": counts_over_time,
        "total_hallucination_frames": sum(1 for r in rows if r["hallucination"]),
        "hallucination_events": hallucination_events,
    }

    # Save JSON (optional)
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    _log("Done.")
    _log(f"- Annotated video: {output_video}")
    n_hall = summary["total_hallucination_frames"]
    _log(f"- Hallucination frames: {n_hall}/{len(rows)} "
         f"({100 * n_hall / max(len(rows), 1):.1f}%)")
    for prompt, stats in counts_over_time.items():
        _log(f"  {prompt}: min={stats['min']} max={stats['max']} "
             f"mean={stats['mean']:.2f} "
             f"zero_frames={stats['frames_with_zero']} "
             f"multi_frames={stats['frames_with_multiple']} "
             f"suppressed={stats['frames_suppressed_occlusion']}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Detect object hallucinations in generated videos using SAM3")
    parser.add_argument("--input", required=True, type=str,
                        help="Input video path")
    parser.add_argument("--out-dir", default=None, type=str,
                        help="Output directory")
    parser.add_argument("--prompts", nargs="+",
                        default=["red block", "green block", "blue block"],
                        help="Object text prompts for SAM3")
    parser.add_argument("--expected-counts", nargs="+", type=int,
                        default=None,
                        help="Expected count for each prompt (same order). "
                             "Default: 1 for each prompt.")
    parser.add_argument("--crop-top-ratio", type=float, default=2 / 3,
                        help="Fraction of frame height to keep from top "
                             "(default 2/3 to crop wrist cameras)")
    parser.add_argument("--occlusion-gap-max", type=int, default=5,
                        help="Max consecutive frames an object may be absent "
                             "before it is counted as a real hallucination "
                             "(default 5). Set to 0 to disable suppression.")
    parser.add_argument("--occlusion-pos-thr", type=float, default=0.15,
                        help="Max normalised L∞ centre-shift allowed when an "
                             "object reappears to be considered the same object "
                             "(default 0.15).")
    args = parser.parse_args()

    # Build expected counts
    if args.expected_counts is not None:
        if len(args.expected_counts) != len(args.prompts):
            parser.error("--expected-counts must match --prompts length")
        expected = dict(zip(args.prompts, args.expected_counts))
    else:
        expected = {p: 1 for p in args.prompts}

    video_stem = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = (args.out_dir or
               os.path.join(os.path.dirname(__file__),
                            "hallucination_outputs", video_stem))
    os.makedirs(out_dir, exist_ok=True)

    process_video(
        input_path=args.input,
        output_video=os.path.join(out_dir, f"{video_stem}_hallucination.mp4"),
        output_csv=os.path.join(out_dir, f"{video_stem}_detections.csv"),
        output_json=os.path.join(out_dir, f"{video_stem}_summary.json"),
        prompts=args.prompts,
        expected_counts=expected,
        crop_top_ratio=args.crop_top_ratio,
        occlusion_gap_max=args.occlusion_gap_max,
        occlusion_pos_thr=args.occlusion_pos_thr,
    )


if __name__ == "__main__":
    main()
