"""
Detect hallucinations in 'put bottles in dustbin' task videos using SAM3.

Key differences from detect_hallucination.py (blocks_ranking_rgb):

1. All bottles share the same text prompt ("bottle") — can't distinguish by color.
2. Bottles legitimately disappear when placed in the dustbin (count should be
   monotonically non-increasing over the course of the video: 3 → 2 → 1 → 0).
3. Hallucination = count INCREASES above the "committed" count (a bottle that was
   placed in the dustbin has impossibly reappeared).
4. Occlusion = brief count decrease that returns to the committed level (robot arm
   temporarily covers a bottle during manipulation).

Algorithm — compute_monotonic_baseline():
  - Maintains a "committed count" (cur_committed) that starts at the initial observed
    count and can only decrease.
  - A contiguous run where raw_count < cur_committed is classified as:
      • occlusion (suppressed): run length ≤ gap_max AND count returns to cur_committed
      • committed decrease: run is longer OR video ends before recovery
  - Any frame where raw_count > cur_committed is a hallucination.

Outputs:
  - Annotated mp4 with SAM3 masks/boxes and a colour-coded status bar.
  - CSV with per-frame counts, committed count, suppressed flag, hallucination flag.
  - JSON summary with overall statistics.
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


# ──────────────────────────────────────────────────────────────────────────────
# Video I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_video_libx264(frames_bgr: list, output_path: str, fps: float) -> None:
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    tensor = torch.from_numpy(np.stack(rgb, axis=0))
    tensor = tensor.float() / 127.5 - 1.0
    tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)
    _wan_save_video(tensor, save_file=output_path, fps=fps,
                    normalize=True, value_range=(-1, 1))


def extract_frames_to_jpeg(video_path: str, crop_h: int | None = None) -> str:
    """Extract video frames to a temp JPEG directory (SAM3 prefers JPEG folders)."""
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


# ──────────────────────────────────────────────────────────────────────────────
# SAM3 video tracker
# ──────────────────────────────────────────────────────────────────────────────

# BGR colour used for all bottle annotations
BOTTLE_COLOR_BGR = (60, 130, 220)   # orange-ish


def track_prompt(predictor, video_resource: str, prompt: str):
    """Run SAM3 video predictor for a single text prompt (one session).

    Returns list[dict] indexed by frame_idx, each with:
        obj_ids, probs, boxes_xywh (N,4) normalised, masks (N,H,W) bool, num_tracked
    """
    resp = predictor.handle_request(dict(
        type="start_session",
        resource_path=video_resource,
    ))
    session_id = resp["session_id"]

    predictor.handle_request(dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text=prompt,
    ))

    frame_results = []
    for resp_frame in predictor.handle_stream_request(dict(
        type="propagate_in_video",
        session_id=session_id,
    )):
        outs = resp_frame["outputs"]
        frame_results.append({
            "obj_ids":    outs["out_obj_ids"],
            "probs":      outs["out_probs"],
            "boxes_xywh": outs["out_boxes_xywh"],
            "masks":      outs["out_binary_masks"],
            "num_tracked": outs["frame_stats"]["num_obj_tracked"],
        })

    predictor.handle_request(dict(
        type="close_session",
        session_id=session_id,
    ))
    return frame_results


# ──────────────────────────────────────────────────────────────────────────────
# Monotonic baseline algorithm
# ──────────────────────────────────────────────────────────────────────────────

def _mean_centroid(frame_result) -> tuple[float, float] | None:
    """Return the mean (cx, cy) of all detected bottles in a frame, normalised 0-1.
    Returns None if no bottles detected."""
    boxes = frame_result["boxes_xywh"]
    if len(boxes) == 0:
        return None
    cx = float(np.mean([b[0] + b[2] / 2 for b in boxes]))
    cy = float(np.mean([b[1] + b[3] / 2 for b in boxes]))
    return (cx, cy)


def compute_monotonic_baseline(
    raw_counts: list[int],
    frame_results: list[dict],
    initial_count: int,
    gap_max: int,
    pos_thr: float = 0.15,
) -> tuple[list[int], list[bool], list[bool]]:
    """Compute committed count, suppression flags, and hallucination flags.

    Rules:
      - committed count starts at initial_count and can only decrease.

      - Brief COUNT DECREASE (raw < committed, ≤ gap_max frames, recovers AND
          bottles reappear within pos_thr of their pre-gap position):
          → suppressed as *occlusion* (robot arm temporarily covers a bottle).
          If the gap is short but the position changed significantly (pos_thr
          exceeded), the gap is treated as a committed placement instead —
          preventing 1→0→1 hallucinations from being masked as occlusions.

      - Long / permanent COUNT DECREASE (raw < committed, > gap_max frames or
          never recovers, OR position change > pos_thr):
          → *committed placement*: committed drops to the new level.

      - Brief COUNT INCREASE (raw > committed, ≤ gap_max frames, returns to
          committed):
          → suppressed as *SAM3 artifact* (fast-moving bottle briefly detected
            as two bottles). No position check — we always want to suppress
            these short detection glitches.

      - Persistent COUNT INCREASE (raw > committed, > gap_max frames or never
          returns to committed):
          → *hallucination* (a previously placed bottle has impossibly reappeared).

    Args:
        raw_counts:    per-frame bottle count from SAM3
        frame_results: SAM3 per-frame result dicts (for centroid computation)
        initial_count: committed count at frame 0
        gap_max:       max frames for suppression (both directions)
        pos_thr:       max normalised L∞ centroid shift to allow decrease suppression
                       (set to 1.0 to disable position check)

    Returns:
        committed_arr (list[int]):  committed count at each frame
        suppressed    (list[bool]): True = brief transient (occlusion OR SAM3 artifact)
        is_hall       (list[bool]): True = persistent hallucination
    """
    n = len(raw_counts)
    committed_arr = [initial_count] * n
    suppressed    = [False] * n
    is_hall       = [False] * n

    cur_committed = initial_count
    i = 0

    while i < n:
        raw = raw_counts[i]

        if raw > cur_committed:
            # Count spike above committed — classify as artifact or real hallucination
            gap_start = i
            j = i
            while j < n and raw_counts[j] > cur_committed:
                j += 1
            gap_end = j
            gap_len = gap_end - gap_start

            if gap_len <= gap_max and gap_end < n:
                # Brief spike that returns to committed → SAM3 artifact, suppress
                # (no position check needed — detection glitches don't move objects)
                for k in range(gap_start, gap_end):
                    suppressed[k]    = True
                    committed_arr[k] = cur_committed
                i = gap_end
            else:
                # Persistent increase → real hallucination
                is_hall[i]       = True
                committed_arr[i] = cur_committed
                i += 1

        elif raw == cur_committed:
            committed_arr[i] = cur_committed
            i += 1

        else:
            # raw < cur_committed: decrease — classify as occlusion or committed placement
            gap_start = i
            j = i
            while j < n and raw_counts[j] < cur_committed:
                j += 1
            gap_end = j
            gap_len = gap_end - gap_start

            # Check both duration AND position to decide if this is a real occlusion
            is_occlusion = False
            if gap_len <= gap_max and gap_end < n:
                # Duration is short enough — now check position
                if pos_thr >= 1.0:
                    # Position check disabled
                    is_occlusion = True
                else:
                    c_before = _mean_centroid(frame_results[gap_start - 1]) if gap_start > 0 else None
                    c_after  = _mean_centroid(frame_results[gap_end])
                    if c_before is not None and c_after is not None:
                        dist = max(abs(c_after[0] - c_before[0]),
                                   abs(c_after[1] - c_before[1]))
                        is_occlusion = dist <= pos_thr
                    else:
                        # Can't compute position (no detections) — be conservative,
                        # treat as occlusion only if gap is very short (≤ 2 frames)
                        is_occlusion = gap_len <= 2

            if is_occlusion:
                for k in range(gap_start, gap_end):
                    suppressed[k]    = True
                    committed_arr[k] = cur_committed
                i = gap_end
            else:
                # Long / position-shifted gap → committed placement
                new_committed = raw_counts[gap_start]
                cur_committed = new_committed
                committed_arr[i] = cur_committed
                i += 1

    return committed_arr, suppressed, is_hall


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame annotation
# ──────────────────────────────────────────────────────────────────────────────

def draw_frame_bottles(
    frame_bgr: np.ndarray,
    frame_result: dict,
    frame_idx: int,
    committed: int,
    suppressed_flag: bool,
    is_hall_flag: bool,
    total_frames: int,
) -> np.ndarray:
    """Draw SAM3 masks/boxes and a coloured status bar onto a single frame."""
    ann = frame_bgr.copy()
    h, w = ann.shape[:2]
    color = BOTTLE_COLOR_BGR
    n = len(frame_result["obj_ids"])

    # Draw masks (semi-transparent overlay)
    for i in range(n):
        mask = frame_result["masks"][i]
        overlay = ann.copy()
        overlay[mask] = color
        ann = cv2.addWeighted(overlay, 0.35, ann, 0.65, 0)

    # Draw bounding boxes
    for i in range(n):
        bx, by, bw, bh = frame_result["boxes_xywh"][i]
        x0, y0 = int(bx * w), int(by * h)
        x1, y1 = int((bx + bw) * w), int((by + bh) * h)
        cv2.rectangle(ann, (x0, y0), (x1, y1), color, 2)
        label = f"bottle {frame_result['probs'][i]:.2f}"
        cv2.putText(ann, label, (x0, max(y0 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Coloured status bar at the top of the frame
    raw = n
    if is_hall_flag:
        status    = f"HALL: detected {raw} > committed {committed}  (persistent)"
        bg_color  = (0, 0, 200)       # red
        txt_color = (255, 255, 255)
    elif suppressed_flag and raw > committed:
        status    = f"ARTIFACT: detected {raw} > committed {committed}  (brief, suppressed)"
        bg_color  = (180, 80, 0)      # blue (BGR) — teal/cyan
        txt_color = (255, 255, 255)
    elif suppressed_flag:
        status    = f"OCCL: detected {raw} < committed {committed}  (brief, suppressed)"
        bg_color  = (0, 180, 180)     # yellow
        txt_color = (0, 0, 0)
    else:
        status    = f"OK: detected {raw}  (committed {committed})"
        bg_color  = (30, 150, 30)     # green
        txt_color = (255, 255, 255)

    cv2.rectangle(ann, (0, 0), (w, 32), bg_color, -1)
    cv2.putText(ann, status, (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, txt_color, 2)

    # Frame counter (bottom-right)
    cv2.putText(ann, f"frame {frame_idx}/{total_frames}",
                (w - 190, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)
    return ann


# ──────────────────────────────────────────────────────────────────────────────
# Main processing function
# ──────────────────────────────────────────────────────────────────────────────

def process_video(
    input_path: str,
    output_video: str,
    output_csv: str | None = None,
    output_json: str | None = None,
    prompt: str = "bottle",
    initial_count: int | None = None,   # None → auto-detect from first frames
    gap_max: int = 5,                    # max frames for suppression (both directions)
    pos_thr: float = 0.15,              # max normalised L∞ centroid shift for decrease suppression
    crop_top_ratio: float = 2 / 3,
    predictor=None,                      # reuse existing SAM3 predictor if provided
    quiet: bool = False,
) -> dict:
    """Process a single video for bottle hallucination detection. Returns summary dict."""
    from sam3.model_builder import build_sam3_video_predictor
    import shutil
    import logging

    def _log(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    own_predictor = predictor is None

    # Read video metadata
    cap = cv2.VideoCapture(input_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_full       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    crop_h = None
    if crop_top_ratio < 1.0:
        crop_h = (int(h_full * crop_top_ratio)) // 16 * 16
    h = crop_h if crop_h else h_full

    _log(f"Video: {total_frames} frames, {w}x{h} @ {fps:.1f} fps")
    _log(f"Prompt: '{prompt}'  |  gap_max={gap_max}  |  pos_thr={pos_thr}  |  crop_top_ratio={crop_top_ratio:.3f}")

    # Extract frames to JPEG temp dir (with optional crop)
    _log("Extracting frames...")
    jpeg_dir = extract_frames_to_jpeg(input_path, crop_h=crop_h)
    _log(f"  Saved to {jpeg_dir}")

    # Build predictor if caller didn't provide one
    if own_predictor:
        _log("Loading SAM3 video predictor...")
        predictor = build_sam3_video_predictor()

    # Track the prompt, suppressing SAM3 output in quiet mode
    _quiet_loggers = []
    if quiet:
        os.environ["TQDM_DISABLE"] = "1"
        for name in list(logging.Logger.manager.loggerDict):
            if name.startswith("sam3"):
                lg = logging.getLogger(name)
                _quiet_loggers.append((lg, lg.level))
                lg.setLevel(logging.ERROR)
    try:
        _log(f"Tracking: '{prompt}'...")
        frame_results = track_prompt(predictor, jpeg_dir, prompt)
        _log(f"  Done. {len(frame_results)} frames tracked.")
    finally:
        if quiet:
            os.environ.pop("TQDM_DISABLE", None)
            for lg, prev_level in _quiet_loggers:
                lg.setLevel(prev_level)

    if own_predictor:
        predictor.shutdown()

    # Raw count sequence
    raw_counts = [len(r["obj_ids"]) for r in frame_results]

    # Auto-detect initial committed count from early frames
    if initial_count is None:
        warmup = min(10, len(raw_counts))
        initial_count = max(raw_counts[:warmup]) if raw_counts else 0
        _log(f"Auto-detected initial count: {initial_count}")
    else:
        _log(f"Using specified initial count: {initial_count}")

    # Compute monotonic baseline
    committed_arr, suppressed_arr, is_hall_arr = compute_monotonic_baseline(
        raw_counts, frame_results, initial_count, gap_max, pos_thr
    )

    # ── Render annotated video ────────────────────────────────────────────────
    _log("Rendering annotated video...")
    out_frames = []
    rows       = []

    for fi in range(total_frames):
        frame_bgr = cv2.imread(os.path.join(jpeg_dir, f"{fi:06d}.jpg"))
        ann = draw_frame_bottles(
            frame_bgr,
            frame_results[fi],
            fi,
            committed     = committed_arr[fi],
            suppressed_flag = suppressed_arr[fi],
            is_hall_flag    = is_hall_arr[fi],
            total_frames    = total_frames,
        )
        out_frames.append(ann)
        # classify suppressed frames as artifact (increase) vs occlusion (decrease)
        sup = suppressed_arr[fi]
        sup_artifact = sup and (raw_counts[fi] > committed_arr[fi])
        sup_occlusion = sup and (raw_counts[fi] < committed_arr[fi])
        rows.append({
            "frame_idx":            fi,
            "raw_count":            raw_counts[fi],
            "committed_count":      committed_arr[fi],
            "suppressed_occlusion": sup_occlusion,
            "suppressed_artifact":  sup_artifact,
            "hallucination":        is_hall_arr[fi],
        })

    # Save annotated video
    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
    _log(f"Saving video ({len(out_frames)} frames)...")
    save_video_libx264(out_frames, output_video, fps)
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
    n_hall           = sum(1 for r in rows if r["hallucination"])
    n_sup_occlusion  = sum(1 for r in rows if r["suppressed_occlusion"])
    n_sup_artifact   = sum(1 for r in rows if r["suppressed_artifact"])
    final_committed  = committed_arr[-1] if committed_arr else initial_count

    summary = {
        "input_video":                  input_path,
        "output_video":                 output_video,
        "frame_count":                  len(rows),
        "fps":                          fps,
        "prompt":                       prompt,
        "initial_count":                initial_count,
        "final_committed_count":        final_committed,
        "gap_max_frames":               gap_max,
        "pos_thr":                      pos_thr,
        "total_hallucination_frames":   n_hall,
        "total_suppressed_occlusion":   n_sup_occlusion,
        "total_suppressed_artifact":    n_sup_artifact,
        "is_hallucinated":              n_hall > 0,
        "raw_count_series":             raw_counts,
        "committed_count_series":       committed_arr,
    }

    # Save JSON (optional)
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    _log("Done.")
    _log(f"  Hallucination frames   : {n_hall}/{len(rows)} ({100*n_hall/max(len(rows),1):.1f}%)")
    _log(f"  Suppressed (occlusion) : {n_sup_occlusion} frames")
    _log(f"  Suppressed (artifact)  : {n_sup_artifact} frames")
    _log(f"  Initial committed={initial_count}  →  Final committed={final_committed}")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect bottle hallucinations in 'put bottles in dustbin' videos using SAM3")
    parser.add_argument("--input", required=True, type=str,
                        help="Input video path")
    parser.add_argument("--out-dir", default=None, type=str,
                        help="Output directory (default: hallucination_outputs/<stem>)")
    parser.add_argument("--prompt", default="bottle", type=str,
                        help="Text prompt for SAM3 (default: 'bottle')")
    parser.add_argument("--initial-count", type=int, default=None,
                        help="Expected initial bottle count. Default: auto-detect from "
                             "first 10 frames (max detected).")
    parser.add_argument("--gap-max", type=int, default=5,
                        help="Max consecutive frames for suppression in either direction "
                             "(default: 5). Set to 0 to disable.")
    parser.add_argument("--pos-thr", type=float, default=0.15,
                        help="Max normalised L∞ centroid shift to suppress a brief decrease "
                             "as occlusion (default: 0.15). Set to 1.0 to disable position check.")
    parser.add_argument("--crop-top-ratio", type=float, default=2 / 3,
                        help="Fraction of frame height to keep from top "
                             "(default 2/3 to crop wrist-camera views)")
    args = parser.parse_args()

    video_stem = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "hallucination_outputs", video_stem)
    os.makedirs(out_dir, exist_ok=True)

    process_video(
        input_path     = args.input,
        output_video   = os.path.join(out_dir, f"{video_stem}_hall_bottles.mp4"),
        output_csv     = os.path.join(out_dir, f"{video_stem}_detections.csv"),
        output_json    = os.path.join(out_dir, f"{video_stem}_summary.json"),
        prompt         = args.prompt,
        initial_count  = args.initial_count,
        gap_max        = args.gap_max,
        pos_thr        = args.pos_thr,
        crop_top_ratio = args.crop_top_ratio,
    )


if __name__ == "__main__":
    main()
