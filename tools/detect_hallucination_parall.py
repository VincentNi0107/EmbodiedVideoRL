"""
Detect object hallucinations in generated manipulation videos using SAM3.

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


def track_prompts_text(predictor, video_resource: str, prompts: list[str]):
    """Track via per-prompt text detection + propagation (reusing one session).

    Each text prompt triggers a full propagation pass, but the session (and its
    loaded frames) is reused across prompts.  This mode can discover objects that
    appear or disappear mid-video — best for hallucination detection.

    Returns:
        dict[str, list[dict]]: prompt -> per-frame results.
    """
    resp = predictor.handle_request(dict(
        type="start_session",
        resource_path=video_resource,
    ))
    session_id = resp["session_id"]

    all_results = {}
    for prompt in prompts:
        print(f"  [text] Tracking: '{prompt}'...")
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
                "obj_ids": outs["out_obj_ids"],
                "probs": outs["out_probs"],
                "boxes_xywh": outs["out_boxes_xywh"],
                "masks": outs["out_binary_masks"],
                "num_tracked": outs["frame_stats"]["num_obj_tracked"],
            })
        all_results[prompt] = frame_results
        print(f"    Done. {len(frame_results)} frames tracked.")

    predictor.handle_request(dict(
        type="close_session",
        session_id=session_id,
    ))
    return all_results


def track_prompts_pointinit(predictor, video_resource: str, prompts: list[str]):
    """Detect all objects on frame 0 with text, then track all via points in one pass.

    Stage 1: For each text prompt, run detection on frame 0 to get masks.
    Stage 2: Sample center point from each mask, register as point prompts
             with unique obj_ids (point prompts don't reset tracker state).
    Stage 3: Single propagate_in_video for all objects simultaneously.

    ~3x faster than text mode (1 propagation instead of N), but can only track
    objects visible in frame 0 — won't detect new duplicates appearing later.

    Returns:
        dict[str, list[dict]]: prompt -> per-frame results (same format as text mode).
    """
    resp = predictor.handle_request(dict(
        type="start_session",
        resource_path=video_resource,
    ))
    session_id = resp["session_id"]

    # --- Stage 1: detect objects per prompt on frame 0 ---
    obj_id_map = {}     # obj_id -> prompt
    obj_id_point = {}   # obj_id -> [x, y] normalised
    next_obj_id = 1     # SAM3 tracker obj_ids start from 1

    for prompt in prompts:
        print(f"  [pointinit] Detecting '{prompt}' on frame 0...")
        det_resp = predictor.handle_request(dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt,
        ))
        det = det_resp["outputs"]
        n_det = len(det["out_obj_ids"])
        for i in range(n_det):
            mask = det["out_binary_masks"][i]
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            cx = float(xs.mean()) / mask.shape[1]
            cy = float(ys.mean()) / mask.shape[0]
            obj_id_map[next_obj_id] = prompt
            obj_id_point[next_obj_id] = [cx, cy]
            next_obj_id += 1

    n_total = len(obj_id_map)
    per_prompt_counts = {}
    for oid, p in obj_id_map.items():
        per_prompt_counts[p] = per_prompt_counts.get(p, 0) + 1
    print(f"  Detected {n_total} objects: "
          + ", ".join(f"{p}={c}" for p, c in per_prompt_counts.items()))

    if n_total == 0:
        predictor.handle_request(dict(type="close_session", session_id=session_id))
        return _empty_results(prompts)

    # --- Stage 2: reset & register all objects via point prompts ---
    predictor.handle_request(dict(
        type="reset_session",
        session_id=session_id,
    ))
    for obj_id, pt in obj_id_point.items():
        predictor.handle_request(dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            points=[pt],
            point_labels=[1],
            obj_id=obj_id,
        ))

    # --- Stage 3: single propagation ---
    print(f"  [pointinit] Propagating {n_total} objects...")
    raw_frames = []
    for resp_frame in predictor.handle_stream_request(dict(
        type="propagate_in_video",
        session_id=session_id,
    )):
        raw_frames.append(resp_frame["outputs"])
    print(f"    Done. {len(raw_frames)} frames tracked.")

    predictor.handle_request(dict(
        type="close_session",
        session_id=session_id,
    ))

    # --- Stage 4: split combined outputs back into per-prompt dicts ---
    all_results = {p: [] for p in prompts}
    for outs in raw_frames:
        per_prompt = {p: {"obj_ids": [], "probs": [], "boxes": [], "masks": []}
                      for p in prompts}

        for i, oid in enumerate(outs["out_obj_ids"]):
            prompt = obj_id_map.get(int(oid))
            if prompt is None:
                continue
            d = per_prompt[prompt]
            d["obj_ids"].append(oid)
            d["probs"].append(outs["out_probs"][i])
            d["boxes"].append(outs["out_boxes_xywh"][i])
            d["masks"].append(outs["out_binary_masks"][i])

        for prompt in prompts:
            d = per_prompt[prompt]
            n = len(d["obj_ids"])
            if n > 0:
                all_results[prompt].append({
                    "obj_ids": np.array(d["obj_ids"]),
                    "probs": np.array(d["probs"]),
                    "boxes_xywh": np.stack(d["boxes"]),
                    "masks": np.stack(d["masks"]),
                    "num_tracked": n,
                })
            else:
                h_mask = outs["out_binary_masks"].shape[1] if len(outs["out_binary_masks"]) > 0 else 1
                w_mask = outs["out_binary_masks"].shape[2] if len(outs["out_binary_masks"]) > 0 else 1
                all_results[prompt].append({
                    "obj_ids": np.zeros(0, dtype=np.int64),
                    "probs": np.zeros(0, dtype=np.float32),
                    "boxes_xywh": np.zeros((0, 4), dtype=np.float32),
                    "masks": np.zeros((0, h_mask, w_mask), dtype=bool),
                    "num_tracked": 0,
                })

    return all_results


def _empty_results(prompts):
    """Return empty per-prompt results when no objects are detected."""
    return {p: [] for p in prompts}


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
    tracking_mode: str = "text",
) -> dict:
    """Process a single video. Returns a summary dict.

    If *predictor* is provided it is reused (caller manages lifecycle).
    Otherwise a new one is built and shut down within this call.

    *tracking_mode*:
        "text"      – per-prompt text detection + propagation (default, best
                       for hallucination detection: can find new duplicates).
        "pointinit" – detect on frame 0, track all via points in one pass
                       (~3x faster, but only tracks objects visible in frame 0).
    """
    from sam3.model_builder import build_sam3_video_predictor
    import shutil

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

    print(f"Video: {total_frames} frames, {w}x{h} @ {fps:.1f} fps")
    print(f"Prompts: {prompts}")
    print(f"Expected counts: {expected_counts}")

    # Extract frames to JPEG dir (with optional crop)
    print("Extracting frames...")
    jpeg_dir = extract_frames_to_jpeg(input_path, crop_h=crop_h)
    print(f"  Frames saved to {jpeg_dir}")

    # Build predictor if not provided
    if own_predictor:
        print("Loading SAM3 video predictor...")
        predictor = build_sam3_video_predictor()

    # Track all prompts
    track_fn = track_prompts_pointinit if tracking_mode == "pointinit" else track_prompts_text
    print(f"Tracking mode: {tracking_mode}")
    all_results = track_fn(predictor, jpeg_dir, prompts)

    if own_predictor:
        predictor.shutdown()

    # Re-read frames for annotation (from the JPEG dir, already cropped)
    print("Rendering annotated video...")
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

            expected = expected_counts.get(prompt, 1)
            if count != expected:
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
    print(f"Saving video ({len(out_frames)} frames)...")
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
        }

    summary = {
        "input_video": input_path,
        "output_video": output_video,
        "frame_count": len(rows),
        "fps": fps,
        "prompts": prompts,
        "expected_counts": expected_counts,
        "counts_over_time": counts_over_time,
        "total_hallucination_frames": sum(1 for r in rows if r["hallucination"]),
        "hallucination_events": hallucination_events,
    }

    # Save JSON (optional)
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("Done.")
    print(f"- Annotated video: {output_video}")
    n_hall = summary["total_hallucination_frames"]
    print(f"- Hallucination frames: {n_hall}/{len(rows)} "
          f"({100 * n_hall / max(len(rows), 1):.1f}%)")
    for prompt, stats in counts_over_time.items():
        print(f"  {prompt}: min={stats['min']} max={stats['max']} "
              f"mean={stats['mean']:.2f} "
              f"zero_frames={stats['frames_with_zero']} "
              f"multi_frames={stats['frames_with_multiple']}")

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
    parser.add_argument("--tracking-mode", choices=["text", "pointinit"],
                        default="text",
                        help="'text': per-prompt propagation (best for hallucination). "
                             "'pointinit': detect frame 0, single propagation (~3x faster).")
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
        tracking_mode=args.tracking_mode,
    )


if __name__ == "__main__":
    main()
