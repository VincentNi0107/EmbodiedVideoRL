"""Core processing logic for constant-count hallucination detection.

Extracted from detect_hallucination.py so reward scorers can import without
depending on root-level scripts.
"""

import csv
import json
import logging
import os
import shutil

import cv2
import numpy as np

from fastvideo.reward.sam3_utils import (
    PALETTE,
    extract_frames_to_jpeg,
    save_video_libx264,
    track_prompt,
)


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
    duplication_spike_max: int = 0,
    quiet: bool = False,
    skip_render: bool = False,
    frames_dir: str | None = None,
) -> dict:
    """Process a single video. Returns a summary dict.

    If *predictor* is provided it is reused (caller manages lifecycle).
    Otherwise a new one is built and shut down within this call.
    If *quiet* is True, suppress all print output and SAM3 progress bars / logs.
    If *skip_render* is True, skip annotated video rendering (saves ~10-15s
    per video).  Hallucination statistics are still computed.
    If *frames_dir* is provided, use the pre-extracted (and optionally
    pre-cropped) JPEG frames instead of decoding from *input_path*.
    The caller retains ownership — this function will NOT delete *frames_dir*.
    """
    from sam3.model_builder import build_sam3_video_predictor

    def _log(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    own_predictor = predictor is None
    own_jpeg = frames_dir is None  # True → we extract frames and must clean up

    if frames_dir is not None:
        # Frames already extracted (and cropped) by caller — infer metadata
        jpeg_dir = frames_dir
        jpeg_files = sorted(f for f in os.listdir(jpeg_dir) if f.endswith(".jpg"))
        total_frames = len(jpeg_files)
        sample = cv2.imread(os.path.join(jpeg_dir, jpeg_files[0]))
        h, w = sample.shape[:2]
        fps = 16.0  # default; only matters for debug video rendering
        crop_h = None  # already cropped
    else:
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

        # Extract frames to JPEG dir (with optional crop)
        _log("Extracting frames...")
        jpeg_dir = extract_frames_to_jpeg(input_path, crop_h=crop_h)
        _log(f"  Frames saved to {jpeg_dir}")

    _log(f"Video: {total_frames} frames, {w}x{h} @ {fps:.1f} fps")
    _log(f"Prompts: {prompts}")
    _log(f"Expected counts: {expected_counts}")

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

    # ── Post-process: suppress brief duplication spikes (count > expected) ────
    if duplication_spike_max > 0:
        for prompt in prompts:
            n_frames = len(all_results[prompt])
            expected = expected_counts.get(prompt, 1)
            sup = suppressed[prompt]

            fi = 0
            while fi < n_frames:
                count_fi = len(all_results[prompt][fi]["obj_ids"])
                if count_fi > expected and not sup[fi]:
                    spike_start = fi
                    spike_end = fi + 1
                    while spike_end < n_frames and len(all_results[prompt][spike_end]["obj_ids"]) > expected:
                        spike_end += 1
                    spike_len = spike_end - spike_start
                    if spike_len <= duplication_spike_max:
                        for sfi in range(spike_start, spike_end):
                            sup[sfi] = True
                    fi = spike_end
                else:
                    fi += 1

    # ── Build per-frame stats (always) and render annotated video (optional) ──
    rows = []
    hallucination_events = []
    out_frames = [] if not skip_render else None

    if not skip_render:
        _log("Rendering annotated video...")

    for fi in range(total_frames):
        # Render annotated frame only when needed
        if not skip_render:
            frame_bgr = cv2.imread(os.path.join(jpeg_dir, f"{fi:06d}.jpg"))
            ann = draw_frame(frame_bgr, prompts, all_results, fi,
                             expected_counts, total_frames)
            out_frames.append(ann)

        # Build per-frame row (always — needed for hallucination summary)
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

            expected_val = expected_counts.get(prompt, 1)
            if count != expected_val and not suppressed[prompt][fi]:
                is_hall = True
                hallucination_events.append({
                    "frame_idx": fi,
                    "prompt": prompt,
                    "expected": expected_val,
                    "detected": count,
                    "type": "disappeared" if count < expected_val else "duplicated",
                })
        row["hallucination"] = is_hall
        rows.append(row)

    # Save annotated video (skip when skip_render=True)
    if not skip_render and output_video:
        os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
        _log(f"Saving video ({len(out_frames)} frames)...")
        save_video_libx264(out_frames, output_video, fps)

    # Cleanup temp frames (only if we created them)
    if own_jpeg:
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
        "duplication_spike_suppression": {
            "max_frames": duplication_spike_max,
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
