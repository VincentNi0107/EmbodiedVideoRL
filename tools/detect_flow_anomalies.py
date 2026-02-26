"""
Detect optical-flow anomalies in generated manipulation videos.

This script is designed for cases such as:
1) Object suddenly drifting away from the dominant scene motion.
2) Object appearing/disappearing with large instantaneous motion spikes.

Outputs:
- An annotated mp4 with anomaly overlays and boxes.
- Per-frame scores in CSV.
- A JSON summary with peak anomaly frames.
"""

import argparse
import csv
import json
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch

# wan.utils.utils.save_video for libx264 output
from fastvideo.models.wan.utils.utils import save_video as _wan_save_video


def save_video_libx264(frames_bgr: list, output_path: str, fps: float) -> None:
    """Save a list of BGR uint8 numpy frames using wan's save_video (libx264)."""
    # Convert BGR -> RGB, stack to (T, H, W, C), then to (1, C, T, H, W) float in (-1,1)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    tensor = torch.from_numpy(np.stack(rgb, axis=0))          # (T, H, W, C) uint8
    tensor = tensor.float() / 127.5 - 1.0                     # (-1, 1)
    tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)          # (1, C, T, H, W)
    _wan_save_video(tensor, save_file=output_path, fps=fps, normalize=True, value_range=(-1, 1))


def compute_flow_farneback(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def flow_to_bgr(flow: np.ndarray, mag_clip: float | None = None) -> np.ndarray:
    """HSV color wheel: hue = direction, value = magnitude."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255

    if mag_clip is None:
        val = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    else:
        val = np.clip(mag / max(mag_clip, 1e-6), 0.0, 1.0) * 255.0
    hsv[..., 2] = val.astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Draw a small color-wheel legend in the bottom-right corner.
    legend_r = 28
    cx = bgr.shape[1] - legend_r - 6
    cy = bgr.shape[0] - legend_r - 6
    for dy in range(-legend_r, legend_r + 1):
        for dx in range(-legend_r, legend_r + 1):
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > legend_r:
                continue
            ang_px = float(np.arctan2(dy, dx))
            hue = int((ang_px % (2 * np.pi)) / (2 * np.pi) * 180)
            val_px = int(min(dist / legend_r, 1.0) * 255)
            px_hsv = np.array([[[hue, 255, val_px]]], dtype=np.uint8)
            px_bgr = cv2.cvtColor(px_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            py, px = cy + dy, cx + dx
            if 0 <= py < bgr.shape[0] and 0 <= px < bgr.shape[1]:
                bgr[py, px] = px_bgr
    cv2.circle(bgr, (cx, cy), legend_r, (200, 200, 200), 1)
    cv2.putText(bgr, "dir", (cx - 10, cy + legend_r + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return bgr


def flow_to_magnitude_heatmap(mag: np.ndarray, mag_clip: float | None = None) -> np.ndarray:
    """Grayscale->COLORMAP_JET heatmap of flow magnitude. Brighter = faster."""
    if mag_clip is None:
        mag_clip = float(np.percentile(mag, 99.0)) if mag.size > 0 else 1.0
    norm = np.clip(mag / max(mag_clip, 1e-6), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    # Add scale bar text
    cv2.putText(heatmap, f"mag (0~{mag_clip:.1f}px)", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return heatmap


def flow_to_quiver(
    flow: np.ndarray,
    bg: np.ndarray,
    stride: int = 20,
    scale: float = 3.0,
    anomaly_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Draw sparse flow arrows on a darkened copy of bg.

    Normal arrows: green. Anomalous arrows (if mask provided): red.
    Arrow length = magnitude * scale pixels.
    """
    canvas = (bg.astype(np.float32) * 0.4).astype(np.uint8)
    h, w = flow.shape[:2]
    for y in range(stride // 2, h, stride):
        for x in range(stride // 2, w, stride):
            fx, fy = float(flow[y, x, 0]), float(flow[y, x, 1])
            mag_px = (fx * fx + fy * fy) ** 0.5
            if mag_px < 0.3:          # skip near-zero
                continue
            ex = int(x + fx * scale)
            ey = int(y + fy * scale)
            is_anomaly = anomaly_mask is not None and anomaly_mask[y, x] > 0
            color = (0, 0, 220) if is_anomaly else (0, 200, 60)
            cv2.arrowedLine(canvas, (x, y), (ex, ey), color,
                            thickness=1, tipLength=0.35)
    cv2.putText(canvas, f"quiver (x{scale:.0f})", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return canvas


def make_grid_frame(ann: np.ndarray, flow_hsv: np.ndarray,
                    mag_heat: np.ndarray, quiver: np.ndarray) -> np.ndarray:
    """Stack four panels into a 2x2 grid with thin separators.
    Output dimensions are rounded down to nearest multiple of 16 for libx264.
    """
    sep = 2
    h, w = ann.shape[:2]
    top = np.concatenate([ann, np.zeros((h, sep, 3), np.uint8), flow_hsv], axis=1)
    bot = np.concatenate([mag_heat, np.zeros((h, sep, 3), np.uint8), quiver], axis=1)
    mid = np.zeros((sep, top.shape[1], 3), np.uint8)
    grid = np.concatenate([top, mid, bot], axis=0)
    # Align to 16-pixel boundary to avoid libx264 macro-block padding warning.
    gh, gw = grid.shape[:2]
    gh16, gw16 = gh // 16 * 16, gw // 16 * 16
    return grid[:gh16, :gw16]


def robust_stats(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 1e-6
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    mad = max(mad, 1e-6)
    return med, mad


def get_global_motion_vector(flow: np.ndarray, mag: np.ndarray) -> np.ndarray:
    # Use high-motion pixels to estimate dominant motion direction.
    thresh = float(np.percentile(mag, 70.0))
    sel = mag > thresh
    if np.count_nonzero(sel) < 50:
        v = flow.reshape(-1, 2).mean(axis=0)
    else:
        v = flow[sel].mean(axis=0)
    return v.astype(np.float32)


def find_anomaly_mask(
    flow: np.ndarray,
    mag: np.ndarray,
    global_v: np.ndarray,
    dir_cos_threshold: float,
    abs_mag_min: float = 1.5,
) -> tuple[np.ndarray, dict]:
    med, mad = robust_stats(mag)
    # Relative spike: must also exceed absolute minimum to avoid firing on
    # near-static frames where MAD collapses to ~0 (video compression noise).
    high_mag = (mag > (med + 3.0 * mad)) & (mag > abs_mag_min)
    # Absolute top-0.5% spike: guard with abs threshold so completely still
    # frames don't produce false positives from their own percentile.
    very_high_mag = (mag > float(np.percentile(mag, 99.5))) & (mag > abs_mag_min * 2)

    gv_norm = float(np.linalg.norm(global_v))
    if gv_norm < 1e-6:
        dir_bad = np.zeros_like(mag, dtype=bool)
    else:
        dot = flow[..., 0] * global_v[0] + flow[..., 1] * global_v[1]
        denom = (mag * gv_norm) + 1e-6
        cos_sim = dot / denom
        dir_bad = cos_sim < dir_cos_threshold

    anomaly = (high_mag & dir_bad) | very_high_mag
    anomaly_u8 = anomaly.astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    anomaly_u8 = cv2.morphologyEx(anomaly_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    anomaly_u8 = cv2.morphologyEx(anomaly_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    info = {
        "mag_median": med,
        "mag_mad": mad,
        "global_motion_px": gv_norm,
        "anomaly_area_ratio": float(np.mean(anomaly_u8 > 0)),
        "max_mag": float(np.max(mag)),
        "mean_mag": float(np.mean(mag)),
    }
    return anomaly_u8, info


def anomaly_boxes(mask_u8: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))
    return boxes


def smooth_signal(values: list[float], win: int) -> list[float]:
    if win <= 1:
        return values[:]
    out = []
    q = deque(maxlen=win)
    for v in values:
        q.append(v)
        out.append(float(np.mean(q)))
    return out


def detect_peaks(values: list[float], min_prominence: float) -> list[int]:
    # Simple local-maximum detector without external deps.
    if len(values) < 3:
        return []
    peaks = []
    arr = np.array(values, dtype=np.float32)
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] >= arr[i + 1]:
            left = arr[max(0, i - 5):i]
            right = arr[i + 1:min(len(arr), i + 6)]
            local_base = float(np.median(np.concatenate([left, right]))) if left.size + right.size > 0 else 0.0
            if arr[i] - local_base >= min_prominence:
                peaks.append(i)
    return peaks


def process_video(
    input_path: str,
    output_video: str,
    output_csv: str,
    output_json: str,
    side_by_side: bool,
    dir_cos_threshold: float,
    min_box_area: int,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Crop to top 2/3: main camera view (discard bottom wrist cameras).
    # Round down to nearest multiple of 16 to avoid libx264 macro-block padding.
    h = (h_full * 2 // 3) // 16 * 16
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Cannot read the first frame.")
    prev = prev[:h]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    out_frames: list = []

    rows = []
    raw_scores = [0.0]

    # First frame placeholder.
    first_ann = prev.copy()
    cv2.putText(first_ann, "frame=0 score=0.000", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    if side_by_side:
        blank = np.zeros_like(prev)
        first_frame = make_grid_frame(first_ann, blank, blank, blank)
    else:
        first_frame = first_ann
    out_frames.append(first_frame)
    rows.append(
        {
            "frame_idx": 0,
            "score_raw": 0.0,
            "score_smooth": 0.0,
            "anomaly_area_ratio": 0.0,
            "mean_mag": 0.0,
            "max_mag": 0.0,
            "global_motion_px": 0.0,
            "num_boxes": 0,
        }
    )

    frame_idx = 1
    print(f"Processing {total_frames} frames ({w}x{h} @ {fps:.2f} fps)")
    while True:
        ret, curr = cap.read()
        if not ret:
            break
        curr = curr[:h]

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flow = compute_flow_farneback(prev_gray, curr_gray)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        global_v = get_global_motion_vector(flow, mag)
        anomaly_mask, info = find_anomaly_mask(flow, mag, global_v, dir_cos_threshold)
        boxes = anomaly_boxes(anomaly_mask, min_box_area)

        # Score emphasizes both anomalous area and extreme motion.
        med, mad = robust_stats(mag)
        spike = max(0.0, (info["max_mag"] - (med + 4.0 * mad)) / (4.0 * mad + 1e-6))
        score_raw = float(info["anomaly_area_ratio"] * 3.0 + min(spike, 2.0))

        mag_clip_val = float(np.percentile(mag, 99.0))

        ann = curr.copy()
        overlay = ann.copy()
        overlay[anomaly_mask > 0] = (0, 0, 255)
        ann = cv2.addWeighted(overlay, 0.35, ann, 0.65, 0)

        gv_angle = float(np.degrees(np.arctan2(global_v[1], global_v[0]))) if np.linalg.norm(global_v) > 1e-6 else 0.0
        cv2.putText(
            ann,
            f"frame={frame_idx} score={score_raw:.3f} area={info['anomaly_area_ratio']:.3f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            ann,
            f"gdir={gv_angle:.0f}deg mean={info['mean_mag']:.1f}px max={info['max_mag']:.1f}px",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        if side_by_side:
            flow_hsv = flow_to_bgr(flow, mag_clip=mag_clip_val)
            mag_heat = flow_to_magnitude_heatmap(mag, mag_clip=mag_clip_val)
            quiver = flow_to_quiver(flow, curr, stride=20, scale=3.0, anomaly_mask=anomaly_mask)
            out_frame = make_grid_frame(ann, flow_hsv, mag_heat, quiver)
        else:
            out_frame = ann
        out_frames.append(out_frame)

        raw_scores.append(score_raw)
        rows.append(
            {
                "frame_idx": frame_idx,
                "score_raw": score_raw,
                "score_smooth": 0.0,  # fill later
                "anomaly_area_ratio": info["anomaly_area_ratio"],
                "mean_mag": info["mean_mag"],
                "max_mag": info["max_mag"],
                "global_motion_px": info["global_motion_px"],
                "num_boxes": len(boxes),
            }
        )

        prev_gray = curr_gray
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{total_frames} frames done")

    cap.release()

    print(f"  Saving video with libx264 ({len(out_frames)} frames)...")
    save_video_libx264(out_frames, output_video, fps)

    smooth_scores = smooth_signal(raw_scores, win=5)
    for i in range(len(rows)):
        rows[i]["score_smooth"] = smooth_scores[i]

    # Robust peak threshold from smoothed score.
    sm = np.array(smooth_scores, dtype=np.float32)
    sm_med, sm_mad = robust_stats(sm)
    peak_thresh = float(sm_med + 3.0 * sm_mad)
    peak_prom = float(max(0.05, 1.5 * sm_mad))
    peak_idxs = detect_peaks(smooth_scores, min_prominence=peak_prom)
    peak_idxs = [i for i in peak_idxs if smooth_scores[i] >= peak_thresh]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "score_raw",
                "score_smooth",
                "anomaly_area_ratio",
                "mean_mag",
                "max_mag",
                "global_motion_px",
                "num_boxes",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    summary = {
        "input_video": input_path,
        "output_video": output_video,
        "frame_count": len(rows),
        "fps": fps,
        "peak_threshold": peak_thresh,
        "peak_prominence": peak_prom,
        "peak_frames": peak_idxs,
        "top10_frames_by_score": sorted(
            [{"frame_idx": r["frame_idx"], "score_smooth": r["score_smooth"]} for r in rows],
            key=lambda x: x["score_smooth"],
            reverse=True,
        )[:10],
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"- Annotated video: {output_video}")
    print(f"- Frame scores CSV: {output_csv}")
    print(f"- Summary JSON: {output_json}")
    print(f"- Peak frames: {peak_idxs}")


def main():
    parser = argparse.ArgumentParser(description="Optical-flow anomaly detector")
    parser.add_argument("--input", required=True, type=str, help="Input video path")
    parser.add_argument("--out-dir", default=None, type=str, help="Output directory")
    parser.add_argument(
        "--dir-cos-threshold",
        default=-0.2,
        type=float,
        help="Direction mismatch threshold (lower = stricter reverse direction)",
    )
    parser.add_argument("--min-box-area", default=120, type=int, help="Min contour area for anomaly box")
    parser.add_argument("--no-side-by-side", action="store_true", help="Save only annotated frame (without flow pane)")
    args = parser.parse_args()

    video_stem = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "flow_anomaly_outputs", video_stem)
    os.makedirs(out_dir, exist_ok=True)

    output_video = os.path.join(out_dir, f"{video_stem}_anomaly.mp4")
    output_csv = os.path.join(out_dir, f"{video_stem}_scores.csv")
    output_json = os.path.join(out_dir, f"{video_stem}_summary.json")

    process_video(
        input_path=args.input,
        output_video=output_video,
        output_csv=output_csv,
        output_json=output_json,
        side_by_side=not args.no_side_by_side,
        dir_cos_threshold=args.dir_cos_threshold,
        min_box_area=args.min_box_area,
    )


if __name__ == "__main__":
    main()
