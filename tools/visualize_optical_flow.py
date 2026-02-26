"""
Optical flow visualization: reads a video, computes dense optical flow
between consecutive frames using Farneback method, and saves a color-coded
flow visualization video.

Usage:
    conda run -n vidar python visualize_optical_flow.py \
        --input /path/to/input.mp4 \
        --output /path/to/output_flow.mp4 \
        --method farneback
"""

import argparse
import os
import cv2
import numpy as np


def compute_flow_farneback(prev_gray, curr_gray):
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )


def flow_to_rgb(flow):
    """Convert optical flow (H, W, 2) to an RGB image using HSV color wheel."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2          # hue  = direction
    hsv[..., 1] = 255                              # full saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # value = magnitude
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def visualize_optical_flow(input_path, output_path, method="farneback", side_by_side=True):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = w * 2 if side_by_side else w
    out_h = h

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read the first frame")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_idx = 0
    print(f"Processing {total} frames ({w}x{h} @ {fps:.1f} fps) ...")

    if method == "farneback":
        compute_flow = compute_flow_farneback
    else:
        raise ValueError(f"Unknown method: {method}")

    # First frame: zero flow
    flow_rgb = np.zeros_like(prev_frame)
    if side_by_side:
        combined = np.hstack([prev_frame, flow_rgb])
    else:
        combined = flow_rgb
    writer.write(combined)
    frame_idx += 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = compute_flow(prev_gray, curr_gray)
        flow_rgb = flow_to_rgb(flow)

        if side_by_side:
            combined = np.hstack([frame, flow_rgb])
        else:
            combined = flow_rgb
        writer.write(combined)

        prev_gray = curr_gray
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{total} frames done")

    cap.release()
    writer.release()
    print(f"Done. Saved optical flow video to: {output_path}")
    print(f"  Total frames written: {frame_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optical flow visualization")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default=None, help="Path to output video")
    parser.add_argument("--method", type=str, default="farneback",
                        choices=["farneback"], help="Optical flow method")
    parser.add_argument("--flow-only", action="store_true",
                        help="Output only flow visualization (no side-by-side)")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_optflow{ext}"

    visualize_optical_flow(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        side_by_side=not args.flow_only,
    )
