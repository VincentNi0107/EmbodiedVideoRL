"""Shared SAM3 video tracking utilities for reward scorers.

Provides:
  - extract_frames_to_jpeg: video → JPEG folder (SAM3 input format)
  - track_prompt: run SAM3 text-prompt tracking on a JPEG folder
  - save_video_libx264: BGR frame list → libx264 mp4 (VSCode-compatible)
"""

import os
import tempfile

import cv2
import numpy as np
import torch

from fastvideo.models.wan.utils.utils import save_video as _wan_save_video


# Distinct colors (BGR) for multi-prompt annotation
PALETTE = [
    (60, 60, 220),    # reddish
    (60, 200, 60),    # greenish
    (220, 140, 40),   # bluish
    (0, 200, 200),    # yellow
    (200, 0, 200),    # magenta
    (200, 200, 0),    # cyan
]


def save_video_libx264(frames_bgr: list, output_path: str, fps: float) -> None:
    """Write a list of BGR numpy frames to an mp4 file using libx264."""
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
