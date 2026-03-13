"""Shared SAM3 video tracking utilities for reward scorers.

Provides:
  - extract_frames_to_jpeg: video → JPEG folder (SAM3 input format)
  - track_prompt: run SAM3 text-prompt tracking on a JPEG folder
  - save_video_libx264: BGR frame list → libx264 mp4 (VSCode-compatible)
  - compute_motion_metrics: per-object trajectory motion metrics
  - compute_motion_score_from_objects: aggregate motion_score for best-of-N
"""

import math
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


def tensor_to_jpeg_dir(video_tensor: torch.Tensor,
                       crop_h: int | None = None) -> str:
    """Convert a video tensor directly to a JPEG directory (SAM3 input format).

    This bypasses the mp4 encode→decode roundtrip for ~5-7s savings per video.

    Args:
        video_tensor: shape (C, F, H, W), values in [-1, 1], RGB order.
        crop_h: if set, keep only the top *crop_h* pixel rows of each frame.

    Returns:
        Temp directory path containing ``{000000.jpg, 000001.jpg, ...}``.
    """
    tmpdir = tempfile.mkdtemp(prefix="sam3_frames_")
    # (C, F, H, W) → (F, H, W, C), float → uint8
    frames = video_tensor.detach().float().clamp(-1, 1)
    frames = ((frames + 1.0) * 127.5).to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # (F, H, W, C) RGB

    for idx in range(frames.shape[0]):
        frame_rgb = frames[idx]
        if crop_h is not None:
            frame_rgb = frame_rgb[:crop_h]
        # RGB → BGR for cv2.imwrite
        frame_bgr = frame_rgb[:, :, ::-1]
        cv2.imwrite(os.path.join(tmpdir, f"{idx:06d}.jpg"), frame_bgr)

    return tmpdir


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


# ── Motion metrics for best-of-N selection ───────────────────────


def compute_motion_metrics(traj: list[dict]) -> dict:
    """Compute motion metrics for a single object trajectory.

    Args:
        traj: list of dicts, each with at least {frame, cx, cy}.

    Returns:
        dict with: traj_length, avg_speed, max_speed, max_speed_jump, num_points.
    """
    if len(traj) < 2:
        return {
            "traj_length": 0.0,
            "avg_speed": 0.0,
            "max_speed": 0.0,
            "max_speed_jump": 0.0,
            "num_points": len(traj),
        }

    velocities = []  # (vx, vy) per step
    speeds = []
    total_length = 0.0
    for i in range(1, len(traj)):
        dx = traj[i]["cx"] - traj[i - 1]["cx"]
        dy = traj[i]["cy"] - traj[i - 1]["cy"]
        dist = math.sqrt(dx * dx + dy * dy)
        frame_gap = max(traj[i]["frame"] - traj[i - 1]["frame"], 1)
        vx = dx / frame_gap
        vy = dy / frame_gap
        velocities.append((vx, vy))
        speeds.append(dist / frame_gap)
        total_length += dist

    # Speed jumps as vector difference norm (captures direction changes)
    speed_jumps = []
    for i in range(1, len(velocities)):
        dvx = velocities[i][0] - velocities[i - 1][0]
        dvy = velocities[i][1] - velocities[i - 1][1]
        speed_jumps.append(math.sqrt(dvx * dvx + dvy * dvy))

    span = traj[-1]["frame"] - traj[0]["frame"]
    return {
        "traj_length": total_length,
        "avg_speed": total_length / span if span > 0 else 0.0,
        "max_speed": max(speeds),
        "max_speed_jump": max(speed_jumps) if speed_jumps else 0.0,
        "num_points": len(traj),
    }


def compute_motion_score_from_objects(objects: dict) -> float:
    """Compute aggregate motion_score across all tracked objects.

    Args:
        objects: {obj_id_str: [{frame, cx, cy, ...}, ...]}
                 Same format as returned by _extract_trajectories().

    Returns:
        motion_score = total_traj_length + max_max_speed.
        Lower is better (less jitter / teleportation).
        Returns float('inf') if no valid trajectories.
    """
    # Filter real objects (first appearance <= frame 5)
    real = {oid: t for oid, t in objects.items() if t and t[0]["frame"] <= 5}
    if not real:
        return float("inf")

    metrics = [compute_motion_metrics(t) for t in real.values()]
    total_traj_length = sum(m["traj_length"] for m in metrics)
    max_max_speed = max(m["max_speed"] for m in metrics)
    return total_traj_length + max_max_speed
