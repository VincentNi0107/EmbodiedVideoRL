#!/usr/bin/env python3
"""Extract 6 frames from each rollout video, create grid images for analysis.

For each video:
  - Extract 6 evenly-spaced frames (including first and last)
  - Create a 2x3 grid image with frame indices labeled
  - Save as PNG for visual inspection

Usage:
    python extract_frames.py \
        --video_dir data/outputs/rollout_robotwin_121 \
        --dataset_json /home/omz1504/code/vidar/data/test/robotwin_121.json \
        --output_dir data/outputs/rollout_robotwin_121_frames
"""

import argparse
import json
import os
import textwrap
from pathlib import Path

import cv2
import numpy as np


def extract_frames(video_path, num_frames=6):
    """Extract evenly-spaced frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    # Pick evenly spaced indices including first and last
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = [int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]

    frames = {}
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frames[idx] = frame
        idx += 1
    cap.release()

    return [frames[i] for i in indices if i in frames]


def create_grid(frames, cols=3, frame_labels=None):
    """Create a grid image from a list of frames.

    Args:
        frames: list of BGR numpy arrays
        cols: number of columns (rows = ceil(len/cols))
        frame_labels: optional list of label strings for each frame

    Returns:
        grid BGR numpy array
    """
    if not frames:
        return None

    rows = (len(frames) + cols - 1) // cols
    # Resize all frames to same size
    h, w = frames[0].shape[:2]
    # Target: 480p per cell for readability
    target_w, target_h = 480, int(480 * h / w)

    cells = []
    for i, frame in enumerate(frames):
        cell = cv2.resize(frame, (target_w, target_h))
        # Add frame label
        if frame_labels and i < len(frame_labels):
            label = frame_labels[i]
            cv2.putText(cell, label, (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                        cv2.LINE_AA)
        cells.append(cell)

    # Pad with black cells if needed
    while len(cells) < rows * cols:
        cells.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    # Build grid
    row_imgs = []
    for r in range(rows):
        row_imgs.append(np.concatenate(cells[r * cols:(r + 1) * cols], axis=1))
    grid = np.concatenate(row_imgs, axis=0)
    return grid


def add_text_banner(grid, text, max_width_chars=120, font_scale=0.6,
                    line_height=24, padding=12):
    """Add a text banner above the grid image."""
    # Word wrap
    lines = textwrap.wrap(text, width=max_width_chars)
    banner_h = padding * 2 + line_height * len(lines)
    banner = np.zeros((banner_h, grid.shape[1], 3), dtype=np.uint8)

    for i, line in enumerate(lines):
        y = padding + line_height * (i + 1)
        cv2.putText(banner, line, (padding, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return np.concatenate([banner, grid], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing rollout video subdirectories")
    parser.add_argument("--dataset_json", type=str, required=True,
                        help="Path to the dataset JSON for prompt lookup")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for frame grid images")
    parser.add_argument("--num_frames", type=int, default=6,
                        help="Number of frames to extract per video")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset for prompt lookup
    with open(args.dataset_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Build prompt map: filename_stem -> prompt (extract task description only)
    prompt_map = {}
    for item in dataset:
        stem = item["filename_stem"]
        prompt = item["prompt"]
        # Extract the task part after "performing the following task: "
        task_marker = "performing the following task: "
        if task_marker in prompt:
            task_desc = prompt[prompt.index(task_marker) + len(task_marker):]
        else:
            task_desc = prompt
        prompt_map[stem] = task_desc

    # Find all video files
    all_videos = sorted(video_dir.rglob("*.mp4"))
    print(f"Found {len(all_videos)} videos in {video_dir}")

    if not all_videos:
        print("No videos found!")
        return

    # Process each video
    for vid_path in all_videos:
        # Determine filename_stem from parent directory name
        sample_stem = vid_path.parent.name
        task_desc = prompt_map.get(sample_stem, "Unknown task")

        # Create output subdirectory per sample
        sample_out_dir = output_dir / sample_stem
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames
        frames = extract_frames(vid_path, num_frames=args.num_frames)
        if not frames:
            print(f"  SKIP (no frames): {vid_path.name}")
            continue

        # Create frame labels (frame index in video)
        cap = cv2.VideoCapture(str(vid_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total <= args.num_frames:
            indices = list(range(total))
        else:
            indices = [int(i * (total - 1) / (args.num_frames - 1))
                       for i in range(args.num_frames)]
        labels = [f"f{idx}" for idx in indices]

        # Build grid
        grid = create_grid(frames, cols=3, frame_labels=labels)
        if grid is None:
            continue

        # Add task description as banner
        banner_text = f"[{sample_stem}] Task: {task_desc}"
        grid_with_text = add_text_banner(grid, banner_text)

        # Save
        out_path = sample_out_dir / f"{vid_path.stem}.png"
        cv2.imwrite(str(out_path), grid_with_text)
        print(f"  saved: {out_path}")

    # Also create per-sample summary grids (1 frame from each rollout)
    print("\nCreating per-sample summary grids...")
    summary_dir = output_dir / "_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])
    for sample_dir in sample_dirs:
        sample_stem = sample_dir.name
        task_desc = prompt_map.get(sample_stem, "Unknown task")
        videos = sorted(sample_dir.glob("*.mp4"))

        if not videos:
            continue

        # Extract middle frame from each rollout
        mid_frames = []
        rollout_labels = []
        for vid_path in videos:
            cap = cv2.VideoCapture(str(vid_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mid_idx = total // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                mid_frames.append(frame)
                # Extract seed from filename
                seed_part = vid_path.stem.split("_s")[-1] if "_s" in vid_path.stem else "?"
                g_part = vid_path.stem.split("_g")[-1].split("_")[0] if "_g" in vid_path.stem else "?"
                rollout_labels.append(f"g{g_part} s{seed_part}")

        if mid_frames:
            # 2x4 grid for 8 rollouts
            grid = create_grid(mid_frames, cols=4, frame_labels=rollout_labels)
            if grid is not None:
                banner = f"[{sample_stem}] Task: {task_desc} | Mid-frame from each rollout"
                grid_with_text = add_text_banner(grid, banner)
                out_path = summary_dir / f"{sample_stem}_summary.png"
                cv2.imwrite(str(out_path), grid_with_text)
                print(f"  summary: {out_path}")

    print(f"\nDone! Grids saved to {output_dir}")


if __name__ == "__main__":
    main()
