#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a DiffSynth-Studio video dataset from vidar-robotwin hdf5 episodes.
This script reads hdf5 directly and does NOT generate per-frame images.

Output:
  DiffSynth-Studio/data/robotwin_wan/
    ├── metadata.csv
    └── videos/*.mp4
"""
import argparse
import csv
from pathlib import Path
from typing import Tuple
import os
import shutil

import cv2
import h5py
import numpy as np
import json

import subprocess
from PIL import Image


def images_to_video(imgs: np.ndarray, out_path: str, fps: float = 30.0, is_rgb: bool = True) -> None:
    if (not isinstance(imgs, np.ndarray) or imgs.ndim != 4 or imgs.shape[3] not in (3, 4)):
        raise ValueError("imgs must be a numpy.ndarray of shape (N, H, W, C), with C equal to 3 or 4.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_frames, H, W, C = imgs.shape
    if C == 3:
        pixel_format = "rgb24" if is_rgb else "bgr24"
    else:
        pixel_format = "rgba"
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            pixel_format,
            "-video_size",
            f"{W}x{H}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            f"{out_path}",
        ],
        stdin=subprocess.PIPE,
    )
    ffmpeg.stdin.write(imgs.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise IOError("Cannot open ffmpeg. Please check the output path and ensure ffmpeg is supported.")

    print(
        f"🎬 Video is saved to `{out_path}`, containing \033[94m{n_frames}\033[0m frames at {W}×{H} resolution and {fps} FPS."
    )


def decode_image(buf: object) -> np.ndarray:
    if isinstance(buf, (bytes, bytearray)):
        arr = np.frombuffer(buf, dtype=np.uint8)
    elif isinstance(buf, np.ndarray) and buf.dtype == np.uint8:
        arr = buf
    else:
        raise TypeError(f"Unsupported buffer type: {type(buf)}")
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None; invalid image bytes")
    return img


def concat_three_views(head_img: np.ndarray, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
    orig_h, orig_w = head_img.shape[:2]
    half_h, half_w = orig_h // 2, orig_w // 2
    left_resized = cv2.resize(left_img, (half_w, half_h))
    right_resized = cv2.resize(right_img, (half_w, half_h))
    bottom_row = np.hstack([left_resized, right_resized])
    return np.vstack([head_img, bottom_row])


def resize_like_vidar(img: np.ndarray, size=(640, 736)) -> np.ndarray:
    # Match vidar server resize: PIL LANCZOS to (width, height)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(size, Image.LANCZOS)
    return np.array(pil_img)


def _format_instruction(instruction: str) -> str:
    if not instruction:
        return instruction
    return instruction[0].lower() + instruction[1:]


def load_prompt(task_name: str, task_dir: Path) -> str:
    path = task_dir / f"{task_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        instruction = json.load(f)["full_description"]
    system_prompt = (
        "The whole scene is in a realistic, industrial art style with three views: "
        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
        "The aloha robot is currently performing the following task: "
    )
    return system_prompt + _format_instruction(instruction)


def write_video(frames: list, out_path: Path, fps: int, is_rgb: bool) -> Tuple[int, int]:
    arr = np.stack(frames, axis=0)
    images_to_video(arr, out_path=str(out_path), fps=float(fps), is_rgb=is_rgb)
    h, w = frames[0].shape[:2]
    return h, w


def uniform_frame_indices(n_frames: int, target_frames: int) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if target_frames == 1:
        return np.array([0], dtype=int)
    indices = np.linspace(0, n_frames - 1, target_frames)
    indices = np.round(indices).astype(int)
    indices[0] = 0
    indices[-1] = n_frames - 1
    return indices


def load_low_success_tasks(csv_path: Path, threshold: float) -> set:
    if not csv_path.exists():
        raise FileNotFoundError(f"success_rates.csv not found: {csv_path}")
    tasks = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = (row.get("task") or "").strip()
            rate_str = (row.get("success_rate") or "").strip()
            if not task or not rate_str:
                continue
            try:
                rate = float(rate_str)
            except ValueError:
                continue
            if rate < threshold:
                tasks.add(task)
    return tasks



def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DiffSynth video dataset from vidar-robotwin hdf5.")
    parser.add_argument("--data-root", type=str, default="vidar-robotwin/data", help="vidar-robotwin data root")
    parser.add_argument("--task-instruction-dir", type=str, default="vidar-robotwin/description/task_instruction", help="task instruction json dir")
    parser.add_argument("--out-root", type=str, default="DiffSynth-Studio/data/robotwin_wan", help="output dataset root")
    parser.add_argument("--fps", type=int, default=10, help="fps for output videos")
    parser.add_argument("--task", type=str, default=None, help="only process a single task name if provided")
    parser.add_argument("--num-frames", type=int, default=61, help="number of frames to uniformly sample per video")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing videos")
    parser.add_argument("--clean_out", action="store_true", help="remove existing output videos and metadata before processing")
    parser.add_argument(
        "--success-csv",
        type=str,
        default="vidar-robotwin/eval_result/ar/single_test/success_rates.csv",
        help="csv file that contains task success rates",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.6,
        help="only process tasks with success_rate below this threshold",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    task_dir = Path(args.task_instruction_dir)
    out_root = Path(args.out_root)
    video_dir = out_root / "videos"
    if args.clean_out and out_root.exists():
        shutil.rmtree(video_dir, ignore_errors=True)
        meta_path = out_root / "metadata.csv"
        if meta_path.exists():
            meta_path.unlink()
    video_dir.mkdir(parents=True, exist_ok=True)

    rows = ["video,prompt"]

    eligible_tasks = None
    if args.task is None:
        eligible_tasks = load_low_success_tasks(Path(args.success_csv), args.success_threshold)
        if not eligible_tasks:
            print("[WARN] No tasks below success threshold; nothing to process.")
            return

    h5_files = sorted(data_root.rglob("*.hdf5"))
    for h5_path in h5_files:
        try:
            rel = h5_path.relative_to(data_root)
            task_name = rel.parts[0]
            demo_name = rel.parts[1]
            episode_name = h5_path.stem
        except Exception:
            continue
        if args.task is not None and task_name != args.task:
            continue
        if eligible_tasks is not None and task_name not in eligible_tasks:
            continue

        out_name = f"{task_name}__{demo_name}__{episode_name}.mp4"
        out_path = video_dir / out_name
        if out_path.exists() and not args.overwrite:
            rows.append(f"videos/{out_name},\"{load_prompt(task_name, task_dir)}\"")
            continue
        print(f"videos/{out_name},\"{load_prompt(task_name, task_dir)}\"")
        with h5py.File(h5_path, "r") as f:
            try:
                head = f["observation"]["head_camera"]["rgb"]
                left = f["observation"]["left_camera"]["rgb"]
                right = f["observation"]["right_camera"]["rgb"]
            except KeyError:
                continue

            length = len(head)
            if len(left) != length or len(right) != length:
                continue

            frames = []
            for i in range(length):
                head_img = decode_image(head[i])
                left_img = decode_image(left[i])
                right_img = decode_image(right[i])
                combined = concat_three_views(head_img, left_img, right_img)
                combined = resize_like_vidar(combined, size=(640, 736))
                frames.append(combined)

        if not frames:
            continue
        indices = uniform_frame_indices(len(frames), args.num_frames)
        frames = [frames[i] for i in indices]
        write_video(frames, out_path, args.fps, is_rgb=True)
        rows.append(f"videos/{out_name},\"{load_prompt(task_name, task_dir)}\"")

    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "metadata.csv"
    meta_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"[OK] Wrote {meta_path}")


if __name__ == "__main__":
    main()
