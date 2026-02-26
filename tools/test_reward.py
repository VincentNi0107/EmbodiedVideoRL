#!/usr/bin/env python3
"""Standalone script to test GPT reward scoring on a single video.

Usage:
    python test_reward.py /path/to/video.mp4
    python test_reward.py /path/to/video.mp4 --save_debug debug_output.jpg
"""

import argparse
import base64
import json
import os
import sys

import cv2
import numpy as np
from openai import OpenAI


# ─── Config ──────────────────────────────────────────────────────
API_BASE = os.environ.get("GPT_API_BASE", "http://35.220.164.252:3888/v1/")
API_KEY = os.environ.get("GPT_API_KEY", "")
MODEL = "gemini-3-flash-preview"
TEMPERATURE = 0.0
def video_to_grid(video_path: str):
    """Read video, crop to main view, pick frames at 1/3, 2/3, 5/6, and end.
    Returns (base64_str, grid_bgr, list_of_picked_frames)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Target frame indices: 1/3, 2/3, 5/6, last
    pick_indices = sorted(set([
        0,
        int(total * 1 / 3),
        int(total * 2 / 3),
        total - 1,
    ]))
    # Clamp to valid range
    pick_indices = [max(0, min(i, total - 1)) for i in pick_indices]
    pick_set = set(pick_indices)

    frames_map = {}
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in pick_set:
            # Crop to main (rear camera) view — top 2/3
            h = frame.shape[0]
            main_h = h * 2 // 3
            frame = frame[:main_h, :, :]
            # Crop left/right 1/10 each
            w = frame.shape[1]
            margin_lr = w // 10
            frame = frame[:, margin_lr:w - margin_lr, :]
            frames_map[idx] = frame
        idx += 1
    cap.release()

    grid_frames_list = [frames_map[i] for i in pick_indices]

    print(f"[info] Total frames in video: {total}")
    print(f"[info] Picked frame indices: {pick_indices}")

    # 2x2 grid
    rows, cols = 2, 2
    row_imgs = []
    for r in range(rows):
        row_imgs.append(np.concatenate(grid_frames_list[r * cols:(r + 1) * cols], axis=1))
    grid = np.concatenate(row_imgs, axis=0)

    _, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    return b64, grid, grid_frames_list


def build_prompt(task_description: str) -> str:
    return f"""You are evaluating an AI-generated robot manipulation video.

The image shows a 2×2 grid of 4 frames sampled from the video in chronological order (read row by row, left-to-right then top-to-bottom).
Each frame is from a fixed rear camera showing the full workspace of a dual-arm "aloha" robot.

This task requires **two arms to collaborate**:
- The **right arm** (on the right bottom side of each image) should **open the drawer** by reaching for and pulling the handle.
- The **left arm** (on the left bottom side of each image) should **pick up the object from the table and place it inside the opened drawer**.

Evaluate the video for the following **failure criteria**. If ANY of them is true, the task FAILS (score = 0). Only if NONE of them is true, the task PASSES (score = 1).

### Failure Criteria
1. **Right arm frozen, drawer opens by itself**: The right arm does not move towards or contact the drawer, yet the drawer opens on its own.
2. **Right arm frozen, left arm does everything**: The right arm stays still while the left arm picks up the object AND also attempts to open the drawer by itself.

Respond with ONLY a JSON object (no markdown, no extra text):
{{"pass": true/false, "reason": "one-sentence explanation", "failures": ["list of triggered failure criterion numbers, e.g. 1,3"]}}
"""


def save_debug_image(grid_bgr, response_text: str, save_path: str):
    """Save grid + GPT response text below."""
    grid_h, grid_w = grid_bgr.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    color = (255, 255, 255)
    line_height = 36
    margin = 16

    usable_w = grid_w - margin * 2
    words = response_text.split()
    lines = []
    cur = ""
    for w in words:
        candidate = f"{cur} {w}" if cur else w
        (tw, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if tw > usable_w and cur:
            lines.append(cur)
            cur = w
        else:
            cur = candidate
    if cur:
        lines.append(cur)

    text_h = margin * 2 + line_height * len(lines)
    text_strip = np.zeros((text_h, grid_w, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        y = margin + line_height * (i + 1)
        cv2.putText(text_strip, line, (margin, y),
                     font, font_scale, color, thickness, cv2.LINE_AA)

    combined = np.concatenate([grid_bgr, text_strip], axis=0)
    cv2.imwrite(save_path, combined)
    print(f"[info] Debug image saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Test GPT reward on a single video")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--task", type=str,
                        default="The robot picks up the bread and places it into the cabinet drawer.",
                        help="Task description for the prompt")
    parser.add_argument("--save_debug", type=str, default=None,
                        help="Path to save debug image (grid + response)")
    parser.add_argument("--save_grid", type=str, default=None,
                        help="Path to save just the grid image (no text)")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    # 1. Read video and build grid
    print(f"[1/3] Reading video: {args.video_path}")
    grid_b64, grid_bgr, frame_list = video_to_grid(args.video_path)
    print(f"[info] Grid shape: {grid_bgr.shape}")

    if args.save_grid:
        cv2.imwrite(args.save_grid, grid_bgr)
        print(f"[info] Grid image saved to: {args.save_grid}")

    # 2. Build prompt
    prompt = build_prompt(args.task)
    print(f"\n[2/3] Prompt:\n{'─' * 60}")
    print(prompt)
    print(f"{'─' * 60}\n")

    # 3. Call GPT API
    print(f"[3/3] Calling GPT API (model={args.model}, temp={args.temperature}) ...")
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    resp = client.chat.completions.create(
        model=args.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{grid_b64}"}},
            ],
        }],
        temperature=args.temperature,
    )

    raw = resp.choices[0].message.content.strip()
    print(f"\n{'═' * 60}")
    print(f"RAW GPT RESPONSE:")
    print(f"{'═' * 60}")
    print(raw)
    print(f"{'═' * 60}\n")

    # Parse JSON
    clean = raw
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(clean)
        passed = bool(result.get("pass", False))
        reason = result.get("reason", "")
        failures = result.get("failures", [])
        reward = 1.0 if passed else 0.0

        print(f"  PASS:     {passed}")
        print(f"  REWARD:   {reward}")
        print(f"  REASON:   {reason}")
        print(f"  FAILURES: {failures}")
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Failed to parse JSON: {e}")
        passed = False
        reason = f"JSON parse error: {e}"
        failures = []
        reward = 0.0

    # Save debug image
    if args.save_debug:
        label = "PASS" if passed else "FAIL"
        response_text = f"[{label}] {reason}  failures={failures}"
        save_debug_image(grid_bgr, response_text, args.save_debug)


if __name__ == "__main__":
    main()
