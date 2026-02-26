#!/usr/bin/env python3
"""Batch-test GPT reward on all videos in step0001_* to step0010_*.

Usage:
    python batch_test_reward.py
"""

import glob
import os
import re
import sys

from test_reward import video_to_grid, build_prompt, save_debug_image

from openai import OpenAI
import json

# ─── Config (same as test_reward.py) ─────────────────────────────
API_BASE = os.environ.get("GPT_API_BASE", "http://35.220.164.252:3888/v1/")
API_KEY = os.environ.get("GPT_API_KEY", "")
MODEL = "gemini-3-flash-preview"
TEMPERATURE = 0.0
TASK = "The robot picks up the bread and places it into the cabinet drawer."

VIDEO_ROOT = "/home/omz1504/code/DanceGRPO/data/outputs/grpo_put_object_cabinet/videos"
DEBUG_ROOT = "/home/omz1504/code/DanceGRPO/data/outputs/grpo_put_object_cabinet/reward_debug_v3"


def score_one(client, video_path: str, task: str):
    """Score a single video, return (passed, reason, failures, raw_response)."""
    grid_b64, grid_bgr, _ = video_to_grid(video_path)
    prompt = build_prompt(task)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{grid_b64}"}},
            ],
        }],
        temperature=TEMPERATURE,
    )

    raw = resp.choices[0].message.content.strip()
    clean = raw
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(clean)
        passed = bool(result.get("pass", False))
        reason = result.get("reason", "")
        failures = result.get("failures", [])
    except json.JSONDecodeError:
        passed = False
        reason = f"JSON parse error"
        failures = []

    return passed, reason, failures, raw, grid_bgr


def main():
    # Collect step dirs matching step0001_* through step0010_*
    step_dirs = sorted(glob.glob(os.path.join(VIDEO_ROOT, "step00[0-1][0-9]_*")))
    # Filter to step0001 - step0010 only
    step_dirs = [d for d in step_dirs if re.match(
        r".*step(000[1-9]|0010)_", d)]

    print(f"Found {len(step_dirs)} step directories")
    os.makedirs(DEBUG_ROOT, exist_ok=True)

    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    total_videos = 0
    total_pass = 0
    total_fail = 0

    for step_dir in step_dirs:
        step_name = os.path.basename(step_dir)
        debug_dir = os.path.join(DEBUG_ROOT, step_name)
        os.makedirs(debug_dir, exist_ok=True)

        videos = sorted(glob.glob(os.path.join(step_dir, "*.mp4")))
        print(f"\n{'═' * 60}")
        print(f"  {step_name}  ({len(videos)} videos)")
        print(f"{'═' * 60}")

        step_pass = 0
        step_fail = 0

        for vp in videos:
            vname = os.path.splitext(os.path.basename(vp))[0]
            print(f"  [{step_name}] {vname} ... ", end="", flush=True)

            try:
                passed, reason, failures, raw, grid_bgr = score_one(
                    client, vp, TASK)
            except Exception as e:
                print(f"ERROR: {e}")
                total_videos += 1
                total_fail += 1
                step_fail += 1
                continue

            tag = "PASS" if passed else "FAIL"
            reward = 1.0 if passed else 0.0

            print(f"{tag}  reason: {reason}  failures: {failures}")

            # Save debug image
            response_text = f"[{tag}] {reason}  failures={failures}"
            debug_path = os.path.join(debug_dir, f"{vname}_{tag}.jpg")
            try:
                save_debug_image(grid_bgr, response_text, debug_path)
            except Exception as e:
                print(f"    [warn] debug image save failed: {e}")

            if passed:
                step_pass += 1
                total_pass += 1
            else:
                step_fail += 1
                total_fail += 1
            total_videos += 1

        print(f"  ── {step_name} summary: {step_pass} PASS / {step_fail} FAIL "
              f"({step_pass}/{step_pass + step_fail} = "
              f"{step_pass / max(1, step_pass + step_fail) * 100:.0f}%)")

    print(f"\n{'═' * 60}")
    print(f"  TOTAL: {total_pass} PASS / {total_fail} FAIL / {total_videos} videos")
    print(f"  Pass rate: {total_pass / max(1, total_videos) * 100:.1f}%")
    print(f"  Results saved to: {DEBUG_ROOT}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
