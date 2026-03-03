"""Benchmark SAM3 segmentation speed for a single object on an image.

Usage:
    python tools/benchmark_sam3_single.py

Tests:
  1. SAM3 predictor initialization time
  2. Single-frame segmentation (1 image as "video")
  3. 121-frame segmentation (simulated video from same image)
"""

import os
import sys
import time
import tempfile
import shutil

import cv2
import torch
import numpy as np


def main():
    image_path = "/gpfs/projects/p33048/DanceGRPO/data/rl_train/blocks_ranking_rgb/robotwin_blocks_ranking_rgb_123500000.png"
    prompt = "red block"

    print(f"Image: {image_path}")
    print(f"Prompt: '{prompt}'")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device: {torch.cuda.current_device()}")
    print("=" * 60)

    # --- 1. Time SAM3 predictor init ---
    print("\n[1] Initializing Sam3VideoPredictor ...")
    t0 = time.time()
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    predictor = Sam3VideoPredictor()
    t_init = time.time() - t0
    print(f"    SAM3 init: {t_init:.2f}s")

    # Read and crop image (top 2/3, same as training)
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    crop_h = h * 2 // 3
    img_cropped = img[:crop_h]
    print(f"    Image size: {w}x{h}, cropped to: {w}x{crop_h}")

    # --- 2. Single-frame test ---
    print("\n[2] Single-frame segmentation ...")
    tmpdir1 = tempfile.mkdtemp(prefix="sam3_bench_1f_")
    cv2.imwrite(os.path.join(tmpdir1, "000000.jpg"), img_cropped)

    t0 = time.time()
    resp = predictor.handle_request(dict(
        type="start_session",
        resource_path=tmpdir1,
    ))
    t_session = time.time() - t0
    session_id = resp["session_id"]
    print(f"    start_session: {t_session:.2f}s")

    t0 = time.time()
    predictor.handle_request(dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text=prompt,
    ))
    t_prompt = time.time() - t0
    print(f"    add_prompt ('{prompt}'): {t_prompt:.2f}s")

    t0 = time.time()
    results_1f = []
    for resp_frame in predictor.handle_stream_request(dict(
        type="propagate_in_video",
        session_id=session_id,
    )):
        results_1f.append(resp_frame)
    t_propagate = time.time() - t0
    print(f"    propagate (1 frame): {t_propagate:.2f}s")
    print(f"    total single-frame: {t_session + t_prompt + t_propagate:.2f}s")

    # Print detection results
    if results_1f:
        outs = results_1f[0]["outputs"]
        n_obj = outs["frame_stats"]["num_obj_tracked"]
        boxes = outs["out_boxes_xywh"]
        print(f"    objects found: {n_obj}")
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                print(f"      obj {i}: xywh = [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]")

    predictor.handle_request(dict(type="close_session", session_id=session_id))
    shutil.rmtree(tmpdir1)

    # --- 3. 121-frame test (simulated video) ---
    n_frames = 121
    print(f"\n[3] {n_frames}-frame segmentation (simulated video) ...")
    tmpdir121 = tempfile.mkdtemp(prefix="sam3_bench_121f_")
    for i in range(n_frames):
        cv2.imwrite(os.path.join(tmpdir121, f"{i:06d}.jpg"), img_cropped)

    t0 = time.time()
    resp = predictor.handle_request(dict(
        type="start_session",
        resource_path=tmpdir121,
    ))
    t_session = time.time() - t0
    session_id = resp["session_id"]
    print(f"    start_session: {t_session:.2f}s")

    t0 = time.time()
    predictor.handle_request(dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text=prompt,
    ))
    t_prompt = time.time() - t0
    print(f"    add_prompt ('{prompt}'): {t_prompt:.2f}s")

    t0 = time.time()
    results_121f = []
    for resp_frame in predictor.handle_stream_request(dict(
        type="propagate_in_video",
        session_id=session_id,
    )):
        results_121f.append(resp_frame)
    t_propagate = time.time() - t0
    t_total = t_session + t_prompt + t_propagate
    print(f"    propagate ({n_frames} frames): {t_propagate:.2f}s")
    print(f"    per-frame propagation: {t_propagate / n_frames * 1000:.1f}ms")
    print(f"    total {n_frames}-frame: {t_total:.2f}s")

    # Summarize object counts across frames
    counts = [r["outputs"]["frame_stats"]["num_obj_tracked"] for r in results_121f]
    print(f"    object count across frames: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}")

    predictor.handle_request(dict(type="close_session", session_id=session_id))
    shutil.rmtree(tmpdir121)

    # --- 4. Multi-prompt test (3 prompts, like blocks_ranking_rgb) ---
    prompts = ["red block", "green block", "blue block"]
    print(f"\n[4] {n_frames}-frame × {len(prompts)} prompts (sequential, like training) ...")
    tmpdir_multi = tempfile.mkdtemp(prefix="sam3_bench_multi_")
    for i in range(n_frames):
        cv2.imwrite(os.path.join(tmpdir_multi, f"{i:06d}.jpg"), img_cropped)

    t_total_multi = 0
    for p in prompts:
        t0 = time.time()
        resp = predictor.handle_request(dict(
            type="start_session", resource_path=tmpdir_multi))
        sid = resp["session_id"]
        predictor.handle_request(dict(
            type="add_prompt", session_id=sid,
            frame_index=0, text=p))
        for _ in predictor.handle_stream_request(dict(
            type="propagate_in_video", session_id=sid)):
            pass
        predictor.handle_request(dict(type="close_session", session_id=sid))
        t_one = time.time() - t0
        t_total_multi += t_one
        print(f"    '{p}': {t_one:.2f}s")

    print(f"    total 3-prompt: {t_total_multi:.2f}s")
    shutil.rmtree(tmpdir_multi)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  SAM3 init:                   {t_init:.2f}s")
    print(f"  1-frame, 1-prompt:           {t_session + t_prompt + t_propagate:.2f}s")
    print(f"  121-frame, 1-prompt:         {t_total:.2f}s")
    print(f"  121-frame, 3-prompt (seq):   {t_total_multi:.2f}s")
    print(f"  Per-frame propagation speed: {t_propagate / n_frames * 1000:.1f}ms/frame")


if __name__ == "__main__":
    main()
