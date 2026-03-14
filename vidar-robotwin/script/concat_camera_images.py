#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concatenate head/left/right camera frames from hdf5 episodes in vidar-robotwin/data.

Layout:
- Top: head camera (original size)
- Bottom left/right: left/right wrist cameras resized to half and stacked horizontally

Output:
Writes concatenated JPGs under each episode's video_concat folder.
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
from PIL import Image


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
    # Stored JPEGs were encoded by OpenCV on RGB arrays; swap back to RGB.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def concat_three_views(head_img: np.ndarray, left_img: np.ndarray, right_img: np.ndarray) -> Optional[np.ndarray]:
    try:
        orig_h, orig_w = head_img.shape[:2]
        half_h, half_w = orig_h // 2, orig_w // 2
        left_resized = cv2.resize(left_img, (half_w, half_h))
        right_resized = cv2.resize(right_img, (half_w, half_h))
        bottom_row = np.hstack([left_resized, right_resized])
        return np.vstack([head_img, bottom_row])
    except Exception:
        return None


def process_hdf5(h5_path: Path, out_dir: Path, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(out_dir.iterdir()):
        print(f"[SKIP] Exists: {out_dir}")
        return

    with h5py.File(h5_path, "r") as f:
        try:
            head = f["observation"]["head_camera"]["rgb"]
            left = f["observation"]["left_camera"]["rgb"]
            right = f["observation"]["right_camera"]["rgb"]
        except KeyError as e:
            print(f"[SKIP] Missing key in {h5_path}: {e}")
            return

        length = len(head)
        if len(left) != length or len(right) != length:
            print(f"[SKIP] Length mismatch in {h5_path}")
            return

        for i in range(length):
            head_img = decode_image(head[i])
            left_img = decode_image(left[i])
            right_img = decode_image(right[i])
            combined = concat_three_views(head_img, left_img, right_img)
            if combined is None:
                print(f"[WARN] concat failed: {h5_path} frame {i}")
                continue
            out_path = out_dir / f"frame_{i:06d}.jpg"
            Image.fromarray(combined).save(out_path, quality=95)

    print(f"[OK] {h5_path} -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate multi-camera frames from hdf5 datasets.")
    parser.add_argument("--data-root", type=str, default="vidar-robotwin/data", help="Root of data directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output dirs")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"data root not found: {data_root}")

    h5_files = sorted(data_root.rglob("*.hdf5"))
    if not h5_files:
        raise SystemExit(f"no hdf5 files under: {data_root}")

    for h5_path in h5_files:
        rel = h5_path.relative_to(data_root)
        # data/<task>/<demo>/data/episodeX.hdf5 -> data/<task>/<demo>/video_concat/episodeX/
        try:
            if rel.parts[2] == "data":
                out_dir = data_root / rel.parts[0] / rel.parts[1] / "video_concat" / h5_path.stem
            else:
                out_dir = data_root / rel.parent / "video_concat" / h5_path.stem
        except IndexError:
            out_dir = data_root / rel.parent / "video_concat" / h5_path.stem
        process_hdf5(h5_path, out_dir, args.overwrite)


if __name__ == "__main__":
    main()
