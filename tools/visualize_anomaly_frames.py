"""
Export anomaly frames to images from flow anomaly detection outputs.

Inputs:
- Original video
- Annotated anomaly video (optional)
- Summary JSON from detect_flow_anomalies.py
- Scores CSV (optional)

Outputs:
- Per-frame PNG files
- A contact sheet PNG for quick review
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, List

import cv2
import numpy as np


def load_scores(csv_path: str | None) -> Dict[int, Dict[str, float]]:
    if not csv_path or not os.path.exists(csv_path):
        return {}
    out: Dict[int, Dict[str, float]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["frame_idx"])
            out[idx] = {
                "score_raw": float(row.get("score_raw", 0.0)),
                "score_smooth": float(row.get("score_smooth", 0.0)),
                "anomaly_area_ratio": float(row.get("anomaly_area_ratio", 0.0)),
                "max_mag": float(row.get("max_mag", 0.0)),
            }
    return out


def collect_candidate_frames(summary_path: str, topk: int, neighbors: int) -> List[int]:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    frames = set()
    for i in summary.get("peak_frames", []):
        frames.add(int(i))

    top_list = summary.get("top10_frames_by_score", [])
    for item in top_list[:topk]:
        frames.add(int(item["frame_idx"]))

    if neighbors > 0:
        base = sorted(frames)
        for i in base:
            for d in range(-neighbors, neighbors + 1):
                frames.add(max(0, i + d))
    return sorted(frames)


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def draw_info(img: np.ndarray, frame_idx: int, score_row: Dict[str, float] | None) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, f"frame={frame_idx}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if score_row:
        txt = (
            f"smooth={score_row['score_smooth']:.3f} "
            f"raw={score_row['score_raw']:.3f} "
            f"area={score_row['anomaly_area_ratio']:.3f} "
            f"max_mag={score_row['max_mag']:.3f}"
        )
        cv2.putText(out, txt, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    return out


def make_contact_sheet(images: List[np.ndarray], cols: int = 4) -> np.ndarray:
    if not images:
        return np.zeros((320, 640, 3), dtype=np.uint8)
    h, w = images[0].shape[:2]
    cols = max(1, cols)
    rows = math.ceil(len(images) / cols)
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize detected anomaly frames")
    parser.add_argument("--video", required=True, type=str, help="Original video path")
    parser.add_argument("--anomaly-video", default=None, type=str, help="Annotated anomaly video path")
    parser.add_argument("--summary-json", required=True, type=str, help="Summary json path")
    parser.add_argument("--scores-csv", default=None, type=str, help="Scores csv path")
    parser.add_argument("--out-dir", default=None, type=str, help="Output directory")
    parser.add_argument("--topk", default=10, type=int, help="How many top-score frames to include")
    parser.add_argument("--neighbors", default=0, type=int, help="Also include +/-N neighboring frames")
    parser.add_argument("--contact-cols", default=4, type=int, help="Contact sheet columns")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.summary_json), "anomaly_frames")
    os.makedirs(out_dir, exist_ok=True)

    scores = load_scores(args.scores_csv)
    frame_indices = collect_candidate_frames(args.summary_json, topk=args.topk, neighbors=args.neighbors)
    if not frame_indices:
        print("No anomaly frames found from summary/topk.")
        return

    cap_org = cv2.VideoCapture(args.video)
    if not cap_org.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    cap_ann = None
    if args.anomaly_video and os.path.exists(args.anomaly_video):
        cap_ann = cv2.VideoCapture(args.anomaly_video)
        if not cap_ann.isOpened():
            cap_ann = None

    contact_images = []
    for idx in frame_indices:
        org = read_frame(cap_org, idx)
        if org is None:
            continue
        score_row = scores.get(idx)
        org = draw_info(org, idx, score_row)

        if cap_ann is not None:
            ann = read_frame(cap_ann, idx)
            if ann is not None:
                ann = draw_info(ann, idx, score_row)
                vis = np.hstack([org, ann])
            else:
                vis = org
        else:
            vis = org

        file_score = score_row["score_smooth"] if score_row else 0.0
        out_name = f"frame_{idx:04d}_score_{file_score:.3f}.png"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, vis)
        contact_images.append(vis)

    cap_org.release()
    if cap_ann is not None:
        cap_ann.release()

    if contact_images:
        sheet = make_contact_sheet(contact_images, cols=args.contact_cols)
        sheet_path = os.path.join(out_dir, "anomaly_contact_sheet.png")
        cv2.imwrite(sheet_path, sheet)
        print(f"Saved {len(contact_images)} anomaly frame images to: {out_dir}")
        print(f"Saved contact sheet: {sheet_path}")
    else:
        print("No frame image was saved (frame indices may be out of range).")


if __name__ == "__main__":
    main()
