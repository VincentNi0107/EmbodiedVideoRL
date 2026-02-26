"""
Full pipeline: SAM3 tracking → trajectory extraction → truncation → hallucination detection
for put_bottles_dustbin videos.

Processes all robotwin_put_bottles_dustbin_* scene directories under --input-root.
For each video:
  1. SAM3 tracks "bottle" across all frames
  2. Extracts per-object trajectories (using obj_ids for identity)
  3. Filters out spurious objects (appeared after frame 5)
  4. Truncates each real object's trajectory when cx < cx_cutoff (bottle reached dustbin)
  5. Applies 3-criteria hallucination check:
     (a) All 3 bottles must reach cx < cx_cutoff  (all placed)
     (b) After truncation, active bottle count must be monotonically non-increasing
     (c) No two bottles can share the same placed_frame  (merge hallucination)
  6. Saves: tracking video (.mp4), trajectory JSON, per-video analysis JSON

Outputs per scene directory:
  <out-root>/<scene>/
    <video_stem>_tracking.mp4       — annotated tracking video with bbox + trails
    <video_stem>_trajectory.json    — full per-object per-frame trajectory data
    <scene>_analysis.json           — per-video hallucination analysis
"""

import argparse
import json
import os
import sys
import shutil
import glob

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from detect_hallucination_bottles import extract_frames_to_jpeg, track_prompt, save_video_libx264


# ── Colours for drawing ──────────────────────────────────────────────────────
OBJ_COLORS_BGR = [
    (0, 0, 255),     # red     - obj 0
    (0, 200, 0),     # green   - obj 1
    (255, 150, 0),   # blue    - obj 2
    (0, 200, 255),   # yellow  - obj 3
    (255, 0, 200),   # magenta - obj 4
]


# ── Drawing ──────────────────────────────────────────────────────────────────

def draw_tracking_frame(
    frame_bgr: np.ndarray,
    frame_result: dict,
    frame_idx: int,
    total_frames: int,
    trail_history: dict,
    placed_ids: set,        # obj_ids already placed (greyed out)
    cx_cutoff: float,
) -> np.ndarray:
    """Draw bboxes, trails, cutoff line, and status on a frame."""
    ann = frame_bgr.copy()
    h, w = ann.shape[:2]

    # Draw cx_cutoff vertical line
    cutoff_x = int(cx_cutoff * w)
    cv2.line(ann, (cutoff_x, 0), (cutoff_x, h), (0, 255, 255), 1, cv2.LINE_AA)

    for i, obj_id in enumerate(frame_result["obj_ids"]):
        oid_str = str(obj_id)
        bx, by, bw, bh = frame_result["boxes_xywh"][i]
        x0, y0 = int(bx * w), int(by * h)
        x1, y1 = int((bx + bw) * w), int((by + bh) * h)
        cx_px, cy_px = (x0 + x1) // 2, (y0 + y1) // 2
        prob = frame_result["probs"][i]

        if oid_str in placed_ids:
            # Already placed → draw in grey with dashed style
            color = (120, 120, 120)
            cv2.rectangle(ann, (x0, y0), (x1, y1), color, 1)
            label = f"id={obj_id} PLACED"
        else:
            color = OBJ_COLORS_BGR[obj_id % len(OBJ_COLORS_BGR)]
            cv2.rectangle(ann, (x0, y0), (x1, y1), color, 2)
            label = f"id={obj_id} {prob:.2f}"

            # Trail only for active objects
            if oid_str not in trail_history:
                trail_history[oid_str] = []
            trail_history[oid_str].append((cx_px, cy_px))

        # Label background
        (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(ann, (x0, max(y0 - th_t - 6, 0)), (x0 + tw + 4, y0), color, -1)
        cv2.putText(ann, label, (x0 + 2, max(y0 - 3, th_t + 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Draw trajectory trails
    for oid_str, trail in trail_history.items():
        if len(trail) < 2:
            continue
        color = OBJ_COLORS_BGR[int(oid_str) % len(OBJ_COLORS_BGR)]
        for k in range(1, len(trail)):
            alpha = 0.3 + 0.7 * (k / len(trail))
            c = tuple(int(v * alpha) for v in color)
            cv2.line(ann, trail[k - 1], trail[k], c, 1, cv2.LINE_AA)

    # Frame counter
    cv2.putText(ann, f"frame {frame_idx}/{total_frames}",
                (w - 190, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)
    return ann


# ── Trajectory extraction ────────────────────────────────────────────────────

def extract_and_render(
    input_path: str,
    output_video: str,
    prompt: str,
    crop_top_ratio: float,
    cx_cutoff: float,
    predictor,
) -> dict:
    """SAM3 track → extract trajectories → render annotated video.

    Returns trajectory dict with per-object data.
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    crop_h = None
    if crop_top_ratio < 1.0:
        crop_h = (int(h_full * crop_top_ratio)) // 16 * 16
    h = crop_h if crop_h else h_full

    print(f"  {total_frames} frames, {vid_w}x{h} @ {fps:.1f} fps")

    jpeg_dir = extract_frames_to_jpeg(input_path, crop_h=crop_h)

    print(f"  Tracking '{prompt}'...")
    frame_results = track_prompt(predictor, jpeg_dir, prompt)
    print(f"  Tracked {len(frame_results)} frames.")

    # ── Build per-object trajectories ────────────────────────────────────
    objects = {}
    for fi, fr in enumerate(frame_results):
        for i, obj_id in enumerate(fr["obj_ids"]):
            oid = str(obj_id)
            if oid not in objects:
                objects[oid] = []
            bx, by, bw, bh = fr["boxes_xywh"][i]
            objects[oid].append({
                "frame": fi,
                "cx": float(bx + bw / 2),
                "cy": float(by + bh / 2),
                "x": float(bx), "y": float(by),
                "w": float(bw), "h": float(bh),
                "prob": float(fr["probs"][i]),
            })

    # ── Pre-compute placed_frame per object (for video rendering) ────────
    placed_frames = {}   # obj_id_str -> frame where cx first < cutoff
    for oid, traj in objects.items():
        for pt in traj:
            if pt["cx"] < cx_cutoff:
                placed_frames[oid] = pt["frame"]
                break

    # ── Render annotated video ───────────────────────────────────────────
    print("  Rendering tracking video...")
    trail_history = {}
    out_frames = []
    current_placed = set()
    for fi in range(total_frames):
        # Update placed set
        for oid, pf in placed_frames.items():
            if fi > pf:
                current_placed.add(oid)

        frame_bgr = cv2.imread(os.path.join(jpeg_dir, f"{fi:06d}.jpg"))
        ann = draw_tracking_frame(
            frame_bgr, frame_results[fi], fi, total_frames,
            trail_history, current_placed, cx_cutoff,
        )
        out_frames.append(ann)

    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
    save_video_libx264(out_frames, output_video, fps)
    print(f"  Saved: {output_video}")

    shutil.rmtree(jpeg_dir, ignore_errors=True)

    return {
        "video": input_path,
        "frame_count": total_frames,
        "frame_size": [vid_w, h],
        "fps": fps,
        "prompt": prompt,
        "crop_top_ratio": crop_top_ratio,
        "num_objects_tracked": len(objects),
        "objects": objects,
    }


# ── Hallucination analysis ───────────────────────────────────────────────────

def truncate_trajectory(traj, cx_cutoff):
    for i, pt in enumerate(traj):
        if pt["cx"] < cx_cutoff:
            return traj[:i], pt["frame"]
    return traj, None


def analyze_hallucination(traj_data: dict, cx_cutoff: float, gap_max: int,
                          reappear_max: int = 10,
                          post_reappear_min_hw: float = 2.0,
                          post_reappear_min_frames: int = 4,
                          teleport_max_gap: int = 5) -> dict:
    """Apply 5-criteria hallucination check on one video's trajectory data.

    Criteria:
      (a) all_placed:     all 3 real bottles reached cx < cx_cutoff
      (b) monotonic:      active count never increases (after truncation)
      (c) no_merge:       no two bottles share the same placed_frame
      (d) no_reappear:    no bottle disappears for > reappear_max frames then reappears
                          (within its truncated trajectory, i.e. before placement)
      (e) no_teleport:    for each placed bottle, the gap between the last tracked
                          detection and the placement frame must not exceed
                          teleport_max_gap.  A large gap means the bottle
                          disappeared and then materialised at the dustbin without
                          being tracked during transport (teleportation hallucination).

    Returns analysis dict.
    """
    total_frames = traj_data["frame_count"]
    objects = traj_data["objects"]

    # Filter real vs spurious
    real_objects = {}
    spurious_ids = []
    for oid, traj in objects.items():
        if traj and traj[0]["frame"] <= 5:
            real_objects[oid] = traj
        else:
            spurious_ids.append(oid)

    # ── Per-object truncation ────────────────────────────────────────────
    object_info = {}
    active_frames = {}   # frame -> set of active obj_ids
    for oid, traj in real_objects.items():
        truncated, placed_frame = truncate_trajectory(traj, cx_cutoff)
        object_info[oid] = {
            "original_frames": len(traj),
            "truncated_frames": len(truncated),
            "placed_frame": placed_frame,
            "first_frame": traj[0]["frame"],
            "last_active_frame": truncated[-1]["frame"] if truncated else None,
            "initial_cx": traj[0]["cx"],
        }
        for pt in truncated:
            fi = pt["frame"]
            if fi not in active_frames:
                active_frames[fi] = set()
            active_frames[fi].add(oid)

    active_counts = [len(active_frames.get(fi, set())) for fi in range(total_frames)]

    # ── Criterion (a): all placed ────────────────────────────────────────
    placed_frames_map = {
        oid: info["placed_frame"]
        for oid, info in object_info.items()
    }
    all_placed = all(pf is not None for pf in placed_frames_map.values())
    not_placed_ids = [oid for oid, pf in placed_frames_map.items() if pf is None]

    # ── Criterion (b): monotonic non-increasing ─────────────────────────
    monotonic_ok = True
    mono_events = []
    committed = active_counts[0] if active_counts else 0
    i = 0
    n = len(active_counts)
    while i < n:
        c = active_counts[i]
        if c > committed:
            spike_start = i
            j = i
            while j < n and active_counts[j] > committed:
                j += 1
            spike_len = j - spike_start
            if spike_len <= gap_max and j < n:
                mono_events.append({
                    "type": "artifact_suppressed",
                    "frames": f"{spike_start}-{j-1}",
                    "length": spike_len,
                })
            else:
                monotonic_ok = False
                mono_events.append({
                    "type": "count_increase",
                    "frames": f"{spike_start}-{j-1}" if j < n else f"{spike_start}-end",
                    "length": spike_len,
                    "peak_count": max(active_counts[spike_start:j]),
                    "committed": committed,
                })
            i = j
        else:
            if c < committed:
                committed = c
            i += 1

    # ── Criterion (c): no merge (no two bottles share same placed_frame) ─
    placed_frame_values = [pf for pf in placed_frames_map.values() if pf is not None]
    merge_ok = len(placed_frame_values) == len(set(placed_frame_values))
    merged_pairs = []
    if not merge_ok:
        from collections import Counter
        counts = Counter(placed_frame_values)
        for frame_val, cnt in counts.items():
            if cnt > 1:
                colliding = [oid for oid, pf in placed_frames_map.items() if pf == frame_val]
                merged_pairs.append({"frame": frame_val, "obj_ids": colliding})

    # ── Criterion (d): no long disappearance + reappearance ──────────────
    #   (d1) Pre-placement: within truncated trajectory, gap > reappear_max
    #   (d2) Post-placement: after placed_frame, SAM3 still detects the object
    #        far from the dustbin (cx > cx_cutoff + 0.15), meaning the bottle
    #        visually reappeared in the video
    reappear_ok = True
    reappear_events = []
    for oid, traj in real_objects.items():
        truncated, placed_frame = truncate_trajectory(traj, cx_cutoff)
        # (d1) pre-placement gaps
        if len(truncated) >= 2:
            for k in range(1, len(truncated)):
                gap = truncated[k]["frame"] - truncated[k-1]["frame"] - 1
                if gap > reappear_max:
                    reappear_ok = False
                    reappear_events.append({
                        "obj_id": oid,
                        "type": "pre_placement_gap",
                        "disappeared_after": truncated[k-1]["frame"],
                        "reappeared_at": truncated[k]["frame"],
                        "gap_frames": gap,
                    })
        # (d2) post-placement: bottle reappears far from dustbin
        #      Two extra filters suppress SAM3 segmentation errors:
        #      - aspect ratio: reappearing box must have h/w >= post_reappear_min_hw
        #        (real bottles are tall/thin ~4.7:1; SAM3 artefacts are flat ~0.5-1.7:1)
        #      - duration: reappearance burst must last >= post_reappear_min_frames
        #        (SAM3 artefacts typically last 1-3 frames)
        if placed_frame is not None:
            post_thr = cx_cutoff + 0.24   # e.g. 0.50 — clearly away from dustbin
            # Collect all post-placement detections that pass the cx threshold
            post_detections = [
                pt for pt in traj
                if pt["frame"] > placed_frame and pt["cx"] > post_thr
            ]
            # Group into consecutive bursts (gap <= gap_max between frames)
            bursts = []
            for pt in post_detections:
                if bursts and pt["frame"] - bursts[-1][-1]["frame"] <= gap_max:
                    bursts[-1].append(pt)
                else:
                    bursts.append([pt])
            # Check each burst against both filters
            for burst in bursts:
                burst_len = burst[-1]["frame"] - burst[0]["frame"] + 1
                has_bottle_shape = any(
                    (pt["h"] / pt["w"] >= post_reappear_min_hw) if pt["w"] > 0 else False
                    for pt in burst
                )
                if burst_len >= post_reappear_min_frames and has_bottle_shape:
                    reappear_ok = False
                    reappear_events.append({
                        "obj_id": oid,
                        "type": "post_placement_reappear",
                        "placed_frame": placed_frame,
                        "reappeared_frame": burst[0]["frame"],
                        "reappeared_cx": round(burst[0]["cx"], 4),
                        "burst_frames": burst_len,
                        "burst_hw_max": round(max(
                            (pt["h"] / pt["w"] if pt["w"] > 0 else 0) for pt in burst
                        ), 2),
                    })
                    break   # one event per object is enough
                else:
                    # Suppressed SAM3 artefact — log it but don't fail
                    reappear_events.append({
                        "obj_id": oid,
                        "type": "post_placement_suppressed",
                        "placed_frame": placed_frame,
                        "reappeared_frame": burst[0]["frame"],
                        "reappeared_cx": round(burst[0]["cx"], 4),
                        "burst_frames": burst_len,
                        "burst_hw_max": round(max(
                            (pt["h"] / pt["w"] if pt["w"] > 0 else 0) for pt in burst
                        ), 2),
                    })

    # ── Criterion (e): no teleport (placement-gap check) ────────────────
    #   For each placed bottle, the gap between the last tracked detection
    #   and the placed_frame must not exceed teleport_max_gap.  In normal
    #   videos the bottle is tracked every frame during the final approach.
    #   A large gap means the bottle vanished and materialised at the
    #   dustbin — the hallucination the user described as "消失后瞬移".
    teleport_ok = True
    teleport_events = []
    if teleport_max_gap >= 0:
        for oid, traj in real_objects.items():
            truncated, placed_frame = truncate_trajectory(traj, cx_cutoff)
            if placed_frame is None or not truncated:
                continue
            last_frame = truncated[-1]["frame"]
            gap_to_placed = placed_frame - last_frame - 1
            if gap_to_placed > teleport_max_gap:
                teleport_ok = False
                teleport_events.append({
                    "obj_id": oid,
                    "type": "teleport_gap",
                    "last_tracked_frame": last_frame,
                    "last_tracked_cx": round(truncated[-1]["cx"], 4),
                    "placed_frame": placed_frame,
                    "gap_frames": gap_to_placed,
                })

    # ── Overall verdict ──────────────────────────────────────────────────
    is_clean = all_placed and monotonic_ok and merge_ok and reappear_ok and teleport_ok

    # Build failure reasons
    fail_reasons = []
    if not all_placed:
        fail_reasons.append(f"not_all_placed({not_placed_ids})")
    if not monotonic_ok:
        fail_reasons.append("count_increase")
    if not merge_ok:
        fail_reasons.append(f"merge({merged_pairs})")
    if not reappear_ok:
        brief = []
        for e in reappear_events:
            if e["type"] == "pre_placement_gap":
                brief.append(f"obj{e['obj_id']}:gap_f{e['disappeared_after']}-f{e['reappeared_at']}({e['gap_frames']}f)")
            elif e["type"] == "post_placement_reappear":
                brief.append(f"obj{e['obj_id']}:post_placed@f{e['placed_frame']}_seen@f{e['reappeared_frame']}"
                             f"(cx={e['reappeared_cx']},burst={e['burst_frames']}f,hw={e['burst_hw_max']})")
            # skip post_placement_suppressed — not a failure
        fail_reasons.append(f"reappear({', '.join(brief)})")
    if not teleport_ok:
        brief = []
        for e in teleport_events:
            brief.append(f"obj{e['obj_id']}:last_f{e['last_tracked_frame']}"
                         f"(cx={e['last_tracked_cx']})"
                         f"→placed_f{e['placed_frame']}"
                         f"(gap={e['gap_frames']}f)")
        fail_reasons.append(f"teleport({', '.join(brief)})")

    # Count transitions string
    transitions = []
    if active_counts:
        prev = active_counts[0]
        transitions.append(str(prev))
        for fi, c in enumerate(active_counts[1:], 1):
            if c != prev:
                transitions.append(f"→{c}@f{fi}")
                prev = c

    return {
        "video": os.path.basename(traj_data["video"]),
        "total_frames": total_frames,
        "cx_cutoff": cx_cutoff,
        "num_real_objects": len(real_objects),
        "spurious_obj_ids": spurious_ids,
        "object_info": object_info,
        "active_count_transitions": " ".join(transitions),
        # 4 criteria
        "all_placed": all_placed,
        "not_placed_ids": not_placed_ids,
        "monotonic_ok": monotonic_ok,
        "mono_events": mono_events,
        "no_merge": merge_ok,
        "merged_pairs": merged_pairs,
        "no_reappear": reappear_ok,
        "reappear_events": reappear_events,
        "no_teleport": teleport_ok,
        "teleport_events": teleport_events,
        # verdict
        "is_clean": is_clean,
        "fail_reasons": fail_reasons,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full bottle hallucination pipeline: SAM3 track → analyse → video")
    parser.add_argument("--input-root", required=True,
                        help="Root dir containing robotwin_put_bottles_dustbin_* scene dirs")
    parser.add_argument("--out-root", required=True,
                        help="Output root directory")
    parser.add_argument("--pattern", default="robotwin_put_bottles_dustbin_*",
                        help="Glob pattern for scene directories")
    parser.add_argument("--prompt", default="bottle")
    parser.add_argument("--crop-top-ratio", type=float, default=2/3)
    parser.add_argument("--cx-cutoff", type=float, default=0.26)
    parser.add_argument("--gap-max", type=int, default=3)
    parser.add_argument("--reappear-max", type=int, default=10,
                        help="Max frames a bottle can disappear before reappearance = hallucination (default: 10)")
    args = parser.parse_args()

    # Find scene directories
    scene_dirs = sorted(glob.glob(os.path.join(args.input_root, args.pattern)))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]
    total_videos = sum(
        len([f for f in os.listdir(d) if f.endswith(".mp4")])
        for d in scene_dirs
    )
    print(f"Found {len(scene_dirs)} scenes, {total_videos} videos total")
    print(f"Settings: cx_cutoff={args.cx_cutoff}, gap_max={args.gap_max}, "
          f"reappear_max={args.reappear_max}, crop_top_ratio={args.crop_top_ratio}")

    # Load SAM3 once
    from sam3.model_builder import build_sam3_video_predictor
    print("\nLoading SAM3 video predictor...")
    predictor = build_sam3_video_predictor()

    global_vid_idx = 0
    all_scene_summaries = []

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        out_dir = os.path.join(args.out_root, scene_name)
        os.makedirs(out_dir, exist_ok=True)

        videos = sorted([
            os.path.join(scene_dir, f)
            for f in os.listdir(scene_dir) if f.endswith(".mp4")
        ])

        print(f"\n{'#'*70}")
        print(f"# Scene: {scene_name}  ({len(videos)} videos)")
        print(f"{'#'*70}")

        scene_results = []
        for vi, vpath in enumerate(videos):
            global_vid_idx += 1
            stem = os.path.splitext(os.path.basename(vpath))[0]
            print(f"\n[{global_vid_idx}/{total_videos}] {stem}")

            # 1) SAM3 track + render video (temp name, rename after analysis)
            tmp_video = os.path.join(out_dir, f"{stem}_tracking_tmp.mp4")
            traj_data = extract_and_render(
                input_path=vpath,
                output_video=tmp_video,
                prompt=args.prompt,
                crop_top_ratio=args.crop_top_ratio,
                cx_cutoff=args.cx_cutoff,
                predictor=predictor,
            )

            # 2) Save trajectory JSON
            traj_json = os.path.join(out_dir, f"{stem}_trajectory.json")
            with open(traj_json, "w") as f:
                json.dump(traj_data, f, indent=2)

            # 3) Analyse hallucination
            analysis = analyze_hallucination(traj_data, args.cx_cutoff, args.gap_max,
                                             args.reappear_max)
            scene_results.append(analysis)

            # 4) Rename video with CLEAN/FAIL tag
            verdict = "CLEAN" if analysis["is_clean"] else "FAIL"
            final_video = os.path.join(out_dir, f"{stem}_{verdict}.mp4")
            os.rename(tmp_video, final_video)

            symbol = "✓" if analysis["is_clean"] else "✗"
            reasons = ", ".join(analysis["fail_reasons"]) if analysis["fail_reasons"] else ""
            print(f"  {symbol} {verdict}  "
                  f"placed=[{', '.join(str(analysis['object_info'][o]['placed_frame']) for o in sorted(analysis['object_info']) if analysis['object_info'][o]['placed_frame'] is not None)}]  "
                  f"count: {analysis['active_count_transitions']}"
                  f"{('  ← ' + reasons) if reasons else ''}")

        # Save scene analysis
        analysis_path = os.path.join(out_dir, f"{scene_name}_analysis.json")
        with open(analysis_path, "w") as f:
            json.dump(scene_results, f, indent=2)

        n_clean = sum(1 for r in scene_results if r["is_clean"])
        n_total = len(scene_results)
        all_scene_summaries.append({
            "scene": scene_name,
            "total": n_total,
            "clean": n_clean,
            "fail": n_total - n_clean,
        })

        print(f"\n  Scene summary: {n_clean}/{n_total} clean")

    predictor.shutdown()

    # ── Grand summary ────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("GRAND SUMMARY")
    print(f"{'='*70}")
    print(f"{'Scene':<45} {'Clean':>6} {'Fail':>6} {'Total':>6}")
    print("-" * 70)
    g_clean = g_fail = 0
    for s in all_scene_summaries:
        print(f"{s['scene']:<45} {s['clean']:>6} {s['fail']:>6} {s['total']:>6}")
        g_clean += s["clean"]
        g_fail += s["fail"]
    print("-" * 70)
    print(f"{'TOTAL':<45} {g_clean:>6} {g_fail:>6} {g_clean+g_fail:>6}")
    print(f"{'='*70}")

    # Save grand summary
    summary_path = os.path.join(args.out_root, "grand_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "settings": {
                "cx_cutoff": args.cx_cutoff,
                "gap_max": args.gap_max,
                "reappear_max": args.reappear_max,
                "crop_top_ratio": args.crop_top_ratio,
            },
            "scenes": all_scene_summaries,
            "total_clean": g_clean,
            "total_fail": g_fail,
            "total_videos": g_clean + g_fail,
        }, f, indent=2)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
