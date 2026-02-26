"""Test bottle hallucination detection against ground truth labels.

Two-stage approach:
  Stage 1 — Filter SAM3 tracking errors (deformed boxes, broken trajectories)
  Stage 2 — Simple 3-criteria hallucination check (all_placed, monotonic, no_merge)

Reads existing trajectory JSONs, runs the analysis, and compares predictions
to ground_truth.json.  No GPU needed.

Usage:
    python test_bottle_hallucination.py
    python test_bottle_hallucination.py --filter-max-gap 5   # sweep params
    python test_bottle_hallucination.py --gt data/outputs/bottle_hall_50steps/ground_truth.json
"""

import argparse
import json
import os
import glob


# ── Stage 1: SAM3 Error Filtering ────────────────────────────────────────────

def filter_sam3_errors(traj, max_gap, min_hw, min_detections):
    """Clean a single object trajectory by removing SAM3 tracking artifacts.

    Steps:
      1. Remove detections with deformed bounding boxes (h/w < min_hw).
         Real bottles are tall-and-narrow (h/w ~ 3.5–5.5); gripper artifacts
         are squarish (h/w < 2.0).
      2. Split trajectory at large gaps (> max_gap frames).
         A gap means SAM3 lost the object; detections after the gap may be
         tracking something else (gripper, noise).
      3. Keep the first continuous segment (most likely the real bottle).
      4. Require at least min_detections to be considered real.

    Returns:
        Cleaned trajectory (list of dicts) or empty list if unreliable.
    """
    # Step 1: filter deformed detections
    filtered = [pt for pt in traj
                if pt["w"] > 0 and pt["h"] / pt["w"] >= min_hw]

    if len(filtered) < min_detections:
        return []

    # Step 2: split at large gaps
    segments = [[filtered[0]]]
    for i in range(1, len(filtered)):
        gap = filtered[i]["frame"] - filtered[i - 1]["frame"] - 1
        if gap > max_gap:
            segments.append([filtered[i]])
        else:
            segments[-1].append(filtered[i])

    # Step 3: keep first segment
    first = segments[0]
    if len(first) < min_detections:
        return []

    return first


# ── Stage 2: Truncation + 3-Criteria Hallucination Check ─────────────────────

def truncate_trajectory(traj, cx_cutoff):
    """Truncate at first detection where cx < cx_cutoff (bottle reached dustbin)."""
    for i, pt in enumerate(traj):
        if pt["cx"] < cx_cutoff:
            return traj[:i], pt["frame"]
    return traj, None


def analyze_hallucination(traj_data, cx_cutoff, spike_max,
                          filter_max_gap, filter_min_hw, filter_min_detections):
    """Two-stage hallucination check.

    Stage 1 — Filter SAM3 errors:
        Remove deformed-box detections and split trajectories at gaps.

    Stage 2 — 3 criteria (all must pass for CLEAN):
        (a) all_placed:  >= 3 real bottles survive filtering AND all reach cx < cx_cutoff
        (b) monotonic:   active bottle count never increases (brief spikes <= spike_max OK)
        (c) no_merge:    no two bottles share the same placed_frame

    Returns dict with is_clean and diagnostic info.
    """
    total_frames = traj_data["frame_count"]
    objects = traj_data["objects"]

    # Pre-filter: only objects appearing in the first 5 frames
    real_objects = {}
    for oid, traj in objects.items():
        if traj and traj[0]["frame"] <= 5:
            real_objects[oid] = traj

    # ── Stage 1: filter SAM3 errors ──────────────────────────────────────
    cleaned = {}
    filtered_out = {}
    for oid, traj in real_objects.items():
        clean_traj = filter_sam3_errors(
            traj, filter_max_gap, filter_min_hw, filter_min_detections)
        if clean_traj:
            cleaned[oid] = clean_traj
        else:
            filtered_out[oid] = len(traj)

    # ── Stage 2: truncate + 3 criteria ───────────────────────────────────
    object_info = {}
    active_frames = {}
    for oid, traj in cleaned.items():
        truncated, placed_frame = truncate_trajectory(traj, cx_cutoff)
        object_info[oid] = {"placed_frame": placed_frame}
        for pt in truncated:
            fi = pt["frame"]
            if fi not in active_frames:
                active_frames[fi] = set()
            active_frames[fi].add(oid)

    active_counts = [len(active_frames.get(fi, set()))
                     for fi in range(total_frames)]

    # (a) all_placed — need >= 3 bottles, all placed
    placed_map = {oid: info["placed_frame"]
                  for oid, info in object_info.items()}
    n_cleaned = len(cleaned)
    n_placed = sum(1 for pf in placed_map.values() if pf is not None)
    all_placed = n_cleaned >= 3 and n_placed == n_cleaned

    # (b) monotonic — active count must not increase (brief spikes tolerated)
    monotonic_ok = True
    committed = active_counts[0] if active_counts else 0
    i, n = 0, len(active_counts)
    while i < n:
        c = active_counts[i]
        if c > committed:
            j = i
            while j < n and active_counts[j] > committed:
                j += 1
            if not (j - i <= spike_max and j < n):
                monotonic_ok = False
            i = j
        else:
            if c < committed:
                committed = c
            i += 1

    # (c) no_merge — no two bottles share the same placed_frame
    placed_vals = [pf for pf in placed_map.values() if pf is not None]
    merge_ok = len(placed_vals) == len(set(placed_vals))

    is_clean = all_placed and monotonic_ok and merge_ok

    # Build diagnostic info
    fail_reasons = []
    if not all_placed:
        not_placed = [oid for oid, pf in placed_map.items() if pf is None]
        fail_reasons.append(
            f"not_all_placed(cleaned={n_cleaned}, placed={n_placed}, "
            f"not_placed={not_placed}, filtered_out={filtered_out})")
    if not monotonic_ok:
        fail_reasons.append("count_increase")
    if not merge_ok:
        fail_reasons.append("merge")

    return {
        "is_clean": is_clean,
        "fail_reasons": fail_reasons,
        "n_real": len(real_objects),
        "n_cleaned": n_cleaned,
        "n_placed": n_placed,
        "filtered_out": filtered_out,
        "all_placed": all_placed,
        "monotonic_ok": monotonic_ok,
        "merge_ok": merge_ok,
    }


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(gt_labels, pred_labels):
    """Compute classification metrics.

    Positive class = CLEAN (true), Negative class = FAIL (false).
    """
    assert set(gt_labels.keys()) == set(pred_labels.keys()), "key mismatch"

    tp = fp = tn = fn = 0
    mismatches = []
    for stem in sorted(gt_labels.keys()):
        g = gt_labels[stem]
        p = pred_labels[stem]
        if g and p:
            tp += 1
        elif g and not p:
            fn += 1
            mismatches.append((stem, "GT=CLEAN  Pred=FAIL"))
        elif not g and p:
            fp += 1
            mismatches.append((stem, "GT=FAIL   Pred=CLEAN"))
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "total": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_clean": tp + fn,
        "gt_fail": tn + fp,
        "pred_clean": tp + fp,
        "pred_fail": tn + fn,
        "mismatches": mismatches,
    }


def print_metrics(m, params_str=""):
    print(f"\n{'=' * 70}")
    if params_str:
        print(f"Params: {params_str}")
    print(f"{'=' * 70}")

    print(f"\nConfusion Matrix (Positive=CLEAN, Negative=FAIL):")
    print(f"                    Predicted CLEAN   Predicted FAIL")
    print(f"  Actual CLEAN       TP = {m['tp']:>3}           FN = {m['fn']:>3}")
    print(f"  Actual FAIL        FP = {m['fp']:>3}           TN = {m['tn']:>3}")

    print(f"\n  GT:   {m['gt_clean']} CLEAN, {m['gt_fail']} FAIL")
    print(f"  Pred: {m['pred_clean']} CLEAN, {m['pred_fail']} FAIL")

    print(f"\n  Accuracy:  {m['accuracy']:.4f}  ({m['tp'] + m['tn']}/{m['total']})")
    print(f"  Precision: {m['precision']:.4f}  (of {m['pred_clean']} predicted CLEAN, {m['tp']} correct)")
    print(f"  Recall:    {m['recall']:.4f}  (of {m['gt_clean']} actual CLEAN, {m['tp']} found)")
    print(f"  F1:        {m['f1']:.4f}")

    if m["mismatches"]:
        print(f"\nMismatches ({len(m['mismatches'])}):")
        for stem, desc in m["mismatches"]:
            print(f"  {desc}  {stem}")
    else:
        print(f"\nNo mismatches — predictions perfectly match ground truth.")
    print(f"{'=' * 70}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test bottle hallucination detection against ground truth")
    parser.add_argument("--gt",
                        default="data/outputs/bottle_hall_50steps/ground_truth.json",
                        help="Path to ground_truth.json")
    parser.add_argument("--traj-root",
                        default=None,
                        help="Root dir with trajectory JSONs (default: same as GT dir)")
    parser.add_argument("--pattern",
                        default="robotwin_put_bottles_dustbin_*")
    # ── Stage 2: hallucination criteria params ──
    parser.add_argument("--cx-cutoff", type=float, default=0.26,
                        help="Truncate trajectory when cx < this (bottle reached dustbin)")
    parser.add_argument("--spike-max", type=int, default=3,
                        help="Max frames for brief count-spike suppression in monotonic check")
    # ── Stage 1: SAM3 error filtering params ──
    parser.add_argument("--filter-max-gap", type=int, default=5,
                        help="Split trajectory at gaps > this many frames")
    parser.add_argument("--filter-min-hw", type=float, default=0.0,
                        help="Min h/w ratio for a detection to be bottle-like (0=disabled)")
    parser.add_argument("--filter-min-detections", type=int, default=3,
                        help="Min detections for a trajectory to be considered real")
    # ── Verbosity ──
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-video diagnostics for mismatches")
    args = parser.parse_args()

    # Load ground truth
    with open(args.gt) as f:
        gt_data = json.load(f)
    gt_labels = gt_data["labels"]
    print(f"Ground truth: {args.gt}")
    print(f"  {sum(gt_labels.values())} CLEAN, "
          f"{len(gt_labels) - sum(gt_labels.values())} FAIL, "
          f"{len(gt_labels)} total")

    # Find trajectory JSONs
    traj_root = args.traj_root or os.path.dirname(args.gt)
    scene_dirs = sorted(glob.glob(os.path.join(traj_root, args.pattern)))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]

    # Run predictions
    pred_labels = {}
    diagnostics = {}
    for scene_dir in scene_dirs:
        traj_jsons = sorted(glob.glob(
            os.path.join(scene_dir, "*_trajectory.json")))
        for tj in traj_jsons:
            stem = os.path.basename(tj).replace("_trajectory.json", "")
            if stem not in gt_labels:
                continue
            with open(tj) as f:
                traj_data = json.load(f)
            result = analyze_hallucination(
                traj_data, args.cx_cutoff, args.spike_max,
                args.filter_max_gap, args.filter_min_hw,
                args.filter_min_detections,
            )
            pred_labels[stem] = result["is_clean"]
            diagnostics[stem] = result

    # Check coverage
    missing = set(gt_labels.keys()) - set(pred_labels.keys())
    if missing:
        print(f"\nWARNING: {len(missing)} GT entries have no trajectory JSON:")
        for s in sorted(missing):
            print(f"  {s}")

    # Only evaluate on videos present in both
    common = set(gt_labels.keys()) & set(pred_labels.keys())
    gt_common = {k: gt_labels[k] for k in common}
    pred_common = {k: pred_labels[k] for k in common}

    params_str = (f"cx_cutoff={args.cx_cutoff}, spike_max={args.spike_max}, "
                  f"filter_max_gap={args.filter_max_gap}, "
                  f"filter_min_hw={args.filter_min_hw}, "
                  f"filter_min_det={args.filter_min_detections}")
    m = compute_metrics(gt_common, pred_common)
    print_metrics(m, params_str)

    # Verbose: show diagnostics for mismatches
    if args.verbose and m["mismatches"]:
        print(f"\n{'─' * 70}")
        print("Mismatch diagnostics:")
        print(f"{'─' * 70}")
        for stem, desc in m["mismatches"]:
            d = diagnostics.get(stem, {})
            print(f"\n  {stem}")
            print(f"    {desc}")
            print(f"    real={d.get('n_real')}, cleaned={d.get('n_cleaned')}, "
                  f"placed={d.get('n_placed')}")
            print(f"    all_placed={d.get('all_placed')}, "
                  f"monotonic={d.get('monotonic_ok')}, "
                  f"merge={d.get('merge_ok')}")
            if d.get("filtered_out"):
                print(f"    filtered_out={d.get('filtered_out')}")
            if d.get("fail_reasons"):
                print(f"    reasons: {d.get('fail_reasons')}")


if __name__ == "__main__":
    main()
