"""SAM3 trajectory-based bottle hallucination reward scorer."""

import os
from typing import Dict, List, Optional

import torch
from PIL import Image

from fastvideo.reward.base import RewardScorer, build_sam3_predictor
from fastvideo.utils.logging_ import main_print


class BottleHallucinationRewardScorer(RewardScorer):
    """Binary reward for put_bottles_dustbin: two-stage trajectory-based hallucination check.

    Uses SAM3 to track "bottle" across all video frames, builds per-object
    trajectories, then applies a two-stage approach:

    Stage 1 — Filter SAM3 tracking errors:
      - Remove spurious objects (first detected after frame 5)
      - Split trajectories at gaps > filter_max_gap frames, keep first segment
        (detections after a large gap are likely tracking the gripper, not the bottle)

    Stage 2 — 3-criteria hallucination check (on cleaned trajectories):
      (a) all_placed:  >= 3 real bottles survive filtering AND all reach cx < cx_cutoff
      (b) monotonic:   active count never increases (brief spikes <= spike_max suppressed)
      (c) no_merge:    no two bottles share the same placed_frame

    Reward = 1.0 if all criteria pass (CLEAN), 0.0 otherwise (FAIL).

    Validated on 80 base-model rollout videos: 100% accuracy against ground truth
    (data/outputs/bottle_hall_50steps/ground_truth.json).

    Uses the base Sam3VideoPredictor (NOT MultiGPU) to avoid DDP env corruption.
    """

    def __init__(
        self,
        prompt: str = "bottle",
        crop_top_ratio: float = 2 / 3,
        cx_cutoff: float = 0.26,
        spike_max: int = 3,
        filter_max_gap: int = 5,
        device_id: int = 0,
    ):
        self._prompt = prompt
        self._crop_top_ratio = crop_top_ratio
        self._cx_cutoff = cx_cutoff
        self._spike_max = spike_max
        self._filter_max_gap = filter_max_gap

        # Build SAM3 with DDP env vars masked to avoid gloo/nccl conflicts.
        main_print(f"  Loading SAM3 VideoPredictor (bottles) on cuda:{device_id} ...")
        self._predictor = build_sam3_predictor(device_id)
        main_print("  SAM3 VideoPredictor (bottles) loaded.")

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _truncate_trajectory(traj, cx_cutoff):
        """Truncate trajectory at the first frame where cx < cx_cutoff."""
        for i, pt in enumerate(traj):
            if pt["cx"] < cx_cutoff:
                return traj[:i], pt["frame"]
        return traj, None

    def _extract_trajectories(self, video_path: str, frames_dir: str = None):
        """Run SAM3 tracking and return per-object trajectories dict.

        If *frames_dir* is provided, use pre-extracted (and optionally
        pre-cropped) JPEG frames instead of decoding from *video_path*.

        Returns:
            objects: {obj_id_str: [{frame, cx, cy, x, y, w, h, prob}, ...]}
            frame_results: raw SAM3 per-frame results (for video rendering)
            total_frames: int
            fps: float
            crop_h: int | None
            jpeg_dir: str — JPEG directory (caller must clean up via
                shutil.rmtree when done, unless it was passed in as frames_dir)
        """
        import cv2
        from fastvideo.reward.sam3_utils import extract_frames_to_jpeg, track_prompt

        if frames_dir is not None:
            # Frames already extracted (and cropped) by caller
            jpeg_dir = frames_dir
            jpeg_files = sorted(f for f in os.listdir(jpeg_dir) if f.endswith(".jpg"))
            total_frames = len(jpeg_files)
            fps = 16.0  # default; only matters for debug video rendering
            crop_h = None  # already cropped
        else:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            crop_h = None
            if self._crop_top_ratio < 1.0:
                crop_h = (int(h_full * self._crop_top_ratio)) // 16 * 16

            jpeg_dir = extract_frames_to_jpeg(video_path, crop_h=crop_h)

        frame_results = track_prompt(self._predictor, jpeg_dir, self._prompt)

        # Build per-object trajectories
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

        # NOTE: jpeg_dir is returned for reuse by _render_debug_video.
        # Caller is responsible for cleanup (shutil.rmtree).
        return objects, frame_results, total_frames, fps, crop_h, jpeg_dir

    @staticmethod
    def _filter_trajectory(traj, max_gap, min_detections=3):
        """Filter SAM3 tracking errors from a single trajectory.

        Splits trajectory at gaps > max_gap frames and keeps the first
        continuous segment.  Detections after a large gap are likely tracking
        the gripper or scene noise, not the real bottle.

        Returns cleaned trajectory list (may be empty).
        """
        if len(traj) < min_detections:
            return []

        # Split at large gaps, keep first segment
        segments = [[traj[0]]]
        for i in range(1, len(traj)):
            gap = traj[i]["frame"] - traj[i - 1]["frame"] - 1
            if gap > max_gap:
                segments.append([traj[i]])
            else:
                segments[-1].append(traj[i])

        first = segments[0]
        return first if len(first) >= min_detections else []

    def _analyze(self, objects, total_frames):
        """Two-stage hallucination analysis on trajectory data.

        Stage 1: Filter SAM3 errors (gap-based trajectory splitting).
        Stage 2: 3-criteria check (all_placed, monotonic, no_merge).

        Returns (is_clean: bool, fail_reasons: list[str], analysis: dict).
        """
        cx_cutoff = self._cx_cutoff
        spike_max = self._spike_max
        max_gap = self._filter_max_gap

        # ── Pre-filter: only objects appearing in first 5 frames ─────────
        real_objects = {}
        for oid, traj in objects.items():
            if traj and traj[0]["frame"] <= 5:
                real_objects[oid] = traj

        # ── Stage 1: filter SAM3 errors ──────────────────────────────────
        cleaned = {}
        filtered_out = {}
        for oid, traj in real_objects.items():
            clean_traj = self._filter_trajectory(traj, max_gap)
            if clean_traj:
                cleaned[oid] = clean_traj
            else:
                filtered_out[oid] = len(traj)

        # ── Stage 2: truncate + 3 criteria ───────────────────────────────
        object_info = {}
        active_frames = {}
        for oid, traj in cleaned.items():
            truncated, placed_frame = self._truncate_trajectory(traj, cx_cutoff)
            object_info[oid] = {
                "placed_frame": placed_frame,
                "truncated_frames": len(truncated),
                "original_frames": len(real_objects.get(oid, [])),
                "cleaned_frames": len(traj),
            }
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

        # (b) monotonic — active count must not increase
        monotonic_ok = True
        committed = active_counts[0] if active_counts else 0
        i = 0
        n = len(active_counts)
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

        fail_reasons = []
        if not all_placed:
            not_placed = [oid for oid, pf in placed_map.items() if pf is None]
            fail_reasons.append(
                f"not_all_placed(cleaned={n_cleaned},placed={n_placed},"
                f"not_placed={not_placed},filtered_out={filtered_out})")
        if not monotonic_ok:
            fail_reasons.append("count_increase")
        if not merge_ok:
            fail_reasons.append("merge")

        return is_clean, fail_reasons, {
            "object_info": object_info,
            "num_real_objects": len(real_objects),
            "n_cleaned": n_cleaned,
            "n_placed": n_placed,
            "filtered_out": filtered_out,
            "all_placed": all_placed,
            "monotonic_ok": monotonic_ok,
            "merge_ok": merge_ok,
        }

    def _render_debug_video(self, video_path, frame_results, objects, total_frames,
                            fps, crop_h, output_path, jpeg_dir=None):
        """Render annotated tracking video with bboxes, trails, and cutoff line.

        If *jpeg_dir* is provided, reuse the existing JPEG frames (avoids
        redundant extraction).  Otherwise extract from *video_path*.
        """
        import cv2
        from fastvideo.reward.sam3_utils import extract_frames_to_jpeg, save_video_libx264
        import shutil

        # Pre-compute placed_frame per object
        placed_frames_map = {}
        for oid, traj in objects.items():
            for pt in traj:
                if pt["cx"] < self._cx_cutoff:
                    placed_frames_map[oid] = pt["frame"]
                    break

        OBJ_COLORS_BGR = [
            (0, 0, 255), (0, 200, 0), (255, 150, 0),
            (0, 200, 255), (255, 0, 200),
        ]

        own_jpeg = jpeg_dir is None
        if own_jpeg:
            jpeg_dir = extract_frames_to_jpeg(video_path, crop_h=crop_h)
        trail_history = {}
        out_frames = []
        current_placed = set()

        for fi in range(total_frames):
            for oid, pf in placed_frames_map.items():
                if fi >= pf:
                    current_placed.add(oid)

            frame_bgr = cv2.imread(os.path.join(jpeg_dir, f"{fi:06d}.jpg"))
            ann = frame_bgr.copy()
            h, w = ann.shape[:2]

            # Draw cutoff line
            cutoff_x = int(self._cx_cutoff * w)
            cv2.line(ann, (cutoff_x, 0), (cutoff_x, h), (0, 255, 255), 1, cv2.LINE_AA)

            fr = frame_results[fi]
            for i, obj_id in enumerate(fr["obj_ids"]):
                oid_str = str(obj_id)
                bx, by, bw, bh = fr["boxes_xywh"][i]
                x0, y0 = int(bx * w), int(by * h)
                x1, y1 = int((bx + bw) * w), int((by + bh) * h)
                cx_px, cy_px = (x0 + x1) // 2, (y0 + y1) // 2
                prob = fr["probs"][i]

                if oid_str in current_placed:
                    color = (120, 120, 120)
                    cv2.rectangle(ann, (x0, y0), (x1, y1), color, 1)
                    label = f"id={obj_id} PLACED"
                else:
                    color = OBJ_COLORS_BGR[obj_id % len(OBJ_COLORS_BGR)]
                    cv2.rectangle(ann, (x0, y0), (x1, y1), color, 2)
                    label = f"id={obj_id} {prob:.2f}"
                    if oid_str not in trail_history:
                        trail_history[oid_str] = []
                    trail_history[oid_str].append((cx_px, cy_px))

                (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(ann, (x0, max(y0 - th_t - 6, 0)),
                              (x0 + tw + 4, y0), color, -1)
                cv2.putText(ann, label, (x0 + 2, max(y0 - 3, th_t + 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Draw trails
            for oid_str, trail in trail_history.items():
                if len(trail) < 2:
                    continue
                color = OBJ_COLORS_BGR[int(oid_str) % len(OBJ_COLORS_BGR)]
                for k in range(1, len(trail)):
                    alpha = 0.3 + 0.7 * (k / len(trail))
                    c = tuple(int(v * alpha) for v in color)
                    cv2.line(ann, trail[k - 1], trail[k], c, 1, cv2.LINE_AA)

            cv2.putText(ann, f"frame {fi}/{total_frames}",
                        (w - 190, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1)
            out_frames.append(ann)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        save_video_libx264(out_frames, output_path, fps)
        if own_jpeg:
            shutil.rmtree(jpeg_dir, ignore_errors=True)

    # -- RewardScorer interface -----------------------------------------

    def score(
        self, prompt: str, first_frame: Image.Image,
        video_path: Optional[str] = None,
        debug_save_path: Optional[str] = None,
        frames_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        if video_path is None:
            raise ValueError("video_path is required for BottleHallucinationRewardScorer")

        import shutil

        own_jpeg = frames_dir is None  # True → scorer creates its own jpeg_dir
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        jpeg_dir = None

        try:
            objects, frame_results, total_frames, fps, crop_h, jpeg_dir = \
                self._extract_trajectories(video_path, frames_dir=frames_dir)
            is_clean, fail_reasons, analysis = self._analyze(objects, total_frames)
        except Exception as exc:
            if jpeg_dir and own_jpeg:
                shutil.rmtree(jpeg_dir, ignore_errors=True)
            main_print(f"  [bottle hall reward] failed: {exc}")
            return {
                "reward": 0.0, "pass": False,
                "_response_text": f"[ERROR] {exc}",
            }

        reward = 1.0 if is_clean else 0.0
        tag = "CLEAN" if is_clean else "FAIL"
        reasons_str = ", ".join(fail_reasons) if fail_reasons else ""
        response_text = f"[{tag}] real_objs={analysis['num_real_objects']} {reasons_str}"

        # Render debug video if requested (reuses jpeg_dir from tracking)
        if debug_save_path:
            debug_dir = os.path.dirname(debug_save_path)
            final_video = os.path.join(debug_dir, f"{video_stem}_{tag}.mp4")
            try:
                self._render_debug_video(
                    video_path, frame_results, objects,
                    total_frames, fps, crop_h, final_video,
                    jpeg_dir=jpeg_dir,
                )
            except Exception as exc:
                main_print(f"  [bottle hall reward] render failed: {exc}")
                final_video = None
        else:
            final_video = None

        # Cleanup jpeg_dir only if we created it (not when caller provided frames_dir)
        if jpeg_dir and own_jpeg:
            shutil.rmtree(jpeg_dir, ignore_errors=True)

        return {
            "reward": reward,
            "pass": is_clean,
            "fail_reasons": fail_reasons,
            "num_real_objects": analysis["num_real_objects"],
            "annotated_video": final_video,
            "_response_text": response_text,
        }

    def shutdown(self):
        """Release SAM3 predictor resources."""
        if hasattr(self, '_predictor') and self._predictor is not None:
            self._predictor.shutdown()
            self._predictor = None
