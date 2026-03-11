"""SAM3 convergence-based bowl stacking reward scorer."""

import os
from typing import Dict, List, Optional

import torch
from PIL import Image

from fastvideo.reward.base import RewardScorer, build_sam3_predictor
from fastvideo.utils.logging_ import main_print


class BowlStackRewardScorer(RewardScorer):
    """Binary reward for stack_bowls_three: convergence + no-hallucination check.

    Uses SAM3 to track "bowl" across all video frames, builds per-object
    trajectories, then applies four criteria:
      (a) correct_initial_count: exactly N bowls detected in early frames
      (b) no_count_increase:     visible count monotonically non-increasing
                                  (brief spikes <= gap_max suppressed)
      (c) spatial_convergence:   in final check_window, all visible bowl centres
                                  within convergence_thr of each other (L-inf)
      (d) no_random_disappear:   no bowl disappears > reappear_max frames then
                                  reappears at a distant location

    Reward = 1.0 if all criteria pass (CLEAN), 0.0 otherwise (FAIL).

    Uses the base Sam3VideoPredictor (NOT MultiGPU) to avoid DDP env corruption.
    """

    def __init__(
        self,
        prompt: str = "bowl",
        initial_count: int = 3,
        crop_top_ratio: float = 2 / 3,
        convergence_thr: float = 0.30,
        check_window_frac: float = 0.20,
        gap_max: int = 5,
        reappear_max: int = 10,
        reappear_pos_thr: float = 0.15,
        device_id: int = 0,
    ):
        self._prompt = prompt
        self._initial_count = initial_count
        self._crop_top_ratio = crop_top_ratio
        self._convergence_thr = convergence_thr
        self._check_window_frac = check_window_frac
        self._gap_max = gap_max
        self._reappear_max = reappear_max
        self._reappear_pos_thr = reappear_pos_thr

        # Build SAM3 with DDP env vars masked to avoid gloo/nccl conflicts.
        main_print(f"  Loading SAM3 VideoPredictor (bowls) on cuda:{device_id} ...")
        self._predictor = build_sam3_predictor(device_id)
        main_print("  SAM3 VideoPredictor (bowls) loaded.")

    # -- helpers --------------------------------------------------------

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

    def _analyze(self, objects, total_frames):
        """Apply 5-criteria stacking analysis on trajectory data.

        Returns (is_clean: bool, fail_reasons: list[str], analysis: dict).
        """
        gap_max = self._gap_max
        reappear_max = self._reappear_max
        reappear_pos_thr = self._reappear_pos_thr
        convergence_thr = self._convergence_thr
        check_window_frac = self._check_window_frac
        initial_count = self._initial_count

        # Filter real vs spurious objects (first appearance <= frame 5)
        real_objects = {}
        spurious_objects = {}
        for oid, traj in objects.items():
            if traj and traj[0]["frame"] <= 5:
                real_objects[oid] = traj
            elif traj:
                spurious_objects[oid] = traj

        # -- (a) correct_initial_count --
        count_ok = len(real_objects) == initial_count

        # -- Build per-frame visible count (real objects only) --
        frame_counts = [0] * total_frames
        for oid, traj in real_objects.items():
            for pt in traj:
                frame_counts[pt["frame"]] += 1

        # -- (b) no_count_increase (monotonic non-increasing with both
        #        spike suppression AND drop suppression) --
        # Drop suppression: if the count dips briefly (≤ gap_max frames)
        # then recovers, don't lower committed (it was a SAM3 glitch).
        monotonic_ok = True
        committed = frame_counts[0] if frame_counts else 0
        i = 0
        n = len(frame_counts)
        while i < n:
            c = frame_counts[i]
            if c > committed:
                # Count increased — check if it's a brief spike
                j = i
                while j < n and frame_counts[j] > committed:
                    j += 1
                spike_len = j - i
                if not (spike_len <= gap_max and j < n):
                    monotonic_ok = False
                i = j
            elif c < committed:
                # Count dropped — check if it's a brief dip that recovers
                j = i
                while j < n and frame_counts[j] < committed:
                    j += 1
                dip_len = j - i
                if dip_len <= gap_max and j < n:
                    # Brief dip that recovers — suppress (don't lower committed)
                    i = j
                else:
                    # Genuine drop — commit to the new lower count
                    committed = c
                    i += 1
            else:
                i += 1

        # -- (c) spatial_convergence --
        check_start = max(0, total_frames - int(total_frames * check_window_frac))
        convergence_ok = True
        convergence_max_dist = 0.0

        # Collect all visible bowl centres in the check window
        window_centres = []  # list of (cx, cy) per frame (only frames with detections)
        for fi in range(check_start, total_frames):
            centres_this_frame = []
            for oid, traj in real_objects.items():
                for pt in traj:
                    if pt["frame"] == fi:
                        centres_this_frame.append((pt["cx"], pt["cy"]))
            if centres_this_frame:
                window_centres.append(centres_this_frame)

        # Check pairwise L-inf distance among centres within each frame
        if window_centres:
            for centres in window_centres:
                if len(centres) <= 1:
                    continue
                for a_idx in range(len(centres)):
                    for b_idx in range(a_idx + 1, len(centres)):
                        dx = abs(centres[a_idx][0] - centres[b_idx][0])
                        dy = abs(centres[a_idx][1] - centres[b_idx][1])
                        dist = max(dx, dy)
                        convergence_max_dist = max(convergence_max_dist, dist)
                        if dist > convergence_thr:
                            convergence_ok = False

        # -- (d) no_random_disappear --
        reappear_ok = True
        reappear_events = []
        for oid, traj in real_objects.items():
            if len(traj) < 2:
                continue
            for k in range(1, len(traj)):
                gap = traj[k]["frame"] - traj[k - 1]["frame"] - 1
                if gap > reappear_max:
                    # Check position shift
                    dx = abs(traj[k]["cx"] - traj[k - 1]["cx"])
                    dy = abs(traj[k]["cy"] - traj[k - 1]["cy"])
                    pos_shift = max(dx, dy)
                    if pos_shift > reappear_pos_thr:
                        reappear_ok = False
                        reappear_events.append(
                            f"obj{oid}:gap_f{traj[k-1]['frame']}-f{traj[k]['frame']}(shift={pos_shift:.3f})")

        # -- (e) no_late_objects: spurious objects appearing after frame 5
        #        that persist for > gap_max frames indicate hallucination --
        late_obj_ok = True
        late_obj_events = []
        for oid, traj in spurious_objects.items():
            if len(traj) > gap_max:
                late_obj_ok = False
                late_obj_events.append(
                    f"obj{oid}:first_f{traj[0]['frame']}_len{len(traj)}")

        # -- (f) no_area_anomaly: detect SAM3 merging bowls into one bbox.
        #        Merging creates abnormally LARGE bboxes. Use median area as
        #        reference: if max_area > threshold * median_area, flag it.
        #        (Small areas from gripper occlusion are benign and ignored.)
        area_anomaly_ok = True
        area_anomaly_events = []
        area_ratio_thr = 4.0  # max_area / median_area threshold
        for oid, traj in real_objects.items():
            areas = sorted([pt["w"] * pt["h"] for pt in traj])
            median_a = areas[len(areas) // 2]
            max_a = areas[-1]
            if median_a > 0:
                ratio = max_a / median_a
                if ratio > area_ratio_thr:
                    area_anomaly_ok = False
                    area_anomaly_events.append(
                        f"obj{oid}:area_ratio={ratio:.1f}x_median")

        is_clean = (count_ok and monotonic_ok and convergence_ok
                    and reappear_ok and late_obj_ok and area_anomaly_ok)

        fail_reasons = []
        if not count_ok:
            fail_reasons.append(f"initial_count({len(real_objects)}!={initial_count})")
        if not monotonic_ok:
            fail_reasons.append("count_increase")
        if not convergence_ok:
            fail_reasons.append(f"no_convergence(max_dist={convergence_max_dist:.3f})")
        if not reappear_ok:
            fail_reasons.append(f"reappear({', '.join(reappear_events)})")
        if not late_obj_ok:
            fail_reasons.append(f"late_object({', '.join(late_obj_events)})")
        if not area_anomaly_ok:
            fail_reasons.append(f"area_anomaly({', '.join(area_anomaly_events)})")

        return is_clean, fail_reasons, {
            "num_real_objects": len(real_objects),
            "count_ok": count_ok,
            "monotonic_ok": monotonic_ok,
            "convergence_ok": convergence_ok,
            "convergence_max_dist": convergence_max_dist,
            "reappear_ok": reappear_ok,
            "late_obj_ok": late_obj_ok,
            "area_anomaly_ok": area_anomaly_ok,
        }

    def _render_debug_video(self, video_path, frame_results, objects, total_frames,
                            fps, crop_h, output_path, jpeg_dir=None):
        """Render annotated tracking video with bboxes, trails, and convergence info.

        If *jpeg_dir* is provided, reuse the existing JPEG frames (avoids
        redundant extraction).  Otherwise extract from *video_path*.
        """
        import cv2
        from fastvideo.reward.sam3_utils import extract_frames_to_jpeg, save_video_libx264
        import shutil

        OBJ_COLORS_BGR = [
            (0, 0, 255), (0, 200, 0), (255, 150, 0),
            (0, 200, 255), (255, 0, 200),
        ]

        check_start = max(0, total_frames - int(total_frames * self._check_window_frac))

        own_jpeg = jpeg_dir is None
        if own_jpeg:
            jpeg_dir = extract_frames_to_jpeg(video_path, crop_h=crop_h)
        trail_history = {}
        out_frames = []

        for fi in range(total_frames):
            frame_bgr = cv2.imread(os.path.join(jpeg_dir, f"{fi:06d}.jpg"))
            ann = frame_bgr.copy()
            h, w = ann.shape[:2]

            # Draw check window boundary
            if fi == check_start:
                cv2.line(ann, (0, 0), (w, 0), (0, 255, 255), 3)

            in_check = fi >= check_start

            fr = frame_results[fi]
            for i, obj_id in enumerate(fr["obj_ids"]):
                oid_str = str(obj_id)
                bx, by, bw, bh = fr["boxes_xywh"][i]
                x0, y0 = int(bx * w), int(by * h)
                x1, y1 = int((bx + bw) * w), int((by + bh) * h)
                cx_px, cy_px = (x0 + x1) // 2, (y0 + y1) // 2
                prob = fr["probs"][i]

                color = OBJ_COLORS_BGR[obj_id % len(OBJ_COLORS_BGR)]
                thickness = 2 if not in_check else 3
                cv2.rectangle(ann, (x0, y0), (x1, y1), color, thickness)
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

            # Frame info
            count = len(fr["obj_ids"])
            zone = "CHECK" if in_check else ""
            cv2.putText(ann, f"frame {fi}/{total_frames}  bowls={count} {zone}",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
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
            raise ValueError("video_path is required for BowlStackRewardScorer")

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
            main_print(f"  [bowl stack reward] failed: {exc}")
            return {
                "reward": 0.0, "pass": False,
                "_response_text": f"[ERROR] {exc}",
            }

        reward = 1.0 if is_clean else 0.0
        tag = "CLEAN" if is_clean else "FAIL"
        reasons_str = ", ".join(fail_reasons) if fail_reasons else ""
        response_text = (
            f"[{tag}] real_objs={analysis['num_real_objects']} "
            f"conv_dist={analysis['convergence_max_dist']:.3f} {reasons_str}"
        )

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
                main_print(f"  [bowl stack reward] render failed: {exc}")
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
            "convergence_max_dist": analysis["convergence_max_dist"],
            "annotated_video": final_video,
            "_response_text": response_text,
        }

    def shutdown(self):
        """Release SAM3 predictor resources."""
        if hasattr(self, '_predictor') and self._predictor is not None:
            self._predictor.shutdown()
            self._predictor = None
