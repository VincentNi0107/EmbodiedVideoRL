"""SAM3 size-ranking reward scorer for blocks_ranking_size."""

import os
import statistics
from typing import Dict, List, Optional

import torch
from PIL import Image

from fastvideo.reward.base import RewardScorer, build_sam3_predictor
from fastvideo.utils.logging_ import main_print


class BlockSizeRankingRewardScorer(RewardScorer):
    """Binary reward for blocks_ranking_size: count + spatial ordering check.

    Uses SAM3 to track "block" across all video frames, calibrates relative
    sizes from initial-frame bbox areas, then verifies:
      (a) correct_count:      exactly N block objects detected
      (b) no_hallucination:   count stays constant (with occlusion suppression)
      (c) final_all_visible:  all N blocks visible in final check window
      (d) spatial_ordering:   largest-to-smallest left-to-right (by median cx)
      (e) center_placement:   mean cx of all blocks near frame centre

    Reward = 1.0 if all criteria pass (CLEAN), 0.0 otherwise (FAIL).

    Uses the base Sam3VideoPredictor (NOT MultiGPU) to avoid DDP env corruption.
    """

    def __init__(
        self,
        prompt: str = "block",
        expected_count: int = 3,
        crop_top_ratio: float = 2 / 3,
        check_window_frac: float = 0.20,
        min_visible_frac: float = 0.50,
        ordering_margin: float = 0.03,
        center_margin: float = 0.15,
        gap_max: int = 5,
        occlusion_gap_max: int = 5,
        occlusion_pos_thr: float = 0.15,
        device_id: int = 0,
    ):
        self._prompt = prompt
        self._expected_count = expected_count
        self._crop_top_ratio = crop_top_ratio
        self._check_window_frac = check_window_frac
        self._min_visible_frac = min_visible_frac
        self._ordering_margin = ordering_margin
        self._center_margin = center_margin
        self._gap_max = gap_max
        self._occlusion_gap_max = occlusion_gap_max
        self._occlusion_pos_thr = occlusion_pos_thr

        # Build SAM3 with DDP env vars masked to avoid gloo/nccl conflicts.
        main_print(f"  Loading SAM3 VideoPredictor (blocks_size) on cuda:{device_id} ...")
        self._predictor = build_sam3_predictor(device_id)
        main_print("  SAM3 VideoPredictor (blocks_size) loaded.")

    # -- helpers --------------------------------------------------------

    def _extract_trajectories(self, video_path: str):
        """Run SAM3 tracking and return per-object trajectories dict.

        Returns:
            objects: {obj_id_str: [{frame, cx, cy, x, y, w, h, area, prob}, ...]}
            frame_results: raw SAM3 per-frame results
            total_frames: int
            fps: float
            crop_h: int | None
        """
        import cv2
        from fastvideo.reward.sam3_utils import extract_frames_to_jpeg, track_prompt
        import shutil

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

        # Build per-object trajectories with area
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
                    "area": float(bw * bh),
                    "prob": float(fr["probs"][i]),
                })

        shutil.rmtree(jpeg_dir, ignore_errors=True)
        return objects, frame_results, total_frames, fps, crop_h

    def _analyze(self, objects, total_frames):
        """Apply 5-criteria size-ranking analysis.

        Returns (is_clean: bool, fail_reasons: list[str], analysis: dict).
        """
        expected = self._expected_count
        gap_max = self._gap_max
        check_window_frac = self._check_window_frac
        min_visible_frac = self._min_visible_frac
        ordering_margin = self._ordering_margin
        center_margin = self._center_margin

        # Filter real vs spurious objects (first appearance <= frame 5)
        real_objects = {}
        for oid, traj in objects.items():
            if traj and traj[0]["frame"] <= 5:
                real_objects[oid] = traj

        # -- (a) correct_count --
        count_ok = len(real_objects) == expected

        # -- Build per-frame visible count --
        frame_counts = [0] * total_frames
        for oid, traj in real_objects.items():
            for pt in traj:
                frame_counts[pt["frame"]] += 1

        # -- (b) no_hallucination: count stays at expected (with suppression) --
        hall_ok = True
        # Check for persistent count deviations from expected
        # Allow brief deviations (occlusion: count < expected, artifact: count > expected)
        committed = expected
        i = 0
        n = len(frame_counts)
        while i < n:
            c = frame_counts[i]
            if c != committed:
                # Start of a deviation
                j = i
                while j < n and frame_counts[j] != committed:
                    j += 1
                dev_len = j - i
                if dev_len > gap_max:
                    # Check if this is a legitimate count change or hallucination
                    if c > committed:
                        # Count increased — hallucination (extra object appeared)
                        hall_ok = False
                    # Count decreased — might be occlusion; only flag if it doesn't recover
                    elif j >= n:
                        # Never recovers — could be a block disappearing (bad)
                        hall_ok = False
                i = j if j > i else i + 1
            else:
                i += 1

        # -- (c) final_all_visible --
        check_start = max(0, total_frames - int(total_frames * check_window_frac))
        check_len = total_frames - check_start
        visible_count = sum(1 for fi in range(check_start, total_frames)
                           if frame_counts[fi] == expected)
        visible_frac = visible_count / check_len if check_len > 0 else 0.0
        final_visible_ok = visible_frac >= min_visible_frac

        # -- (d) spatial_ordering (largest-left, smallest-right) --
        # First, calibrate size from initial frames (0-5)
        initial_areas = {}  # oid -> list of areas
        for oid, traj in real_objects.items():
            areas = [pt["area"] for pt in traj if pt["frame"] <= 5]
            if areas:
                initial_areas[oid] = statistics.median(areas)

        # Sort by initial area descending -> size_rank[oid] = 0 (largest), 1, 2 (smallest)
        sorted_by_size = sorted(initial_areas.items(), key=lambda x: x[1], reverse=True)
        size_rank = {oid: rank for rank, (oid, _) in enumerate(sorted_by_size)}
        size_labels = {oid: ["L", "M", "S"][rank] if rank < 3 else f"#{rank}"
                       for oid, rank in size_rank.items()}

        # Compute median cx per object in check window
        final_cx = {}
        for oid, traj in real_objects.items():
            cxs = [pt["cx"] for pt in traj if pt["frame"] >= check_start]
            if cxs:
                final_cx[oid] = statistics.median(cxs)

        ordering_ok = False
        ordering_detail = ""
        if count_ok and len(final_cx) == expected and len(size_rank) == expected:
            # Check: for objects sorted by size_rank (0=largest first),
            # their final cx should be strictly increasing (left-to-right)
            ordered_oids = sorted(size_rank.keys(), key=lambda oid: size_rank[oid])
            cx_values = [final_cx[oid] for oid in ordered_oids]
            ordering_ok = True
            for k in range(1, len(cx_values)):
                if cx_values[k] - cx_values[k - 1] < ordering_margin:
                    ordering_ok = False
                    break
            ordering_detail = " ".join(
                f"{size_labels[oid]}={final_cx[oid]:.3f}" for oid in ordered_oids
            )
        else:
            ordering_detail = f"cannot_check(count={len(final_cx)},rank={len(size_rank)})"

        # -- (e) center_placement --
        center_ok = False
        mean_cx = 0.0
        if final_cx:
            mean_cx = statistics.mean(final_cx.values())
            center_ok = abs(mean_cx - 0.5) <= center_margin

        is_clean = count_ok and hall_ok and final_visible_ok and ordering_ok and center_ok

        fail_reasons = []
        if not count_ok:
            fail_reasons.append(f"count({len(real_objects)}!={expected})")
        if not hall_ok:
            fail_reasons.append("hallucination")
        if not final_visible_ok:
            fail_reasons.append(f"not_all_visible({visible_frac:.2f}<{min_visible_frac})")
        if not ordering_ok:
            fail_reasons.append(f"wrong_order({ordering_detail})")
        if not center_ok:
            fail_reasons.append(f"not_centered(mean_cx={mean_cx:.3f})")

        return is_clean, fail_reasons, {
            "num_real_objects": len(real_objects),
            "count_ok": count_ok,
            "hall_ok": hall_ok,
            "final_visible_ok": final_visible_ok,
            "visible_frac": visible_frac,
            "ordering_ok": ordering_ok,
            "ordering_detail": ordering_detail,
            "center_ok": center_ok,
            "mean_cx": mean_cx,
            "size_rank": size_rank,
            "size_labels": size_labels,
            "initial_areas": initial_areas,
            "final_cx": final_cx,
        }

    def _render_debug_video(self, video_path, frame_results, objects, total_frames,
                            fps, crop_h, output_path, analysis):
        """Render annotated tracking video with bboxes, size labels, and ordering info."""
        import cv2
        from fastvideo.reward.sam3_utils import extract_frames_to_jpeg, save_video_libx264
        import shutil

        OBJ_COLORS_BGR = [
            (0, 0, 255), (0, 200, 0), (255, 150, 0),
            (0, 200, 255), (255, 0, 200),
        ]

        size_labels = analysis.get("size_labels", {})
        final_cx = analysis.get("final_cx", {})
        check_start = max(0, total_frames - int(total_frames * self._check_window_frac))

        jpeg_dir = extract_frames_to_jpeg(video_path, crop_h=crop_h)
        trail_history = {}
        out_frames = []

        for fi in range(total_frames):
            frame_bgr = cv2.imread(os.path.join(jpeg_dir, f"{fi:06d}.jpg"))
            ann = frame_bgr.copy()
            h, w = ann.shape[:2]

            in_check = fi >= check_start

            fr = frame_results[fi]
            for i, obj_id in enumerate(fr["obj_ids"]):
                oid_str = str(obj_id)
                bx, by, bw, bh = fr["boxes_xywh"][i]
                x0, y0 = int(bx * w), int(by * h)
                x1, y1 = int((bx + bw) * w), int((by + bh) * h)
                cx_px, cy_px = (x0 + x1) // 2, (y0 + y1) // 2
                prob = fr["probs"][i]
                area = bw * bh

                color = OBJ_COLORS_BGR[obj_id % len(OBJ_COLORS_BGR)]
                thickness = 2 if not in_check else 3
                cv2.rectangle(ann, (x0, y0), (x1, y1), color, thickness)

                slabel = size_labels.get(oid_str, "?")
                label = f"{slabel} id={obj_id} a={area:.4f}"

                if oid_str not in trail_history:
                    trail_history[oid_str] = []
                trail_history[oid_str].append((cx_px, cy_px))

                (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(ann, (x0, max(y0 - th_t - 6, 0)),
                              (x0 + tw + 4, y0), color, -1)
                cv2.putText(ann, label, (x0 + 2, max(y0 - 3, th_t + 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # In check window, draw vertical lines at each block's final cx
            if in_check and final_cx:
                for oid_str, cx_val in final_cx.items():
                    cx_x = int(cx_val * w)
                    slabel = size_labels.get(oid_str, "?")
                    color = OBJ_COLORS_BGR[int(oid_str) % len(OBJ_COLORS_BGR)]
                    cv2.line(ann, (cx_x, 0), (cx_x, h), color, 1, cv2.LINE_AA)
                    cv2.putText(ann, slabel, (cx_x + 3, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
            cv2.putText(ann, f"frame {fi}/{total_frames}  blocks={count} {zone}",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1)
            out_frames.append(ann)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        save_video_libx264(out_frames, output_path, fps)
        shutil.rmtree(jpeg_dir, ignore_errors=True)

    # -- RewardScorer interface -----------------------------------------

    def score(
        self, prompt: str, first_frame: Image.Image,
        video_path: Optional[str] = None,
        debug_save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        if video_path is None:
            raise ValueError("video_path is required for BlockSizeRankingRewardScorer")

        video_stem = os.path.splitext(os.path.basename(video_path))[0]

        try:
            objects, frame_results, total_frames, fps, crop_h = \
                self._extract_trajectories(video_path)
            is_clean, fail_reasons, analysis = self._analyze(objects, total_frames)
        except Exception as exc:
            main_print(f"  [block size reward] failed: {exc}")
            return {
                "reward": 0.0, "pass": False,
                "_response_text": f"[ERROR] {exc}",
            }

        reward = 1.0 if is_clean else 0.0
        tag = "CLEAN" if is_clean else "FAIL"
        reasons_str = ", ".join(fail_reasons) if fail_reasons else ""
        response_text = (
            f"[{tag}] real_objs={analysis['num_real_objects']} "
            f"order={analysis['ordering_detail']} "
            f"mean_cx={analysis['mean_cx']:.3f} {reasons_str}"
        )

        # Render debug video if requested
        if debug_save_path:
            debug_dir = os.path.dirname(debug_save_path)
            final_video = os.path.join(debug_dir, f"{video_stem}_{tag}.mp4")
            try:
                self._render_debug_video(
                    video_path, frame_results, objects,
                    total_frames, fps, crop_h, final_video, analysis,
                )
            except Exception as exc:
                main_print(f"  [block size reward] render failed: {exc}")
                final_video = None
        else:
            final_video = None

        return {
            "reward": reward,
            "pass": is_clean,
            "fail_reasons": fail_reasons,
            "num_real_objects": analysis["num_real_objects"],
            "ordering_detail": analysis["ordering_detail"],
            "mean_cx": analysis["mean_cx"],
            "annotated_video": final_video,
            "_response_text": response_text,
        }

    def shutdown(self):
        """Release SAM3 predictor resources."""
        if hasattr(self, '_predictor') and self._predictor is not None:
            self._predictor.shutdown()
            self._predictor = None
