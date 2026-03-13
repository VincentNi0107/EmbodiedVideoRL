"""SAM3 hallucination detection reward scorer (constant object count)."""

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

from fastvideo.reward.base import RewardScorer, build_sam3_predictor
from fastvideo.utils.logging_ import main_print


class HallucinationRewardScorer(RewardScorer):
    """Binary reward based on SAM3 hallucination detection.

    Reward is 1.0 if no hallucination frames are detected, 0.0 otherwise.
    Uses fastvideo.reward.hallucination_process.process_video() for per-video tracking.

    Uses the base Sam3VideoPredictor (NOT Sam3VideoPredictorMultiGPU) to avoid
    corruption of MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE env vars that the
    MultiGPU variant overwrites in its __init__.
    """

    def __init__(
        self,
        prompts: Optional[List[str]] = None,
        expected_counts: Optional[Dict[str, int]] = None,
        crop_top_ratio: float = 2 / 3,
        occlusion_gap_max: int = 5,
        occlusion_pos_thr: float = 0.15,
        duplication_spike_max: int = 0,
        device_id: int = 0,
    ):
        if prompts is None:
            prompts = ["red block", "green block", "blue block"]
        if expected_counts is None:
            expected_counts = {p: 1 for p in prompts}

        self._prompts = prompts
        self._expected_counts = expected_counts
        self._crop_top_ratio = crop_top_ratio
        self._occlusion_gap_max = occlusion_gap_max
        self._occlusion_pos_thr = occlusion_pos_thr
        self._duplication_spike_max = duplication_spike_max

        # Build SAM3 with DDP env vars masked to avoid gloo/nccl conflicts.
        main_print(f"  Loading SAM3 VideoPredictor on cuda:{device_id} ...")
        self._predictor = build_sam3_predictor(device_id)
        main_print("  SAM3 VideoPredictor loaded.")

    def score(
        self, prompt: str, first_frame: Image.Image,
        video_path: Optional[str] = None,
        debug_save_path: Optional[str] = None,
        frames_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        if video_path is None:
            raise ValueError("video_path is required for HallucinationRewardScorer")

        from fastvideo.reward.hallucination_process import process_video

        video_stem = os.path.splitext(os.path.basename(video_path))[0]

        # When no debug output requested, skip video rendering entirely
        skip_render = (debug_save_path is None)

        if not skip_render:
            debug_dir = os.path.dirname(debug_save_path)
            output_video = os.path.join(debug_dir, f"{video_stem}_hallucination_tmp.mp4")
        else:
            output_video = None

        try:
            summary = process_video(
                input_path=str(Path(video_path).resolve()),
                output_video=output_video,
                output_csv=None,
                output_json=None,
                prompts=self._prompts,
                expected_counts=self._expected_counts,
                crop_top_ratio=self._crop_top_ratio,
                predictor=self._predictor,
                occlusion_gap_max=self._occlusion_gap_max,
                occlusion_pos_thr=self._occlusion_pos_thr,
                duplication_spike_max=self._duplication_spike_max,
                quiet=True,
                skip_render=skip_render,
                frames_dir=frames_dir,
                return_tracking_data=True,
            )
        except Exception as exc:
            main_print(f"  [hall reward] process_video failed: {exc}")
            return {
                "reward": 0.0, "pass": False,
                "hall_frames": -1, "total_frames": -1,
                "_response_text": f"[ERROR] {exc}",
            }

        # Compute motion_score from tracking data (for best-of-N selection)
        tracking_data = summary.pop("_tracking_data", None)
        motion_score = float("inf")
        if tracking_data is not None:
            from fastvideo.reward.sam3_utils import compute_motion_score_from_objects
            objects = {}
            for pr in self._prompts:
                for fi, fr in enumerate(tracking_data[pr]):
                    for i, obj_id in enumerate(fr["obj_ids"]):
                        key = f"{pr}_{obj_id}"
                        if key not in objects:
                            objects[key] = []
                        bx, by, bw, bh = fr["boxes_xywh"][i]
                        objects[key].append({
                            "frame": fi,
                            "cx": float(bx + bw / 2),
                            "cy": float(by + bh / 2),
                        })
            motion_score = compute_motion_score_from_objects(objects)

        n_hall = summary["total_hallucination_frames"]
        total = summary["frame_count"]
        is_clean = (n_hall == 0)
        reward = 1.0 if is_clean else 0.0

        # Rename annotated video to include result tag: CLEAN or HALL
        tag = "CLEAN" if is_clean else "HALL"
        if not skip_render and output_video:
            final_video = os.path.join(
                os.path.dirname(output_video), f"{video_stem}_{tag}.mp4",
            )
            try:
                os.rename(output_video, final_video)
            except OSError:
                final_video = output_video  # fallback if rename fails
        else:
            final_video = None

        display_tag = "CLEAN" if is_clean else f"HALL({n_hall}/{total})"
        response_text = f"[{display_tag}] hall_frames={n_hall}/{total}"

        return {
            "reward": reward,
            "pass": is_clean,
            "hall_frames": n_hall,
            "total_frames": total,
            "motion_score": motion_score,
            "annotated_video": final_video,
            "_response_text": response_text,
        }

    def shutdown(self):
        """Release SAM3 predictor resources."""
        if hasattr(self, '_predictor') and self._predictor is not None:
            self._predictor.shutdown()
            self._predictor = None
