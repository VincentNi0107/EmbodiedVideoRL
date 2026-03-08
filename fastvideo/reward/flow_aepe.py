"""Flow AEPE (Average End-Point Error) reward scorer.

Uses SEA-RAFT optical flow to compute forward-backward consistency (EPE)
between consecutive frames.  High EPE ⇒ temporal inconsistency (hallucination).

Based on WorldArena/video_quality/WorldArena/flow_aepe_metrics.py.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fastvideo.reward.base import RewardScorer

# ---------------------------------------------------------------------------
# SEA-RAFT paths
# ---------------------------------------------------------------------------
_WORLDARENA_THIRD_PARTY = Path("/gpfs/projects/p33048/WorldArena/video_quality/WorldArena/third_party")
_SEA_RAFT_DIR = _WORLDARENA_THIRD_PARTY / "SEA-RAFT"
_DEFAULT_CFG = _SEA_RAFT_DIR / "config/eval/spring-M.json"
_DEFAULT_CKPT = _WORLDARENA_THIRD_PARTY / "checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth"


def _ensure_searaft_on_path():
    """Add SEA-RAFT to sys.path so ``from core.raft import RAFT`` works."""
    p = str(_SEA_RAFT_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# EPE helpers  (from flow_aepe_metrics.py)
# ---------------------------------------------------------------------------

def _compute_epe(flow1: np.ndarray, flow2: np.ndarray,
                 crop: int = 30, error_threshold: float = 5.0):
    """Forward-backward consistency EPE between two flow fields.

    Returns (avg_epe, failure_mask).
    """
    H, W, _ = flow1.shape
    crop_H = H - crop
    crop_W = W - crop
    sx = (W - crop_W) // 2
    sy = (H - crop_H) // 2

    f1 = flow1[sy:sy + crop_H, sx:sx + crop_W, :]
    f2 = flow2[sy:sy + crop_H, sx:sx + crop_W, :]

    y_coords, x_coords = np.meshgrid(np.arange(crop_H), np.arange(crop_W), indexing="ij")

    wx1 = x_coords + f1[..., 0]
    wy1 = y_coords + f1[..., 1]

    wx1r = np.clip(np.round(wx1).astype(int), 0, crop_W - 1)
    wy1r = np.clip(np.round(wy1).astype(int), 0, crop_H - 1)

    f2_at_warp = f2[wy1r, wx1r, :]

    bx = wx1 + f2_at_warp[..., 0]
    by = wy1 + f2_at_warp[..., 1]

    epe = np.sqrt((bx - x_coords) ** 2 + (by - y_coords) ** 2)
    failure_mask = (epe > error_threshold).astype(np.uint8)
    return float(np.mean(epe)), failure_mask


def _compute_dynamic_raw(flow: np.ndarray) -> float:
    """Mean magnitude of the top-5% motion pixels."""
    rad = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    h, w = rad.shape
    cut = int(h * w * 0.05)
    if cut == 0:
        return 0.0
    top = np.sort(rad.reshape(-1))[-cut:]
    return float(np.mean(np.abs(top)))


def _soft_motion_score(score: float, thres: float, alpha: float = 5.0) -> float:
    x = score / thres - 1.0
    return 1.0 / (1.0 + np.exp(-alpha * x))


# ---------------------------------------------------------------------------
# Core metric class  (self-contained, no WorldArena imports)
# ---------------------------------------------------------------------------

class _FlowAEPEMetric:
    """Wraps SEA-RAFT to compute video-level flow AEPE score."""

    def __init__(self, cfg_path: str, ckpt_path: str, device: torch.device):
        _ensure_searaft_on_path()
        from core.raft import RAFT            # type: ignore
        from core.utils.utils import load_ckpt  # type: ignore
        from core.parser import parse_args     # type: ignore

        args_ns = argparse.Namespace(cfg=str(cfg_path), path=str(ckpt_path))
        args_ns = parse_args(args_ns)

        model = RAFT(args_ns)
        load_ckpt(model, args_ns.path)
        model.to(device)
        model.eval()

        self._model = model
        self._args = args_ns
        self._device = device

    # -- internals ----------------------------------------------------------

    def _load_image(self, frame_bgr_or_rgb: np.ndarray) -> torch.Tensor:
        """Convert HWC uint8 RGB array to 1×3×H×W float tensor on device."""
        img = torch.tensor(frame_bgr_or_rgb, dtype=torch.float32).permute(2, 0, 1)
        return img[None].to(self._device)

    def _compute_flow(self, img1: torch.Tensor, img2: torch.Tensor) -> np.ndarray:
        scale = self._args.scale
        i1 = F.interpolate(img1, scale_factor=2 ** scale, mode="bilinear", align_corners=False)
        i2 = F.interpolate(img2, scale_factor=2 ** scale, mode="bilinear", align_corners=False)
        with torch.amp.autocast(device_type="cuda", enabled=self._device.type == "cuda"):
            output = self._model(i1, i2, iters=self._args.iters, test_mode=True)
        flow = output["flow"][-1]
        flow_down = F.interpolate(
            flow, scale_factor=0.5 ** scale, mode="bilinear", align_corners=False
        ) * (0.5 ** scale)
        return flow_down.cpu().numpy().squeeze().transpose(1, 2, 0)

    def _dynamic_thres(self, img_tensor: torch.Tensor) -> float:
        scale = min(img_tensor.shape[-2:])
        return 6.0 * (scale / 256.0)

    # -- public API ----------------------------------------------------------

    def compute_video(self, frames_rgb: List[np.ndarray]) -> Dict[str, float]:
        """Compute flow-AEPE score for a list of RGB uint8 frames.

        Returns dict with keys:
          score        – 1/avg_epe (higher = better), weighted by dynamic_degree
          avg_epe      – raw average EPE
          dynamic_degree – soft motion score (0–1)
          per_pair_epe – list of per-pair EPE values
        """
        if len(frames_rgb) < 2:
            return {"score": 0.0, "avg_epe": 0.0, "dynamic_degree": 0.0, "per_pair_epe": []}

        epe_scores: List[float] = []
        dyn_raw_scores: List[float] = []

        with torch.no_grad():
            for f1, f2 in zip(frames_rgb[:-1], frames_rgb[1:]):
                i1 = self._load_image(f1)
                i2 = self._load_image(f2)
                flow_fwd = self._compute_flow(i1, i2)
                flow_bwd = self._compute_flow(i2, i1)
                epe, _ = _compute_epe(flow_fwd, flow_bwd)
                epe_scores.append(epe)
                dyn_raw_scores.append(_compute_dynamic_raw(flow_fwd))

        avg_epe = float(np.mean(epe_scores))
        score = 1.0 / avg_epe if avg_epe > 0 else float("inf")

        thres = self._dynamic_thres(i1)
        dyn_soft = [_soft_motion_score(s, thres) for s in dyn_raw_scores]
        dyn_degree = float(np.mean(dyn_soft)) if dyn_soft else 0.0

        if dyn_degree <= 0.1213:
            score = score * dyn_degree

        return {
            "score": float(score),
            "avg_epe": avg_epe,
            "dynamic_degree": dyn_degree,
            "per_pair_epe": epe_scores,
        }


# ---------------------------------------------------------------------------
# Frame reading helper
# ---------------------------------------------------------------------------

def _read_video_frames_rgb(video_path: str, crop_top_ratio: float = 1.0) -> List[np.ndarray]:
    """Read video frames as RGB numpy arrays, optionally cropping bottom."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if crop_top_ratio < 1.0:
            h = frame_rgb.shape[0]
            frame_rgb = frame_rgb[: int(h * crop_top_ratio)]
        frames.append(frame_rgb)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# RewardScorer implementation
# ---------------------------------------------------------------------------

class FlowAEPERewardScorer(RewardScorer):
    """Binary reward based on flow forward-backward consistency (EPE).

    score = 1/avg_epe (modulated by dynamic_degree).
    Reward = 1.0 if score >= epe_threshold, else 0.0.
    """

    def __init__(
        self,
        cfg_path: str = str(_DEFAULT_CFG),
        ckpt_path: str = str(_DEFAULT_CKPT),
        epe_threshold: float = 0.5,
        crop_top_ratio: float = 2 / 3,
        frame_step: int = 1,
        device_id: int = 0,
    ):
        self._crop_top_ratio = crop_top_ratio
        self._epe_threshold = epe_threshold
        self._frame_step = frame_step
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        print(f"[FlowAEPERewardScorer] Loading SEA-RAFT on {device} ...")
        self._metric = _FlowAEPEMetric(cfg_path, ckpt_path, device)
        print(f"[FlowAEPERewardScorer] Ready (threshold={epe_threshold}, "
              f"crop_top_ratio={crop_top_ratio}, frame_step={frame_step})")

    def score(
        self,
        prompt: str,
        first_frame: Image.Image,
        video_path: Optional[str] = None,
        debug_save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        if video_path is None:
            return {"reward": 0.0, "_response_text": "[ERROR] no video_path"}

        try:
            frames = _read_video_frames_rgb(video_path, self._crop_top_ratio)
            if self._frame_step > 1:
                frames = frames[:: self._frame_step]

            result = self._metric.compute_video(frames)
            passed = result["score"] >= self._epe_threshold
            reward = 1.0 if passed else 0.0

            tag = "CLEAN" if passed else "HALL"
            text = (f"[{tag}] score={result['score']:.4f} avg_epe={result['avg_epe']:.4f} "
                    f"dynamic={result['dynamic_degree']:.4f}")

            return {
                "reward": reward,
                "pass": passed,
                "flow_score": result["score"],
                "avg_epe": result["avg_epe"],
                "dynamic_degree": result["dynamic_degree"],
                "_response_text": text,
            }
        except Exception as exc:
            return {"reward": 0.0, "pass": False, "_response_text": f"[ERROR] {exc}"}

    def score_continuous(
        self,
        video_path: str,
        crop_top_ratio: Optional[float] = None,
        frame_step: Optional[int] = None,
    ) -> Dict[str, float]:
        """Return continuous scores (no thresholding). For analysis scripts."""
        crop = crop_top_ratio if crop_top_ratio is not None else self._crop_top_ratio
        step = frame_step if frame_step is not None else self._frame_step

        frames = _read_video_frames_rgb(video_path, crop)
        if step > 1:
            frames = frames[::step]
        return self._metric.compute_video(frames)

    def shutdown(self):
        del self._metric
        torch.cuda.empty_cache()
