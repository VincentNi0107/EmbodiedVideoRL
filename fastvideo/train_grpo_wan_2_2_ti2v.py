#!/usr/bin/env python3
"""Wan2.2 TI2V GRPO Training Pipeline.

Adapted from HunyuanVideo DanceGRPO for the Wan2.2 5B TI2V flow-matching model.

Pipeline per training step:
  1. Sample a prompt + first-frame image from the dataset
  2. Encode text (T5) and image (VAE) once
  3. Generate num_generations videos using SDE sampling (Euler + noise injection)
     → collect intermediate latents and per-step log-probs
  4. Decode videos (VAE) and compute rewards (VideoAlign)
  5. Compute group-wise advantages (per-prompt z-score normalisation)
  6. Optional best-of-N selection (top-half + bottom-half)
  7. Replay denoising steps with gradients → PPO clipped loss → optimizer update

Key difference from HunyuanVideo:
  • Wan2.2 uses explicit CFG (two forward passes), not embedded guidance.
  • Image conditioning via mask-blending (first latent frame = encoded image).
  • During training, only the conditional branch carries gradients.
"""

import argparse
import gc
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fastvideo.utils.logging_ import main_print


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _str2bool(v):
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in ("yes", "true", "t", "y", "1"):
        return True
    if val in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _maybe_init_dist():
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def _get_dit_blocks(dit):
    """Retrieve the WanAttentionBlock list from a possibly wrapped DiT.

    Handles: raw WanModel, PeftModel(WanModel), FSDP(WanModel),
    PeftModel(FSDP(WanModel)), etc.
    """
    # Try the most common paths to reach model.blocks
    candidate = dit
    # Unwrap PeftModel
    if hasattr(candidate, "get_base_model"):
        candidate = candidate.get_base_model()
    # Unwrap FSDP
    if hasattr(candidate, "_fsdp_wrapped_module"):
        candidate = candidate._fsdp_wrapped_module
    # Unwrap a second PeftModel layer (shouldn't happen, but be safe)
    if hasattr(candidate, "get_base_model"):
        candidate = candidate.get_base_model()
    if hasattr(candidate, "blocks"):
        return candidate.blocks
    raise AttributeError(
        f"Cannot find .blocks on dit of type {type(dit)} "
        f"(unwrapped to {type(candidate)})"
    )


def _enable_gradient_checkpointing(dit):
    """Wrap each WanAttentionBlock.forward with gradient checkpointing.

    Trades ~2x forward compute for a massive reduction in activation memory.
    Works with raw DiT, FSDP-wrapped, and/or PeftModel-wrapped DiT.
    """
    import torch.utils.checkpoint as torch_ckpt

    blocks = _get_dit_blocks(dit)
    for block in blocks:
        # Unwrap inner FSDP wrapper on individual blocks if present
        inner = block._fsdp_wrapped_module if hasattr(block, "_fsdp_wrapped_module") else block
        _orig_forward = inner.forward

        def _make_ckpt_fwd(orig_fwd):
            def _ckpt_fwd(*args, **kwargs):
                return torch_ckpt.checkpoint(
                    orig_fwd, *args, use_reentrant=False, **kwargs,
                )
            return _ckpt_fwd

        inner.forward = _make_ckpt_fwd(_orig_forward)
    main_print(f"  Gradient checkpointing enabled on {len(blocks)} DiT blocks")


def _offload_vae_t5(model, to_cpu=True):
    """Move VAE and T5 to CPU to free GPU VRAM during training phase."""
    if to_cpu:
        if hasattr(model, "vae") and model.vae is not None:
            model.vae.model.cpu()
        if hasattr(model, "text_encoder") and model.text_encoder is not None:
            if hasattr(model.text_encoder, "model"):
                model.text_encoder.model.cpu()
        torch.cuda.empty_cache()


def _import_vidar_modules(vidar_root: str = ""):
    # Try importing wan from fastvideo.models.wan first, fall back to vidar_root
    try:
        import fastvideo.models.wan as wan  # type: ignore
        from fastvideo.models.wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS  # type: ignore
        from fastvideo.models.wan.utils.utils import save_video, masks_like, best_output_size  # type: ignore
    except ImportError:
        if not vidar_root:
            raise ImportError(
                "Cannot import 'fastvideo.models.wan' module. Either ensure "
                "fastvideo/models/wan/ exists or pass --vidar_root pointing to "
                "the vidar repository."
            )
        vidar_path = Path(vidar_root).resolve()
        if not vidar_path.exists():
            raise FileNotFoundError(f"vidar_root not found: {vidar_path}")
        if str(vidar_path) not in sys.path:
            sys.path.insert(0, str(vidar_path))
        import wan  # type: ignore
        from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS  # type: ignore
        from wan.utils.utils import save_video, masks_like, best_output_size  # type: ignore

    return (
        wan, WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS,
        save_video, masks_like, best_output_size,
    )


def _dit_dtype(dit) -> torch.dtype:
    """Return dtype of the first DiT parameter."""
    return next(dit.parameters()).dtype


# ──────────────────────────────────────────────────────────────────────────────
# SDE / Flow-matching helpers
# ──────────────────────────────────────────────────────────────────────────────

def sd3_time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    """Apply SD3-style time-shift to a sigma schedule."""
    return (shift * t) / (1.0 + (shift - 1.0) * t)


def get_sigma_schedule(
    num_steps: int, shift: float, device: torch.device,
) -> torch.Tensor:
    """Linear sigma schedule 1 → 0, then time-shifted."""
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    return sd3_time_shift(shift, sigmas)


def flow_sde_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Euler-SDE step for flow-matching diffusion.

    The flow model predicts  v_theta(x_t, t).
      x0_pred  = x_t - sigma_t * v_theta
      ODE mean = x_t + d_sigma * v_theta          (Euler step)
      SDE      = ODE mean + score correction + N(0, std^2)

    Args:
        model_output : predicted flow  v_theta(x_t, t)
        latents      : current noisy latent  x_t
        eta          : SDE noise level (0 → deterministic ODE)
        sigmas       : sigma schedule  (num_steps+1,)
        index        : current step index
        prev_sample  : if given, reuse this as the sampled next latent (replay)
        mask         : binary mask (1 = noisy region, 0 = clean / skip).
                       log-prob is averaged only over mask==1 elements.

    Returns:
        (next_sample, x0_pred, log_prob)
    """
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma            # negative (sigma decreases)
    x0_pred = latents - sigma * model_output
    prev_sample_mean = latents + dsigma * model_output

    delta_t = (sigma - sigmas[index + 1]).clamp(min=1e-10)
    std_dev_t = eta * delta_t.sqrt()

    # Langevin-style score correction so that the SDE marginals match the ODE
    if eta > 0:
        score = -(latents - x0_pred * (1.0 - sigma)) / (sigma ** 2 + 1e-10)
        prev_sample_mean = prev_sample_mean + (-0.5 * eta ** 2 * score) * dsigma

    # Sample or replay
    if prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    # Gaussian log-prob  N(prev_sample ; mean, std^2)
    diff = prev_sample.detach().float() - prev_sample_mean.float()
    log_prob_elem = (
        -(diff ** 2) / (2.0 * std_dev_t ** 2 + 1e-20)
        - std_dev_t.float().log().clamp(min=-20)
        - 0.5 * math.log(2.0 * math.pi)
    )

    if mask is not None:
        m = mask.float()
        log_prob = (log_prob_elem * m).sum() / m.sum().clamp(min=1.0)
    else:
        log_prob = log_prob_elem.mean()

    return prev_sample, x0_pred, log_prob


def _expand_timestep(
    mask2: torch.Tensor,
    sigma_val: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand a scalar sigma to spatially-varying timesteps using mask2.

    First-frame patches get t=0 (clean), the rest get sigma*1000.
    """
    t_val = sigma_val * 1000.0
    # mask2: (C, F, H, W) — channel 0, downsample spatial by 2 for patches
    ts = (mask2[0][:, ::2, ::2] * t_val).flatten()
    ts = torch.cat([ts, ts.new_ones(seq_len - ts.size(0)) * t_val])
    return ts.unsqueeze(0).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class GRPOPromptDataset(Dataset):
    """Returns one (prompt, image_path) per item — no per-generation expansion."""

    def __init__(self, dataset_json: str, max_samples: int = -1):
        p = Path(dataset_json).resolve()
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if 0 < max_samples < len(raw):
            raw = raw[:max_samples]
        self.items: List[Dict] = []
        for i, row in enumerate(raw):
            mp = Path(row.get("media_path", ""))
            if not mp.is_file():
                mp = (p.parent / row.get("media_path", "")).resolve()
            self.items.append({
                "prompt": row["prompt"],
                "media_path": str(mp),
                "filename_stem": row.get("filename_stem", f"sample_{i:06d}"),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Reward scorers  (kept from the original file)
# ──────────────────────────────────────────────────────────────────────────────

class RewardScorer:
    def score(
        self, prompt: str, first_frame: Image.Image,
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError


class NoRewardScorer(RewardScorer):
    def score(self, prompt, first_frame, video_path=None):
        return {"reward": 0.0}


class GPTRewardScorer(RewardScorer):
    """Binary (0/1) reward from a vision-language model (Gemini / GPT).

    Sends a 2x2 grid of 4 sampled video frames to the API and asks for a
    structured pass/fail judgement.  Any single failure criterion → score 0.
    """

    def __init__(
        self,
        api_base: str = "http://35.220.164.252:3888/v1/",
        api_key: str = None,
        model: str = "gemini-3-flash-preview",
        temperature: float = 0.0,
        max_retries: int = 2,
    ):
        import os
        import cv2 as _cv2  # noqa – lazy import
        self._cv2 = _cv2
        from openai import OpenAI
        if api_key is None:
            api_key = os.environ.get("GPT_API_KEY", "")
        self._client = OpenAI(base_url=api_base, api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries

    # ── video → base64 grid ───────────────────────────────────────
    def _video_to_grid_base64(self, video_path: str):
        """Returns (base64_str, grid_numpy_bgr)."""
        import base64, io
        import numpy as np
        cv2 = self._cv2

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick frames at 0, 1/3, 2/3, and the last frame
        pick_indices = sorted(set([
            0,
            int(total * 1 / 3),
            int(total * 2 / 3),
            total - 1,
        ]))
        pick_indices = [max(0, min(i, total - 1)) for i in pick_indices]
        pick_set = set(pick_indices)

        frames_map = {}
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx in pick_set:
                # Crop to main view (head camera) only — top 2/3
                h = frame.shape[0]
                main_h = h * 2 // 3
                frame = frame[:main_h, :, :]
                # Crop left/right 1/10 each
                w = frame.shape[1]
                margin_lr = w // 10
                frame = frame[:, margin_lr:w - margin_lr, :]
                frames_map[idx] = frame
            idx += 1
        cap.release()

        grid_frames = [frames_map[i] for i in pick_indices]

        # 2×2 grid
        rows, cols = 2, 2
        row_imgs = []
        for r in range(rows):
            row_imgs.append(np.concatenate(grid_frames[r * cols:(r + 1) * cols], axis=1))
        grid = np.concatenate(row_imgs, axis=0)

        _, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("utf-8"), grid

    # ── prompt ────────────────────────────────────────────────────
    @staticmethod
    def _build_prompt(task_description: str) -> str:
        return f"""You are evaluating an AI-generated robot manipulation video.

The image shows a 2×2 grid of 4 frames sampled from the video in chronological order (read row by row, left-to-right then top-to-bottom).
Each frame is from a fixed rear camera showing the full workspace of a dual-arm "aloha" robot.

**Task description:** {task_description}

This task requires **two arms to collaborate**:
- The **right arm** (on the right bottom side of each image) should **open the drawer** by reaching for and pulling the handle.
- The **left arm** (on the left bottom side of each image) should **pick up the object from the table and place it inside the opened drawer**.

Evaluate the video for the following **failure criteria**. If ANY of them is true, the task FAILS (score = 0). Only if NONE of them is true, the task PASSES (score = 1).

### Failure Criteria
1. **Right arm frozen, drawer opens by itself**: The right arm does not move towards or contact the drawer, yet the drawer opens on its own.
2. **Right arm frozen, left arm does everything**: The right arm stays still while the left arm picks up the object AND also attempts to open the drawer by itself.

Respond with ONLY a JSON object (no markdown, no extra text):
{{"pass": true/false, "reason": "one-sentence explanation", "failures": ["list of triggered failure criterion numbers, e.g. 1,3"]}}
"""

    # ── save debug image: grid + GPT response text below ─────────
    @staticmethod
    def _save_debug_image(grid_bgr, response_text: str, save_path: str):
        """Save the frame grid with GPT response rendered below it."""
        import numpy as np
        cv2_mod = __import__("cv2")

        grid_h, grid_w = grid_bgr.shape[:2]

        # ── render text into a strip below the grid ──────────
        font = cv2_mod.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        color = (255, 255, 255)   # white text
        line_height = 36
        margin = 16

        # Word-wrap the response to fit the grid width using actual text size
        usable_w = grid_w - margin * 2
        words = response_text.split()
        lines: list[str] = []
        cur = ""
        for w in words:
            candidate = f"{cur} {w}" if cur else w
            (tw, _), _ = cv2_mod.getTextSize(candidate, font, font_scale, thickness)
            if tw > usable_w and cur:
                lines.append(cur)
                cur = w
            else:
                cur = candidate
        if cur:
            lines.append(cur)

        text_h = margin * 2 + line_height * len(lines)
        text_strip = np.zeros((text_h, grid_w, 3), dtype=np.uint8)
        for i, line in enumerate(lines):
            y = margin + line_height * (i + 1)
            cv2_mod.putText(text_strip, line, (margin, y),
                            font, font_scale, color, thickness, cv2_mod.LINE_AA)

        combined = np.concatenate([grid_bgr, text_strip], axis=0)
        cv2_mod.imwrite(save_path, combined)

    # ── API call ──────────────────────────────────────────────────
    def score(self, prompt: str, first_frame: Image.Image,
              video_path: Optional[str] = None,
              debug_save_path: Optional[str] = None) -> Dict[str, float]:
        if video_path is None:
            raise ValueError("video_path is required for GPTRewardScorer")

        grid_b64, grid_bgr = self._video_to_grid_base64(str(Path(video_path).resolve()))
        gpt_prompt = self._build_prompt(prompt)

        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": gpt_prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{grid_b64}"}},
                        ],
                    }],
                    temperature=self._temperature,
                )
                raw = resp.choices[0].message.content.strip()
                # Strip markdown fences if present
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                result = json.loads(raw)
                passed = bool(result.get("pass", False))
                reason = result.get("reason", "")
                failures = result.get("failures", [])
                reward = 1.0 if passed else 0.0

                label = "PASS" if passed else "FAIL"
                response_text = f"[{label}] {reason}  failures={failures}"

                # Save debug visualisation: grid + GPT response
                if debug_save_path:
                    try:
                        self._save_debug_image(grid_bgr, response_text, debug_save_path)
                    except Exception as e:
                        main_print(f"  [GPT reward] debug image save failed: {e}")

                return {
                    "reward": reward,
                    "pass": passed,
                    "reason": reason,
                    "failures": failures,
                    "_grid_bgr": grid_bgr,
                    "_response_text": response_text,
                }
            except Exception as exc:
                if attempt < self._max_retries:
                    main_print(f"  [GPT reward] attempt {attempt+1} failed: {exc}, retrying...")
                    continue
                main_print(f"  [GPT reward] all attempts failed: {exc}")
                err_text = f"[API ERROR] {exc}"
                # Still save the grid even on API failure
                if debug_save_path:
                    try:
                        self._save_debug_image(grid_bgr, err_text, debug_save_path)
                    except Exception:
                        pass
                return {
                    "reward": 0.0, "pass": False,
                    "reason": f"API error: {exc}", "failures": [],
                    "_grid_bgr": grid_bgr, "_response_text": err_text,
                }


class VideoAlignScorer(RewardScorer):
    _KEY_MAP = {"vq": "VQ", "mq": "MQ", "ta": "TA", "overall": "Overall"}

    def __init__(self, device, ckpt_dir, score_key="overall", use_norm=True):
        from fastvideo.models.videoalign.inference import VideoVLMRewardInference

        self._score_key = score_key.lower()
        if self._score_key not in self._KEY_MAP:
            raise ValueError(f"Unsupported score_key: {score_key}")
        self._inferencer = VideoVLMRewardInference(
            ckpt_dir, device=f"cuda:{device.index}", dtype=torch.bfloat16,
        )
        self._use_norm = use_norm

    @torch.no_grad()
    def score(self, prompt, first_frame, video_path=None):
        if video_path is None:
            raise ValueError("video_path required for VideoAlign")
        rw = self._inferencer.reward(
            [str(Path(video_path).resolve())], [prompt],
            use_norm=self._use_norm,
        )
        r = rw[0]
        sel_key = self._KEY_MAP[self._score_key]
        return {
            "reward": float(r[sel_key]),
            "VQ": float(r["VQ"]), "MQ": float(r["MQ"]),
            "TA": float(r["TA"]), "Overall": float(r["Overall"]),
        }


def _build_reward_scorer(args, device):
    if args.reward_backend == "none":
        return NoRewardScorer()
    if args.reward_backend == "gpt":
        return GPTRewardScorer(
            api_base=args.gpt_api_base,
            api_key=args.gpt_api_key,
            model=args.gpt_model,
            temperature=args.gpt_temperature,
        )
    if args.reward_backend == "videoalign":
        return VideoAlignScorer(
            device=device, ckpt_dir=args.videoalign_ckpt_dir,
            score_key=args.videoalign_score_key,
            use_norm=args.videoalign_use_norm,
        )
    raise ValueError(f"Unsupported reward_backend: {args.reward_backend}")


def _video_first_frame_pil(video_tensor: torch.Tensor) -> Image.Image:
    frame = video_tensor[:, 0].detach().float().clamp(-1, 1)
    frame = ((frame + 1.0) * 127.5).to(torch.uint8).cpu()
    return Image.fromarray(frame.permute(1, 2, 0).numpy())


def _save_reward_curve(
    steps: List[int],
    mean_rewards: List[float],
    save_path: str,
):
    """Save / overwrite a reward-vs-step line chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, mean_rewards, marker="o", markersize=4, linewidth=1.5,
            color="#2196F3", label="mean reward")
    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Mean Reward", fontsize=13)
    ax.set_title("GRPO Training — Mean Reward per Step", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    # y-axis always includes [0, 1] for binary rewards, but expand if needed
    y_lo = min(0.0, min(mean_rewards) - 0.05)
    y_hi = max(1.0, max(mean_rewards) + 0.05)
    ax.set_ylim(y_lo, y_hi)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# LoRA helpers
# ──────────────────────────────────────────────────────────────────────────────

# Default target modules in WanModel DiT — attention projections + FFN
WAN_LORA_TARGET_MODULES = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
]


def _apply_lora(dit, lora_rank: int, lora_alpha: int,
                target_modules: Optional[List[str]] = None,
                resume_path: Optional[str] = None):
    """Inject LoRA adapters into the DiT and optionally resume from checkpoint.

    Uses HuggingFace ``peft`` library.  Only LoRA parameters are trainable;
    all base weights are frozen.
    """
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = WAN_LORA_TARGET_MODULES

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True,
        target_modules=target_modules,
    )

    # Freeze base weights first
    dit.requires_grad_(False)

    # peft wraps the model — for nn.Module that is NOT a PreTrainedModel we
    # use add_adapter (which modifies in-place) rather than get_peft_model.
    if hasattr(dit, "add_adapter"):
        dit.add_adapter(lora_config)
    else:
        dit = get_peft_model(dit, lora_config)

    # Resume LoRA weights from a previous checkpoint.
    # Our checkpoint saves raw named_parameters keys (not peft format)
    # so we load them back the same way.
    if resume_path:
        main_print(f"  Loading LoRA checkpoint from {resume_path}")
        saved = torch.load(resume_path, map_location="cpu")
        model_dict = dict(dit.named_parameters())
        loaded = 0
        for k, v in saved.items():
            if k in model_dict:
                model_dict[k].data.copy_(v)
                loaded += 1
            else:
                main_print(f"  [LoRA resume] skipping unknown key: {k}")
        main_print(f"  [LoRA resume] loaded {loaded}/{len(saved)} tensors")

    # Only LoRA params are trainable
    lora_params = [p for p in dit.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in dit.parameters())
    main_print(
        f"  LoRA injected: rank={lora_rank}, alpha={lora_alpha}, "
        f"target_modules={target_modules}"
    )
    main_print(
        f"  Trainable: {n_lora / 1e6:.1f} M / {n_total / 1e6:.1f} M total "
        f"({100 * n_lora / n_total:.2f}%)"
    )
    return dit


def _save_lora_checkpoint(dit, save_path: str):
    """Save only the LoRA adapter weights.

    Cannot use ``get_peft_model_state_dict`` when the base model is wrapped
    by FSDP, because peft renames Linear → base_layer which breaks FSDP's
    state_dict hook.  Instead we manually collect LoRA parameters by name.
    """
    lora_state = {}
    for name, param in dit.named_parameters():
        if "lora_" in name:
            lora_state[name] = param.detach().cpu()
    torch.save(lora_state, save_path)
    main_print(f"    Saved {len(lora_state)} LoRA tensors ({sum(p.numel() for p in lora_state.values()) / 1e6:.1f} M params)")


def _extract_resume_step_from_path(resume_path: Optional[str]) -> int:
    """Parse resume step/epoch from checkpoint filename.

    Supports names like:
      lora_step000040.pt
      lora_epoch40.pt
    """
    if not resume_path:
        return 0
    name = Path(resume_path).name
    m = re.search(r"(?:step|epoch)(\d+)", name)
    if m is None:
        return 0
    return int(m.group(1))


# ──────────────────────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_model(args, rank, local_rank, world_size, wan, cfg):
    """Build WanTI2V model.

    When ``world_size > 1``, the DiT is wrapped with FSDP (FULL_SHARD).
    Only rank 0 loads ``pt_dir`` fine-tuned weights; FSDP broadcasts them
    via ``sync_module_states=True``.

    NOTE: _configure_model inside WanTI2V sets model.eval().requires_grad_(False).
    The training loop re-enables grad on the DiT when needed.
    """
    use_fsdp = world_size > 1
    model = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        # Only rank 0 loads fine-tuned DiT weights; FSDP syncs to others.
        pt_dir=args.pt_dir if (rank == 0 or not use_fsdp) else None,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=use_fsdp,
        use_sp=False,
        t5_cpu=True,                # T5 on CPU to save GPU VRAM
        init_on_cpu=not use_fsdp,   # FSDP handles device placement
        convert_model_dtype=args.convert_model_dtype and not use_fsdp,
    )
    return model


def _prepare_image(model, img_pil, max_area, best_output_size_fn):
    """Crop / resize an image the same way as WanTI2V.generate()."""
    ih, iw = img_pil.height, img_pil.width
    dh = model.patch_size[1] * model.vae_stride[1]
    dw = model.patch_size[2] * model.vae_stride[2]
    ow, oh = best_output_size_fn(iw, ih, dw, dh, max_area)
    scale = max(ow / iw, oh / ih)
    img_pil = img_pil.resize(
        (round(iw * scale), round(ih * scale)), Image.LANCZOS,
    )
    x1, y1 = (img_pil.width - ow) // 2, (img_pil.height - oh) // 2
    img_pil = img_pil.crop((x1, y1, x1 + ow, y1 + oh))
    img_tensor = TF.to_tensor(img_pil).sub_(0.5).div_(0.5)
    img_tensor = img_tensor.to(model.device).unsqueeze(1)  # (3, 1, H, W)
    return img_tensor, oh, ow


def _compute_seq_len(frame_num, oh, ow, vae_stride, patch_size):
    lat_f = (frame_num - 1) // vae_stride[0] + 1
    lat_h = oh // vae_stride[1]
    lat_w = ow // vae_stride[2]
    return lat_f * lat_h * lat_w // (patch_size[1] * patch_size[2])


# ──────────────────────────────────────────────────────────────────────────────
# GRPO: SDE rollout  (torch.no_grad — eval mode)
# ──────────────────────────────────────────────────────────────────────────────

def sde_rollout_single(
    dit, noise, z_img, mask2,
    ctx_cond, ctx_null, seq_len,
    sigmas, eta, guide_scale,
    device, num_train_timesteps,
):
    """Run the full SDE denoising loop for **one** video.

    Returns
    -------
    all_latents : list[Tensor]   length = S+1, each (C, F, H, W) on CPU
    all_log_probs : Tensor       shape (S,) on CPU
    x0_pred : Tensor             last step's x0 prediction on GPU
    """
    S = len(sigmas) - 1
    dtype = _dit_dtype(dit)

    # Initial blend: image at frame-0, noise elsewhere
    latent = ((1.0 - mask2) * z_img + mask2 * noise).to(device).float()

    all_latents: List[torch.Tensor] = [latent.detach().cpu()]
    all_log_probs: List[torch.Tensor] = []
    x0_pred = latent  # placeholder

    for i in range(S):
        ts = _expand_timestep(mask2, sigmas[i], seq_len, device)
        lat_in = latent.to(dtype)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v_c = dit([lat_in], t=ts, context=[ctx_cond[0]], seq_len=seq_len)[0]
            v_u = dit([lat_in], t=ts, context=ctx_null, seq_len=seq_len)[0]

        v = (v_u + guide_scale * (v_c - v_u)).float()
        next_lat, x0_pred, lp = flow_sde_step(
            v, latent, eta, sigmas, i, prev_sample=None, mask=mask2,
        )
        # Re-apply mask: keep image frame intact
        next_lat = (1.0 - mask2) * z_img.float() + mask2 * next_lat

        all_latents.append(next_lat.detach().cpu())
        all_log_probs.append(lp.detach().cpu())
        latent = next_lat.detach()

        del v_c, v_u, v, lat_in

    return all_latents, torch.stack(all_log_probs), x0_pred


# ──────────────────────────────────────────────────────────────────────────────
# GRPO: replay one step with gradients
# ──────────────────────────────────────────────────────────────────────────────

def grpo_replay_step(
    dit, latent_t, next_latent, mask2,
    ctx_cond, ctx_null, seq_len,
    step_idx, sigmas, eta, guide_scale,
    device, num_train_timesteps,
):
    """Replay one denoising step with grad through both CFG branches.

    Matching the original DanceGRPO: gradient flows through both the
    conditional and unconditional forward passes so that the CFG gap
    (v_c - v_u) is actively maintained during training.

    Returns
    -------
    log_prob : scalar Tensor (with grad)
    """
    dtype = _dit_dtype(dit)
    ts = _expand_timestep(mask2, sigmas[step_idx], seq_len, device)
    lat = latent_t.to(device, dtype=dtype)

    # Both branches WITH gradient (matches original DanceGRPO)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        v_c = dit([lat], t=ts, context=[ctx_cond[0]], seq_len=seq_len)[0]
        v_u = dit([lat], t=ts, context=ctx_null, seq_len=seq_len)[0]

    # CFG — grad flows through both v_c and v_u
    v = (v_u + guide_scale * (v_c - v_u)).float()

    _, _, lp = flow_sde_step(
        v, latent_t.to(device).float(), eta, sigmas, step_idx,
        prev_sample=next_latent.to(device).float(), mask=mask2,
    )
    return lp


# ──────────────────────────────────────────────────────────────────────────────
# Advantage computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_advantages(rewards: torch.Tensor, num_gen: int) -> torch.Tensor:
    """Per-group (per-prompt) mean-baseline advantage.

    For binary (0/1) rewards, z-score normalisation creates extreme
    asymmetric advantages (e.g. -2.65 vs +0.38) that destabilise
    training.  Using a simple mean baseline keeps advantages bounded
    in [-1, +1], giving balanced positive/negative signals.
    """
    adv = torch.zeros_like(rewards)
    n = len(rewards) // num_gen
    for i in range(n):
        s, e = i * num_gen, (i + 1) * num_gen
        g = rewards[s:e]
        adv[s:e] = g - g.mean()
    return adv


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Wan2.2 TI2V GRPO Training")

    # model / vidar
    p.add_argument("--vidar_root", type=str, default="",
                    help="Path to vidar repo (optional if wan/ exists in project root)")
    p.add_argument("--task", type=str, default="ti2v-5B")
    p.add_argument("--size", type=str, default="640*736")
    p.add_argument("--frame_num", type=int, default=61)
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--pt_dir", type=str, default=None)
    p.add_argument("--convert_model_dtype", action="store_true", default=False)
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--offload_model", type=_str2bool, default=False)

    # data
    p.add_argument("--dataset_json", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--output_dir", type=str, required=True)

    # sampling / SDE
    p.add_argument("--sample_steps", type=int, default=20)
    p.add_argument("--sample_shift", type=float, default=5.0)
    p.add_argument("--sample_guide_scale", type=float, default=5.0)
    p.add_argument("--eta", type=float, default=1.0,
                    help="SDE noise level. 0 = deterministic ODE, >0 = SDE.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neg_prompt", type=str, default="")

    # reward
    p.add_argument("--reward_backend", type=str, default="videoalign",
                    choices=["videoalign", "gpt", "none"])
    p.add_argument("--gpt_api_base", type=str,
                    default="http://35.220.164.252:3888/v1/")
    p.add_argument("--gpt_api_key", type=str,
                    default=None,
                    help="API key for GPT/Gemini reward. Defaults to $GPT_API_KEY env var.")
    p.add_argument("--gpt_model", type=str, default="gpt-4o")
    p.add_argument("--gpt_temperature", type=float, default=0.2)
    p.add_argument("--videoalign_ckpt_dir", type=str, default="./videoalign_ckpt")
    p.add_argument("--videoalign_score_key", type=str, default="overall",
                    choices=["vq", "mq", "ta", "overall"])
    p.add_argument("--videoalign_use_norm", type=_str2bool, default=True)

    # GRPO hyper-parameters
    p.add_argument("--num_generations", type=int, default=4,
                    help="N videos per prompt per training step")
    p.add_argument("--bestofn", type=int, default=4,
                    help="Select top+bottom bestofn/2 for training")
    p.add_argument("--clip_range", type=float, default=1e-4,
                    help="PPO ratio clip epsilon")
    p.add_argument("--adv_clip_max", type=float, default=5.0,
                    help="Advantage clipping")
    p.add_argument("--timestep_fraction", type=float, default=1.0,
                    help="Fraction of denoising steps to train on")

    # training
    p.add_argument("--max_train_steps", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=2.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4,
                    help="Accumulate grads over this many prompts before optimizer.step()")
    p.add_argument("--checkpointing_steps", type=int, default=10)
    p.add_argument("--log_every", type=int, default=1)

    # memory optimisations (only used in single-GPU mode; ignored with FSDP)
    p.add_argument("--gradient_checkpointing", type=_str2bool, default=True,
                    help="Enable gradient checkpointing on DiT (single-GPU only)")
    p.add_argument("--use_8bit_adam", type=_str2bool, default=True,
                    help="Use bitsandbytes 8-bit AdamW (single-GPU only)")

    # LoRA
    p.add_argument("--use_lora", type=_str2bool, default=False,
                    help="Enable LoRA fine-tuning on the DiT")
    p.add_argument("--lora_rank", type=int, default=64,
                    help="LoRA rank (r)")
    p.add_argument("--lora_alpha", type=int, default=64,
                    help="LoRA alpha scaling factor")
    p.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                    help="Which DiT sub-modules to inject LoRA into (default: attention Q/K/V/O)")
    p.add_argument("--resume_from_lora_checkpoint", type=str, default=None,
                    help="Path to a saved LoRA checkpoint to resume from")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rank, world_size, local_rank = _maybe_init_dist()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    # ── vidar imports ───────────────────────────────────────────────
    (wan, WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS,
     save_video, masks_like, best_output_size) = _import_vidar_modules(args.vidar_root)

    cfg = WAN_CONFIGS[args.task]
    if not args.neg_prompt:
        args.neg_prompt = cfg.sample_neg_prompt
    max_area = MAX_AREA_CONFIGS[args.size]

    # ── directories ────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    vid_dir = out_dir / "videos"
    reward_dir = out_dir / "reward_debug"
    ckpt_dir = out_dir / "checkpoints"
    for d in (out_dir, vid_dir, reward_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── model ──────────────────────────────────────────────────────
    use_fsdp = world_size > 1
    main_print(f"Building Wan2.2 TI2V model ... (FSDP={use_fsdp}, world_size={world_size})")
    model = _build_model(args, rank, local_rank, world_size, wan, cfg)
    dit = model.model  # WanModel (DiT) — the trainable part

    # ── reward scorer ──────────────────────────────────────────────
    main_print(f"Building reward scorer ({args.reward_backend}) ...")
    scorer = _build_reward_scorer(args, device)

    # ── sigma schedule ─────────────────────────────────────────────
    sigmas = get_sigma_schedule(args.sample_steps, args.sample_shift, device)

    # ── dataset ────────────────────────────────────────────────────
    ds = GRPOPromptDataset(args.dataset_json, args.max_samples)
    main_print(f"Dataset: {len(ds)} prompts")

    # ── LoRA (optional) ────────────────────────────────────────────
    if args.use_lora:
        main_print("Applying LoRA to DiT ...")
        dit = _apply_lora(
            dit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            resume_path=args.resume_from_lora_checkpoint,
        )
        model.model = dit   # update reference so rollout uses LoRA-wrapped model

    # ── gradient checkpointing ────────────────────────────────────
    # Always enable when requested — critical for fitting activations
    # in memory during training, with or without FSDP / LoRA.
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(dit)

    # ── optimizer ──────────────────────────────────────────────────
    if args.use_lora:
        # Only LoRA params are trainable (already set by _apply_lora)
        pass
    else:
        dit.requires_grad_(True)
    trainable_params = [p for p in dit.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    main_print(f"Trainable parameters: {n_params / 1e6:.1f} M")

    if args.use_8bit_adam and not use_fsdp:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            trainable_params, lr=args.learning_rate,
            betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8,
        )
        main_print("  Using 8-bit AdamW (bitsandbytes)")
    else:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.learning_rate,
            betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8,
        )
        if use_fsdp:
            main_print("  Using fp32 AdamW (FSDP shards optimizer states)")
    if not args.use_lora:
        dit.requires_grad_(False)
    dit.eval()

    # ── distributed rollout sizing ─────────────────────────────────
    N = args.num_generations
    if use_fsdp:
        assert N % world_size == 0, (
            f"num_generations ({N}) must be divisible by world_size ({world_size})")
        N_local = N // world_size
    else:
        N_local = N

    # ── logging ────────────────────────────────────────────────────
    log_path = out_dir / "training_log.jsonl"
    resume_step = _extract_resume_step_from_path(args.resume_from_lora_checkpoint)
    start_step = resume_step
    start_train_step = start_step + 1
    end_train_step = args.max_train_steps
    num_steps_this_run = max(0, end_train_step - start_step)

    if args.resume_from_lora_checkpoint:
        if resume_step > 0:
            main_print(
                f"Resume detected from checkpoint name: step={resume_step} "
                f"(path={args.resume_from_lora_checkpoint})"
            )
        else:
            main_print(
                f"Resume checkpoint provided but no step/epoch found in filename: "
                f"{args.resume_from_lora_checkpoint} (fallback step=0)"
            )

    if num_steps_this_run <= 0:
        main_print(
            f"No steps to run: resume_step={resume_step}, "
            f"max_train_steps={args.max_train_steps}"
        )
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    main_print(
        f"Starting GRPO | global_step_range=[{start_train_step}, {end_train_step}]  "
        f"new_steps={num_steps_this_run}  N={N}  "
        f"N_local/rank={N_local}  bestofn={args.bestofn}  eta={args.eta}  "
        f"lr={args.learning_rate}  sample_steps={args.sample_steps}  "
        f"timestep_frac={args.timestep_fraction}"
    )

    # ── training loop ──────────────────────────────────────────────
    # Use a seeded sampler so all ranks iterate the dataset in the same order.
    # This is critical: every rank must process the SAME prompt at each step
    # so that all_gather(rewards) computes a meaningful global advantage.
    dl_generator = torch.Generator().manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0,
                    generator=dl_generator)
    data_iter = iter(dl)
    accum_loss = 0.0       # accumulated loss across gradient_accumulation_steps
    accum_n_fwd = 0        # accumulated forward count
    reward_history_steps: List[int] = []
    reward_history_means: List[float] = []

    for step in range(start_train_step, end_train_step + 1):
        # ─── get next prompt (same on all ranks) ─────────
        try:
            batch = next(data_iter)
        except StopIteration:
            dl_generator = torch.Generator().manual_seed(args.seed + step)
            dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0,
                            generator=dl_generator)
            data_iter = iter(dl)
            batch = next(data_iter)

        prompt = batch["prompt"][0]
        media_path = batch["media_path"][0]
        fname_stem = batch["filename_stem"][0]
        main_print(f"\n{'=' * 60}\nStep {step}/{args.max_train_steps}  |  {fname_stem}")

        # ─── encode text ────────────────────────────────────
        if model.t5_cpu:
            # Move T5 to GPU for faster encoding, then offload back
            model.text_encoder.model.to(device)
        ctx_c = model.text_encoder([prompt], device)
        ctx_n = model.text_encoder([args.neg_prompt], device)
        if model.t5_cpu:
            model.text_encoder.model.cpu()
            torch.cuda.empty_cache()

        # ─── encode image (VAE) ───────────────────────────
        model.vae.model.to(device)
        img_pil = Image.open(media_path).convert("RGB")
        img_tensor, oh, ow = _prepare_image(
            model, img_pil, max_area, best_output_size,
        )
        z_img = model.vae.encode([img_tensor])[0]  # (C, 1, H', W')

        F = args.frame_num
        seq_len = _compute_seq_len(
            F, oh, ow, model.vae_stride, model.patch_size,
        )
        latent_shape = (
            model.vae.model.z_dim,
            (F - 1) // model.vae_stride[0] + 1,
            oh // model.vae_stride[1],
            ow // model.vae_stride[2],
        )

        # ─── mask (first frame = image, rest = noise) ─────
        noise_tmp = torch.randn(latent_shape, device=device, dtype=torch.float32)
        _, mask2_list = masks_like([noise_tmp], zero=True)
        mask2 = mask2_list[0].to(device)
        del noise_tmp

        # Move DiT to GPU if single-GPU & it lives on CPU
        if not use_fsdp and (args.offload_model or model.init_on_cpu):
            dit.to(device)
            torch.cuda.empty_cache()

        # ════════════════════════════════════════════════════
        #  PHASE 1a: ROLLOUT — generate all videos (GPU-bound)
        #
        #  Multi-GPU: each rank generates N_local = N/world_size
        #  videos with different seeds.  Reward scoring is deferred
        #  to Phase 1b so that the GPU stays busy generating.
        # ════════════════════════════════════════════════════
        dit.eval()
        local_lats: List[List[torch.Tensor]] = []
        local_lps: List[torch.Tensor] = []

        scene_suffix = fname_stem[-4:] if len(fname_stem) >= 4 else fname_stem
        step_vid_dir = vid_dir / f"step{step:04d}_{scene_suffix}"
        step_reward_dir = reward_dir / f"step{step:04d}_{scene_suffix}"
        step_vid_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_reward_debug_video:
            step_reward_dir.mkdir(parents=True, exist_ok=True)

        # Collect (video_path, debug_base_name, first_frame_pil) for deferred scoring
        pending_scores: List[tuple] = []  # (video_path, debug_base, first_frame_pil | None)

        base_seed = args.seed + step * N
        for g_local in range(N_local):
            g = rank * N_local + g_local if use_fsdp else g_local
            seed_g = base_seed + g
            gen = torch.Generator(device=device).manual_seed(seed_g)
            noise = torch.randn(
                latent_shape, dtype=torch.float32,
                generator=gen, device=device,
            )

            lats, lps, x0_pred = sde_rollout_single(
                dit, noise, z_img, mask2,
                ctx_c, ctx_n, seq_len,
                sigmas, args.eta, args.sample_guide_scale,
                device, model.num_train_timesteps,
            )
            local_lats.append(lats)
            local_lps.append(lps)

            # Decode video using x0_pred (clean model prediction)
            final_lat = x0_pred.to(device, dtype=model.param_dtype)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vids = model.vae.decode([final_lat])

            if vids and vids[0] is not None:
                vid = vids[0]
                vp = str(step_vid_dir / f"{fname_stem}_g{g:03d}_s{seed_g}.mp4")
                save_video(
                    tensor=vid[None], save_file=vp,
                    fps=cfg.sample_fps, nrow=1,
                    normalize=True, value_range=(-1, 1),
                )
                # debug_base: PASS/FAIL suffix added after scoring
                dbg_base = f"{fname_stem}_g{g:03d}_s{seed_g}"
                pending_scores.append((vp, dbg_base, _video_first_frame_pil(vid)))
            else:
                pending_scores.append((None, None, None))

            del noise, final_lat
            torch.cuda.empty_cache()

        # ════════════════════════════════════════════════════
        #  PHASE 1b: REWARD SCORING (CPU / network-bound)
        #
        #  All GPU work is done.  Score videos sequentially
        #  via the GPT API (or other reward backend).
        # ════════════════════════════════════════════════════
        local_rws: List[float] = []
        for vp, dbg_base, first_frame in pending_scores:
            if vp is not None:
                try:
                    rd = scorer.score(
                        prompt=prompt,
                        first_frame=first_frame,
                        video_path=vp,
                        debug_save_path=None,  # save manually below with PASS/FAIL tag
                    )
                except Exception as exc:
                    rd = {"reward": -10.0, "error": str(exc)}
                rw_val = float(rd["reward"])
                # Save debug image with PASS/FAIL in filename
                if not args.skip_reward_debug_video:
                    tag = "PASS" if rw_val > 0 else "FAIL"
                    dbg_path = str(step_reward_dir / f"{dbg_base}_{tag}.jpg")
                    grid_bgr = rd.get("_grid_bgr")
                    resp_text = rd.get("_response_text")
                    if grid_bgr is not None and resp_text is not None:
                        try:
                            GPTRewardScorer._save_debug_image(grid_bgr, resp_text, dbg_path)
                        except Exception as e:
                            main_print(f"  [debug img] save failed: {e}")
            else:
                rd = {"reward": -10.0, "error": "video_none"}
                rw_val = float(rd["reward"])
            local_rws.append(rw_val)

        # ─── gather rewards across ranks ──────────────────
        if use_fsdp:
            local_rw_t = torch.tensor(local_rws, dtype=torch.float32, device=device)
            gathered = [torch.zeros_like(local_rw_t) for _ in range(world_size)]
            dist.all_gather(gathered, local_rw_t)
            all_rws_global = torch.cat(gathered).tolist()
        else:
            all_rws_global = local_rws

        rw_tensor = torch.tensor(all_rws_global, dtype=torch.float32, device=device)
        advs = compute_advantages(rw_tensor, N)

        local_advs = advs[rank * N_local:(rank + 1) * N_local] if use_fsdp else advs

        step_mean_rw = rw_tensor.mean().item()
        main_print(f"  rewards : {[f'{r:.3f}' for r in all_rws_global]}")
        main_print(f"  mean_rw : {step_mean_rw:.4f}")

        # ─── best-of-N selection (per rank) ────────────────
        # Select top bestofn/2 + bottom bestofn/2 from local samples,
        # matching the original DanceGRPO HunyuanVideo implementation.
        if args.bestofn < N_local:
            sorted_idx = torch.argsort(local_advs)
            half = args.bestofn // 2
            top_idx = sorted_idx[-half:]
            bottom_idx = sorted_idx[:half]
            selected_idx = torch.cat([top_idx, bottom_idx])
            # Shuffle so training order is random
            selected_idx = selected_idx[torch.randperm(len(selected_idx))]

            local_lats = [local_lats[i] for i in selected_idx.tolist()]
            local_lps = [local_lps[i] for i in selected_idx.tolist()]
            local_advs = local_advs[selected_idx]
            N_local_train = len(selected_idx)
            main_print(f"  bestofn : selected {N_local_train}/{N_local} samples (top {half} + bottom {half})")
        else:
            N_local_train = N_local

        # ─── Update reward curve plot ────────────────────
        reward_history_steps.append(step)
        reward_history_means.append(step_mean_rw)
        if rank == 0:
            try:
                _save_reward_curve(
                    reward_history_steps,
                    reward_history_means,
                    str(out_dir / "reward_curve.png"),
                )
            except Exception as e:
                main_print(f"  [reward curve] save failed: {e}")

        # ─── Offload VAE / T5 to CPU before training ──────
        _offload_vae_t5(model, to_cpu=True)

        # ════════════════════════════════════════════════════
        #  PHASE 2: GRPO TRAINING  (with grad)
        #
        #  Each rank trains on its N_local_train samples
        #  (after best-of-N filtering).
        #  FSDP reduce-scatter averages gradients across ranks.
        # ════════════════════════════════════════════════════
        S = args.sample_steps
        trainable_steps = S - 1
        train_S = max(1, int(trainable_steps * args.timestep_fraction))

        if args.use_lora:
            # peft keeps LoRA params requires_grad=True permanently;
            # just switch to train mode for dropout etc.
            dit.train()
        else:
            dit.requires_grad_(True)
            dit.train()

        # Zero grad only at the start of each accumulation window
        local_step_idx = step - start_train_step + 1
        accum_idx = (local_step_idx - 1) % args.gradient_accumulation_steps
        if accum_idx == 0:
            optimizer.zero_grad()

        for local_i in range(N_local_train):
            gen_lats = local_lats[local_i]
            gen_lps = local_lps[local_i]
            adv_i = local_advs[local_i].clamp(-args.adv_clip_max, args.adv_clip_max)

            perm = torch.randperm(trainable_steps)[:train_S]

            for ti in perm:
                ti = int(ti.item())

                new_lp = grpo_replay_step(
                    dit,
                    gen_lats[ti],
                    gen_lats[ti + 1],
                    mask2,
                    ctx_c, ctx_n, seq_len,
                    ti, sigmas, args.eta,
                    args.sample_guide_scale,
                    device, model.num_train_timesteps,
                )

                old_lp = gen_lps[ti].to(device)
                ratio = torch.exp(new_lp - old_lp)

                loss_u = -adv_i * ratio
                loss_c = -adv_i * ratio.clamp(
                    1.0 - args.clip_range, 1.0 + args.clip_range,
                )
                loss = torch.maximum(loss_u, loss_c)
                # Normalise by (samples × timesteps × grad_accum_steps)
                loss = loss / (N_local_train * train_S * args.gradient_accumulation_steps)

                loss.backward()
                accum_loss += loss.detach().item()
                accum_n_fwd += 1

        step_loss = accum_loss   # for per-step logging

        # Optimiser step only every gradient_accumulation_steps
        is_optim_step = (accum_idx == args.gradient_accumulation_steps - 1)
        if is_optim_step:
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in dit.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()
            gn_val = gn.item() if isinstance(gn, torch.Tensor) else float(gn)
            main_print(
                f"  [optim step] accum_loss={accum_loss:.6f}  grad_norm={gn_val:.4f}  n_fwd={accum_n_fwd}"
            )
            accum_loss = 0.0
            accum_n_fwd = 0
        else:
            gn_val = 0.0
            main_print(
                f"  [accum {accum_idx+1}/{args.gradient_accumulation_steps}] loss_so_far={accum_loss:.6f}  n_fwd={accum_n_fwd}"
            )

        if not args.use_lora:
            dit.requires_grad_(False)
        dit.eval()

        # ─── logging (rank 0 only) ────────────────────────
        if rank == 0:
            entry = {
                "step": step,
                "loss": step_loss,
                "grad_norm": gn_val,
                "is_optim_step": is_optim_step,
                "accum_idx": accum_idx,
                "mean_reward": rw_tensor.mean().item(),
                "rewards": all_rws_global,
                "advantages": advs.cpu().tolist(),
                "prompt": prompt[:120],
            }
            with log_path.open("a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # ─── checkpoint (FSDP-safe / LoRA-aware) ──────────
        if step % args.checkpointing_steps == 0:
            if args.use_lora:
                cp = ckpt_dir / f"lora_step{step:06d}.pt"
                main_print(f"  Saving LoRA checkpoint -> {cp}")
                if rank == 0:
                    _save_lora_checkpoint(dit, str(cp))
            elif use_fsdp:
                cp = ckpt_dir / f"dit_step{step:06d}.pt"
                main_print(f"  Saving checkpoint -> {cp}")
                from torch.distributed.fsdp import (
                    FullStateDictConfig, StateDictType,
                    FullyShardedDataParallel as FSDP,
                )
                save_policy = FullStateDictConfig(
                    offload_to_cpu=True, rank0_only=True,
                )
                with FSDP.state_dict_type(
                    dit, StateDictType.FULL_STATE_DICT, save_policy,
                ):
                    state_dict = dit.state_dict()
                if rank == 0:
                    torch.save(state_dict, str(cp))
            else:
                cp = ckpt_dir / f"dit_step{step:06d}.pt"
                main_print(f"  Saving checkpoint -> {cp}")
                torch.save(dit.state_dict(), str(cp))

        # ─── cleanup ──────────────────────────────────────
        del local_lats, local_lps, ctx_c, ctx_n, z_img, mask2, img_tensor
        gc.collect()
        torch.cuda.empty_cache()

    # ── flush leftover accumulated gradients ──────────────────────
    leftover = num_steps_this_run % args.gradient_accumulation_steps
    if leftover != 0:
        main_print(f"Flushing leftover {leftover} accumulated steps …")
        if not args.use_lora:
            dit.requires_grad_(True)
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in dit.parameters() if p.requires_grad],
            args.max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad()
        if not args.use_lora:
            dit.requires_grad_(False)
        gn_val = gn.item() if isinstance(gn, torch.Tensor) else float(gn)
        main_print(f"  [final optim step] accum_loss={accum_loss:.6f}  grad_norm={gn_val:.4f}")

    # ── final save ─────────────────────────────────────────────────
    if args.use_lora:
        final_cp = ckpt_dir / "lora_final.pt"
        main_print(f"Saving final LoRA checkpoint -> {final_cp}")
        if rank == 0:
            _save_lora_checkpoint(dit, str(final_cp))
    elif use_fsdp:
        final_cp = ckpt_dir / "dit_final.pt"
        main_print(f"Saving final checkpoint -> {final_cp}")
        from torch.distributed.fsdp import (
            FullStateDictConfig, StateDictType,
            FullyShardedDataParallel as FSDP,
        )
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            dit, StateDictType.FULL_STATE_DICT, save_policy,
        ):
            state_dict = dit.state_dict()
        if rank == 0:
            torch.save(state_dict, str(final_cp))
    else:
        final_cp = ckpt_dir / "dit_final.pt"
        main_print(f"Saving final checkpoint -> {final_cp}")
        torch.save(dit.state_dict(), str(final_cp))

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    main_print("Training complete.")


if __name__ == "__main__":
    main()
