#!/usr/bin/env python3
"""Wan2.2 TI2V SFT (Supervised Fine-Tuning) Training Pipeline.

Trains the Wan2.2 TI2V 5B flow-matching model with standard SFT loss on
ground-truth demonstration videos, following DiffSynth-Studio's implementation.

Pipeline per training step:
  1. Load a ground-truth video + prompt + reference image from the dataset
  2. Encode text (T5), image (VAE), and video (VAE) to latents
  3. Sample random timesteps from the 1000-step training schedule
  4. For each timestep:
     a. Construct noisy latent x_t = (1-σ)*x0 + σ*noise  (flow-matching)
     b. First-frame fusion: keep image latent at frame 0 (mask2 mechanism)
     c. Forward model → velocity prediction
     d. Velocity target = noise - x0
     e. BSMNTW-weighted MSE loss, excluding first frame
  5. Optimizer step (with gradient accumulation)

Key differences from NFT/GRPO:
  • No rollout generation — uses ground-truth videos.
  • No reward scoring — direct supervision via MSE loss.
  • Single LoRA adapter (no dual "old" adapter).
  • 1000-step training sigma schedule (not 20-step inference schedule).
  • BSMNTW timestep weighting (DiffSynth-Studio style).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from fastvideo.utils.logging_ import main_print


# ──────────────────────────────────────────────────────────────────────────────
# Utilities  (shared with GRPO / NFT variants)
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
    """Retrieve the WanAttentionBlock list from a possibly wrapped DiT."""
    candidate = dit
    if hasattr(candidate, "get_base_model"):
        candidate = candidate.get_base_model()
    if hasattr(candidate, "_fsdp_wrapped_module"):
        candidate = candidate._fsdp_wrapped_module
    if hasattr(candidate, "get_base_model"):
        candidate = candidate.get_base_model()
    if hasattr(candidate, "blocks"):
        return candidate.blocks
    raise AttributeError(
        f"Cannot find .blocks on dit of type {type(dit)} "
        f"(unwrapped to {type(candidate)})"
    )


def _enable_gradient_checkpointing(dit):
    """Wrap each WanAttentionBlock.forward with gradient checkpointing."""
    import torch.utils.checkpoint as torch_ckpt

    blocks = _get_dit_blocks(dit)
    for block in blocks:
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
    try:
        import fastvideo.models.wan as wan
        from fastvideo.models.wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
        from fastvideo.models.wan.utils.utils import save_video, masks_like, best_output_size
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
        import wan
        from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
        from wan.utils.utils import save_video, masks_like, best_output_size

    return (
        wan, WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS,
        save_video, masks_like, best_output_size,
    )


def _dit_dtype(dit) -> torch.dtype:
    """Return dtype of the first DiT parameter (handles DDP wrapping)."""
    model = dit.module if hasattr(dit, 'module') else dit
    return next(model.parameters()).dtype


# ──────────────────────────────────────────────────────────────────────────────
# Flow-matching helpers
# ──────────────────────────────────────────────────────────────────────────────

def sd3_time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    """Apply SD3-style time-shift to a sigma schedule."""
    return (shift * t) / (1.0 + (shift - 1.0) * t)


def get_sigma_schedule(
    num_steps: int, shift: float, device: torch.device,
) -> torch.Tensor:
    """Linear sigma schedule 1 -> 0, then time-shifted."""
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    return sd3_time_shift(shift, sigmas)


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
    ts = (mask2[0][:, ::2, ::2] * t_val).flatten()
    ts = torch.cat([ts, ts.new_ones(seq_len - ts.size(0)) * t_val])
    return ts.unsqueeze(0).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# SFT Dataset
# ──────────────────────────────────────────────────────────────────────────────

class SFTVideoDataset(Dataset):
    """Returns (prompt, image_path, video_path, filename_stem) per item.

    JSON format:
      [{"prompt": "...", "media_path": "ref_img.png",
        "video_path": "demo.mp4", "filename_stem": "scene_001"}, ...]

    Paths can be absolute or relative to the JSON file's parent directory.
    """

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
            vp = Path(row.get("video_path", ""))
            if not vp.is_file():
                vp = (p.parent / row.get("video_path", "")).resolve()
            self.items.append({
                "prompt": row["prompt"],
                "media_path": str(mp),
                "video_path": str(vp),
                "filename_stem": row.get("filename_stem", f"sample_{i:06d}"),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Video loading
# ──────────────────────────────────────────────────────────────────────────────

def load_video_frames(
    video_path: str,
    target_frames: int,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Load MP4 video, uniformly sample frames, resize/crop to target size.

    Returns tensor of shape [3, target_frames, target_h, target_w] in [-1, 1].
    """
    reader = imageio.get_reader(video_path, "ffmpeg")
    all_frames = []
    for frame in reader:
        all_frames.append(frame)
    reader.close()

    total = len(all_frames)
    if total >= target_frames:
        indices = [int(i * (total - 1) / (target_frames - 1)) for i in range(target_frames)]
    else:
        indices = list(range(total)) + [total - 1] * (target_frames - total)

    processed = []
    for idx in indices:
        f_pil = Image.fromarray(all_frames[idx])
        iw, ih = f_pil.size
        scale = max(target_w / iw, target_h / ih)
        f_pil = f_pil.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
        x1 = (f_pil.width - target_w) // 2
        y1 = (f_pil.height - target_h) // 2
        f_pil = f_pil.crop((x1, y1, x1 + target_w, y1 + target_h))
        t = TF.to_tensor(f_pil).sub_(0.5).div_(0.5)  # [3, H, W] in [-1, 1]
        processed.append(t)

    return torch.stack(processed, dim=1)  # [3, F, H, W]


# ──────────────────────────────────────────────────────────────────────────────
# BSMNTW timestep weighting
# ──────────────────────────────────────────────────────────────────────────────

def bsmntw_weights(num_timesteps: int = 1000) -> torch.Tensor:
    """Compute BSMNTW timestep weights (DiffSynth-Studio style).

    w(t) = exp(-2 * ((t - 500) / 1000)^2), shifted to zero min,
    normalised so sum = num_timesteps.

    Returns tensor of shape [num_timesteps].
    """
    t = torch.arange(num_timesteps, dtype=torch.float32)
    w = torch.exp(-2.0 * ((t - 500.0) / 1000.0) ** 2)
    w = w - w.min()
    w = w / w.sum() * num_timesteps
    return w


# ──────────────────────────────────────────────────────────────────────────────
# LoRA helpers — single adapter for SFT
# ──────────────────────────────────────────────────────────────────────────────

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

    dit.requires_grad_(False)

    if hasattr(dit, "add_adapter"):
        dit.add_adapter(lora_config)
    else:
        dit = get_peft_model(dit, lora_config)

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
    """Save only the LoRA adapter weights."""
    lora_state = {}
    for name, param in dit.named_parameters():
        if "lora_" in name:
            lora_state[name] = param.detach().cpu()
    torch.save(lora_state, save_path)
    main_print(f"    Saved {len(lora_state)} LoRA tensors ({sum(p.numel() for p in lora_state.values()) / 1e6:.1f} M params)")


# ──────────────────────────────────────────────────────────────────────────────
# Model helpers  (shared with GRPO / NFT variants)
# ──────────────────────────────────────────────────────────────────────────────

def _build_model(args, rank, local_rank, world_size, wan, cfg):
    """Build WanTI2V model."""
    model = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        pt_dir=args.pt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        init_on_cpu=args.offload_model,
        convert_model_dtype=args.convert_model_dtype,
    )
    return model


def _prepare_image(model, img_pil, max_area, best_output_size_fn):
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
    img_tensor = img_tensor.to(model.device).unsqueeze(1)
    return img_tensor, oh, ow


def _compute_seq_len(frame_num, oh, ow, vae_stride, patch_size):
    lat_f = (frame_num - 1) // vae_stride[0] + 1
    lat_h = oh // vae_stride[1]
    lat_w = ow // vae_stride[2]
    return lat_f * lat_h * lat_w // (patch_size[1] * patch_size[2])


# ──────────────────────────────────────────────────────────────────────────────
# SFT loss computation
# ──────────────────────────────────────────────────────────────────────────────

def sft_compute_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    mask2: torch.Tensor,
    timestep_weight: float,
) -> torch.Tensor:
    """Compute weighted MSE flow-matching loss, excluding first frame.

    The mask2 mechanism ensures the first latent frame (reference image)
    does not contribute to the loss.
    """
    v_pred_f = v_pred.float()
    v_target_f = v_target.float()
    mask_f = mask2.float()

    err = ((v_pred_f - v_target_f) ** 2) * mask_f
    loss = err.sum() / mask_f.sum().clamp(min=1.0)
    loss = loss * timestep_weight

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Wan2.2 TI2V SFT Training")

    # model / vidar
    p.add_argument("--vidar_root", type=str, default="",
                    help="Path to vidar repo (optional if wan/ exists in project root)")
    p.add_argument("--task", type=str, default="ti2v-5B")
    p.add_argument("--size", type=str, default="640*736")
    p.add_argument("--frame_num", type=int, default=121)
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--pt_dir", type=str, default=None)
    p.add_argument("--convert_model_dtype", action="store_true", default=False)
    p.add_argument("--offload_model", type=_str2bool, default=False)

    # data
    p.add_argument("--dataset_json", type=str, required=True,
                    help="JSON with {prompt, media_path, video_path, filename_stem} entries")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--output_dir", type=str, required=True)

    # SFT-specific
    p.add_argument("--num_train_timesteps", type=int, default=1000,
                    help="Number of training timesteps (sigma schedule resolution)")
    p.add_argument("--sample_shift", type=float, default=5.0,
                    help="Sigma schedule shift (same as inference)")
    p.add_argument("--timestep_fraction", type=float, default=0.05,
                    help="Fraction of training timesteps to sample per video")
    p.add_argument("--use_bsmntw", type=_str2bool, default=True,
                    help="Use BSMNTW timestep weighting (True) or uniform (False)")
    p.add_argument("--seed", type=int, default=42)

    # training
    p.add_argument("--num_epochs", type=int, default=100,
                    help="Number of training epochs over the dataset")
    p.add_argument("--max_train_steps", type=int, default=0,
                    help="Max global steps (0 = unlimited, controlled by num_epochs)")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=2.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4,
                    help="Accumulate grads over this many samples before optimizer.step()")
    p.add_argument("--checkpointing_steps", type=int, default=10)
    p.add_argument("--log_every", type=int, default=1)

    # memory optimisations
    p.add_argument("--gradient_checkpointing", type=_str2bool, default=True)
    p.add_argument("--use_8bit_adam", type=_str2bool, default=True)

    # LoRA (single adapter)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                    help="LoRA target modules (default: attention q/k/v/o only)")
    p.add_argument("--resume_from_lora_checkpoint", type=str, default=None)

    # wandb
    p.add_argument("--wandb_project", type=str, default=None,
                    help="Wandb project name (None = disable wandb logging)")
    p.add_argument("--wandb_run_name", type=str, default=None,
                    help="Wandb run name (default: auto-generated)")

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
    max_area = MAX_AREA_CONFIGS[args.size]

    # ── directories ────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    for d in (out_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── wandb ───────────────────────────────────────────────────────
    use_wandb = (rank == 0 and args.wandb_project is not None)
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            dir=str(out_dir),
        )
        main_print(f"Wandb logging enabled: project={args.wandb_project}")

    # ── model ──────────────────────────────────────────────────────
    use_ddp = world_size > 1
    main_print(f"Building Wan2.2 TI2V model ... (DDP={use_ddp}, world_size={world_size})")
    model = _build_model(args, rank, local_rank, world_size, wan, cfg)
    dit = model.model

    # ── keep T5 / VAE on GPU when VRAM is sufficient ─────────────
    if not args.offload_model:
        main_print("  Moving T5 encoder to GPU permanently (offload_model=false) ...")
        model.text_encoder.model.to(device)
        model.t5_cpu = False
        torch.cuda.empty_cache()
        main_print("  T5 + VAE will stay on GPU for maximum speed.")

    # ── sigma schedule (1000 training timesteps) ───────────────────
    full_sigmas = get_sigma_schedule(args.num_train_timesteps, args.sample_shift, device)

    # ── BSMNTW weights ─────────────────────────────────────────────
    if args.use_bsmntw:
        bsmntw_w = bsmntw_weights(args.num_train_timesteps).to(device)
        main_print(f"  BSMNTW timestep weighting enabled ({args.num_train_timesteps} steps)")
    else:
        bsmntw_w = torch.ones(args.num_train_timesteps, device=device)
        main_print(f"  Uniform timestep weighting ({args.num_train_timesteps} steps)")

    # ── dataset ────────────────────────────────────────────────────
    ds = SFTVideoDataset(args.dataset_json, args.max_samples)
    main_print(f"Dataset: {len(ds)} samples")

    # ── LoRA (single adapter) ──────────────────────────────────────
    main_print("Applying LoRA (single adapter) for SFT ...")
    dit = _apply_lora(
        dit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        resume_path=args.resume_from_lora_checkpoint,
    )
    model.model = dit

    # ── gradient checkpointing ────────────────────────────────────
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(dit)

    # ── move DiT to GPU and wrap with DDP ──────────────────────────
    dit.to(device)
    torch.cuda.empty_cache()
    if use_ddp:
        dit_ddp = DDP(dit, device_ids=[local_rank], output_device=local_rank,
                       find_unused_parameters=False)
        main_print(f"  DiT wrapped with DDP on {world_size} GPUs")
    else:
        dit_ddp = dit
    dit_raw = dit

    # ── optimizer ──────────────────────────────────────────────────
    trainable_params = [p for p in dit_raw.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    main_print(f"Trainable parameters: {n_params / 1e6:.1f} M")

    if args.use_8bit_adam:
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

    # ── training timestep config ──────────────────────────────────
    num_ts_per_sample = max(1, int(args.num_train_timesteps * args.timestep_fraction))
    main_print(f"  Training {num_ts_per_sample} timesteps per sample "
               f"(fraction={args.timestep_fraction})")

    # ── DataLoader ─────────────────────────────────────────────────
    if use_ddp:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        dl = DataLoader(ds, batch_size=1, sampler=sampler, num_workers=2,
                        pin_memory=False)
    else:
        dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    # ── logging ────────────────────────────────────────────────────
    log_path = out_dir / "training_log.jsonl"
    main_print(
        f"\nStarting SFT | epochs={args.num_epochs}  max_steps={args.max_train_steps}  "
        f"num_ts/sample={num_ts_per_sample}  lr={args.learning_rate}  "
        f"grad_accum={args.gradient_accumulation_steps}  bsmntw={args.use_bsmntw}"
    )

    # ── resume step counter from checkpoint filename ──────────────
    start_step = 0
    if args.resume_from_lora_checkpoint:
        import re
        m = re.search(r"lora_step(\d+)", args.resume_from_lora_checkpoint)
        if m:
            start_step = int(m.group(1))
            main_print(f"Resuming from step {start_step}")

    # ── training loop ──────────────────────────────────────────────
    global_step = start_step
    accum_loss = 0.0
    accum_n_ts = 0
    global_optim_step = 0
    loss_history: List[float] = []

    for epoch in range(args.num_epochs):
        if use_ddp:
            sampler.set_epoch(epoch)

        for batch in dl:
            global_step += 1
            if args.max_train_steps > 0 and global_step > args.max_train_steps:
                break

            prompt = batch["prompt"][0]
            media_path = batch["media_path"][0]
            video_path = batch["video_path"][0]
            fname_stem = batch["filename_stem"][0]

            t_step_start = time.time()

            # ─── encode text (T5) ────────────────────────────────
            if model.t5_cpu:
                model.text_encoder.model.to(device)
            ctx_c = model.text_encoder([prompt], device)
            if model.t5_cpu:
                model.text_encoder.model.cpu()
                torch.cuda.empty_cache()

            # ─── encode reference image (VAE) ───────────────────
            if args.offload_model:
                model.vae.model.to(device)
            img_pil = Image.open(media_path).convert("RGB")
            img_tensor, oh, ow = _prepare_image(
                model, img_pil, max_area, best_output_size,
            )
            z_img = model.vae.encode([img_tensor])[0]  # [48, 1, H_lat, W_lat]

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

            # ─── load and encode ground-truth video (VAE) ────────
            vid_tensor = load_video_frames(video_path, F, oh, ow)  # [3, F, H, W]
            vid_tensor = vid_tensor.to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                x0 = model.vae.encode([vid_tensor])[0]  # [48, F_lat, H_lat, W_lat]
            del vid_tensor

            # ─── build mask2 (first frame = 0, rest = 1) ─────────
            noise_tmp = torch.randn(latent_shape, device=device, dtype=torch.float32)
            _, mask2_list = masks_like([noise_tmp], zero=True)
            mask2 = mask2_list[0].to(device)
            del noise_tmp

            # ─── offload VAE/T5 ──────────────────────────────────
            if args.offload_model:
                _offload_vae_t5(model, to_cpu=True)

            # ─── SFT training: random timesteps ──────────────────
            dit_ddp.train()
            dtype = _dit_dtype(dit_ddp)

            # Zero grad at start of accumulation window
            accum_idx = (global_step - start_step - 1) % args.gradient_accumulation_steps
            if accum_idx == 0:
                optimizer.zero_grad()

            perm = torch.randperm(args.num_train_timesteps, device=device)[:num_ts_per_sample]
            step_loss_sum = 0.0

            for ti in perm:
                ti_idx = int(ti.item())
                sigma = full_sigmas[ti_idx]
                sigma_val = sigma.float()

                # Construct noisy latent: x_t = (1-sigma)*x0 + sigma*noise
                noise = torch.randn_like(x0.float())
                x_t = (1.0 - sigma_val) * x0.float() + sigma_val * noise

                # First-frame fusion: keep image latent at frame 0
                x_t = (1.0 - mask2) * z_img.float() + mask2 * x_t

                # Expand timestep (first-frame tokens get t=0)
                ts = _expand_timestep(mask2, sigma, seq_len, device)
                lat_in = x_t.to(dtype)

                # Forward pass (conditional only, no CFG)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    v_pred = dit_ddp([lat_in], t=ts, context=[ctx_c[0]], seq_len=seq_len)[0]

                # Velocity target: v = noise - x0
                v_target = noise - x0.float()

                # BSMNTW weight for this timestep
                w_t = bsmntw_w[ti_idx].item()

                # Compute loss
                loss = sft_compute_loss(v_pred, v_target, mask2, w_t)

                # Normalise by (timesteps × grad_accum)
                loss = loss / (num_ts_per_sample * args.gradient_accumulation_steps)
                loss.backward()

                step_loss_sum += loss.detach().item()
                del v_pred, v_target, x_t, lat_in, noise, loss

            accum_loss += step_loss_sum
            accum_n_ts += num_ts_per_sample

            # ─── optimizer step ──────────────────────────────────
            is_optim_step = (accum_idx == args.gradient_accumulation_steps - 1)
            if is_optim_step:
                gn = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_optim_step += 1
                gn_val = gn.item() if isinstance(gn, torch.Tensor) else float(gn)
                main_print(
                    f"  [optim step {global_optim_step}] accum_loss={accum_loss:.6f}  "
                    f"grad_norm={gn_val:.4f}  n_ts={accum_n_ts}"
                )
                loss_history.append(accum_loss)
                accum_loss = 0.0
                accum_n_ts = 0
            else:
                gn_val = 0.0

            dit_ddp.eval()

            t_step_end = time.time()

            # ─── logging ──────────────────────────────────────────
            if rank == 0 and global_step % args.log_every == 0:
                main_print(
                    f"  Step {global_step} (epoch {epoch+1}) | {fname_stem} | "
                    f"loss={step_loss_sum:.6f} | {t_step_end - t_step_start:.1f}s"
                )
                entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": step_loss_sum,
                    "grad_norm": gn_val,
                    "is_optim_step": is_optim_step,
                    "global_optim_step": global_optim_step,
                    "prompt": prompt[:120],
                    "filename_stem": fname_stem,
                    "time_s": round(t_step_end - t_step_start, 1),
                }
                with log_path.open("a") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                if use_wandb:
                    wandb.log({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": step_loss_sum,
                        "grad_norm": gn_val,
                        "global_optim_step": global_optim_step,
                    }, step=global_step)

            # ─── checkpoint ──────────────────────────────────────
            if global_step % args.checkpointing_steps == 0:
                cp = ckpt_dir / f"lora_step{global_step:06d}.pt"
                main_print(f"  Saving LoRA checkpoint -> {cp}")
                if rank == 0:
                    _save_lora_checkpoint(dit_raw, str(cp))

            # ─── cleanup ──────────────────────────────────────────
            del x0, z_img, mask2, ctx_c, img_tensor
            torch.cuda.empty_cache()

        # End of epoch
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    # ── flush leftover accumulated gradients ──────────────────────
    actual_steps = global_step - start_step
    leftover = actual_steps % args.gradient_accumulation_steps
    if leftover != 0 and accum_n_ts > 0:
        main_print(f"Flushing leftover {leftover} accumulated steps ...")
        gn = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        global_optim_step += 1

    # ── final save ─────────────────────────────────────────────────
    final_cp = ckpt_dir / "lora_final.pt"
    main_print(f"Saving final LoRA checkpoint -> {final_cp}")
    if rank == 0:
        _save_lora_checkpoint(dit_raw, str(final_cp))

    # ── save loss curve ────────────────────────────────────────────
    if rank == 0 and loss_history:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(1, len(loss_history) + 1), loss_history, "b-o", markersize=3)
            ax.set_xlabel("Optimizer Step")
            ax.set_ylabel("Loss")
            ax.set_title("SFT Training Loss")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(out_dir / "loss_curve.png"), dpi=120)
            plt.close(fig)
            main_print(f"  Loss curve saved to {out_dir / 'loss_curve.png'}")
        except Exception as e:
            main_print(f"  [loss curve] save failed: {e}")

    if use_wandb:
        wandb.finish()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    main_print(f"SFT training complete. {global_step} steps, {global_optim_step} optimizer steps.")


if __name__ == "__main__":
    main()
