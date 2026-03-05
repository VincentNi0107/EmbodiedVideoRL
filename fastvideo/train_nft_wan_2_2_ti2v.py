#!/usr/bin/env python3
"""Wan2.2 TI2V DiffusionNFT Training Pipeline.

Adapted from HunyuanVideo DanceGRPO for the Wan2.2 5B TI2V flow-matching model,
using the DiffusionNFT (Noise-Free Training) algorithm from NVIDIA.

DiffusionNFT principle:
  Instead of policy gradient with log-probs (GRPO/SDE), NFT uses a contrastive
  loss that pushes the model's predicted velocity toward high-reward samples
  and away from low-reward ones — similar to contrastive learning.

Pipeline per training step:
  1. Sample a prompt + first-frame image from the dataset
  2. Encode text (T5) and image (VAE) once
  3. Generate N videos using **deterministic ODE** sampling with the "old" model
     (no SDE noise, no log-prob computation needed)
  4. Decode videos (VAE) and compute rewards
  5. Compute group-wise advantages → clip and normalise to [0, 1]
  6. Training phase: for each sample at random timesteps:
     a. Construct noisy latent x_t = (1-t)*x0 + t*noise  (flow-matching)
     b. Forward current model, old model, and reference model (base w/o LoRA)
     c. Compute NFT contrastive loss + KL divergence penalty
  7. Optimizer step (with gradient accumulation)
  8. Update "old" model weights via exponential moving average

Key differences from the GRPO variant:
  • No SDE sampling — deterministic ODE (faster, simpler).
  • No log-prob computation or storage — saves memory.
  • Loss is contrastive (positive/negative velocity directions) not PPO-clipped.
  • Requires dual LoRA adapters: "default" (trainable) + "old" (delayed copy).
  • During training, only the conditional branch is used (no CFG needed).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP

from fastvideo.utils.logging_ import main_print
from fastvideo.reward import (
    GPTRewardScorer,
    build_reward_scorer,
    add_reward_args,
    save_reward_curve,
)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities  (shared with GRPO variant)
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
    """Linear sigma schedule 1 → 0, then time-shifted."""
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    return sd3_time_shift(shift, sigmas)


def flow_ode_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One Euler-ODE step for flow-matching diffusion (deterministic).

    Returns (next_sample, x0_pred).
    """
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma  # negative (sigma decreases)
    x0_pred = latents - sigma * model_output
    next_sample = latents + dsigma * model_output
    return next_sample, x0_pred


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
# Dataset  (shared with GRPO variant)
# ──────────────────────────────────────────────────────────────────────────────

class GRPOPromptDataset(Dataset):
    """Returns one (prompt, image_path) per item."""

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
# LoRA helpers — dual adapter for NFT  ("default" + "old")
# ──────────────────────────────────────────────────────────────────────────────

WAN_LORA_TARGET_MODULES = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
]


def _apply_lora_nft(
    dit,
    lora_rank: int,
    lora_alpha: int,
    target_modules: Optional[List[str]] = None,
    resume_path: Optional[str] = None,
):
    """Inject LoRA adapters into the DiT for NFT training.

    Creates two adapters:
      • "default" — the trainable adapter (current policy)
      • "old"     — a delayed copy of "default" (used for sampling)

    The base model with all adapters disabled serves as the reference model
    (for KL divergence penalty).
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

    # Freeze base weights
    dit.requires_grad_(False)

    # Create the PeftModel with "default" adapter
    if hasattr(dit, "add_adapter"):
        dit.add_adapter(lora_config)
    else:
        dit = get_peft_model(dit, lora_config)

    # Add the "old" adapter with the same architecture
    dit.add_adapter("old", lora_config)

    # Activate "default" for training
    dit.set_adapter("default")

    # Resume LoRA weights from a previous checkpoint into "default"
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
        main_print(f"  [LoRA resume] loaded {loaded}/{len(saved)} tensors into 'default'")

    # Collect parameter lists for both adapters
    dit.set_adapter("default")
    default_params = list(filter(lambda p: p.requires_grad, dit.parameters()))
    dit.set_adapter("old")
    old_params = list(filter(lambda p: p.requires_grad, dit.parameters()))
    dit.set_adapter("default")

    # Copy "default" weights → "old" so they start identical
    for src_param, tgt_param in zip(default_params, old_params, strict=True):
        tgt_param.data.copy_(src_param.detach().data)
        assert src_param is not tgt_param, "Adapters must not share tensors"

    n_lora = sum(p.numel() for p in default_params)
    n_total = sum(p.numel() for p in dit.parameters())
    main_print(
        f"  LoRA NFT injected: rank={lora_rank}, alpha={lora_alpha}, "
        f"target_modules={target_modules}"
    )
    main_print(
        f"  Trainable (per adapter): {n_lora / 1e6:.1f} M / {n_total / 1e6:.1f} M total "
        f"({100 * n_lora / n_total:.2f}%)"
    )
    return dit, default_params, old_params


def _save_lora_checkpoint(dit, save_path: str):
    """Save only the LoRA adapter weights (default adapter)."""
    lora_state = {}
    for name, param in dit.named_parameters():
        if "lora_" in name:
            lora_state[name] = param.detach().cpu()
    torch.save(lora_state, save_path)
    main_print(f"    Saved {len(lora_state)} LoRA tensors ({sum(p.numel() for p in lora_state.values()) / 1e6:.1f} M params)")


# ──────────────────────────────────────────────────────────────────────────────
# Model helpers  (shared with GRPO variant)
# ──────────────────────────────────────────────────────────────────────────────

def _build_model(args, rank, local_rank, world_size, wan, cfg):
    """Build WanTI2V model.

    DDP approach: every rank loads the full model weights (no FSDP sharding).
    The DiT will be wrapped with DDP externally after LoRA injection.
    """
    model = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        pt_dir=args.pt_dir,            # all ranks load weights for DDP
        device_id=local_rank,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False,                # DDP instead of FSDP
        use_sp=False,
        t5_cpu=True,
        # Avoid large host-memory spikes in DDP: each rank loading on CPU can
        # exceed Slurm memcg limits and trigger SIGKILL. Let offload_model
        # explicitly opt into CPU init/offload behavior.
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
# NFT: Deterministic ODE rollout  (torch.no_grad — eval mode)
# ──────────────────────────────────────────────────────────────────────────────

def ode_rollout_single(
    dit, noise, z_img, mask2,
    ctx_cond, ctx_null, seq_len,
    sigmas, guide_scale, device,
):
    """Run the full **deterministic ODE** denoising loop for one video.

    Unlike the SDE variant, no log-probs are computed.

    Returns
    -------
    x0_pred : Tensor    the model's clean-latent prediction (GPU)
    """
    S = len(sigmas) - 1
    dtype = _dit_dtype(dit)

    # Initial blend: image at frame-0, noise elsewhere
    latent = ((1.0 - mask2) * z_img + mask2 * noise).to(device).float()
    x0_pred = latent

    for i in range(S):
        ts = _expand_timestep(mask2, sigmas[i], seq_len, device)
        lat_in = latent.to(dtype)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v_c = dit([lat_in], t=ts, context=[ctx_cond[0]], seq_len=seq_len)[0]
            v_u = dit([lat_in], t=ts, context=ctx_null, seq_len=seq_len)[0]

        v = (v_u + guide_scale * (v_c - v_u)).float()
        next_lat, x0_pred = flow_ode_step(v, latent, sigmas, i)

        # Re-apply mask: keep image frame intact
        next_lat = (1.0 - mask2) * z_img.float() + mask2 * next_lat
        latent = next_lat.detach()

        del v_c, v_u, v, lat_in

    return x0_pred


def ode_rollout_batch(
    dit, noises, z_img, mask2,
    ctx_cond, ctx_null, seq_len,
    sigmas, guide_scale, device,
    max_batch_cfg=0,
):
    """Batched deterministic ODE rollout with CFG batching.

    Generates B videos simultaneously, batching conditional + unconditional
    forwards into a single DiT call per ODE step.  When B videos are
    combined with CFG, each forward pass processes 2*B samples.

    Args:
        dit          : DiT model (possibly DDP-wrapped)
        noises       : list of B noise tensors, each [C, F, H, W]
        z_img        : encoded first frame [C, 1, H', W'] (shared)
        mask2        : binary mask (shared)
        ctx_cond     : conditional context (list of 1 tensor)
        ctx_null     : unconditional context (list of 1 tensor)
        seq_len      : int
        sigmas       : sigma schedule tensor
        guide_scale  : float, CFG scale
        device       : CUDA device
        max_batch_cfg: max items in a single DiT forward call
                       (0 = 2*B = no sub-batching)

    Returns:
        list of B x0_pred tensors (clean-latent predictions)
    """
    B = len(noises)
    if B == 0:
        return []
    S = len(sigmas) - 1
    dtype = _dit_dtype(dit)

    if max_batch_cfg <= 0:
        max_batch_cfg = 2 * B

    # Videos per sub-batch (each video needs 2 slots: cond + uncond)
    vids_per_sub = max(1, max_batch_cfg // 2)

    # Initialise all latents
    latents = [
        ((1.0 - mask2) * z_img + mask2 * noise).to(device).float()
        for noise in noises
    ]
    x0_preds = list(latents)  # will be overwritten each ODE step

    for i in range(S):
        ts_single = _expand_timestep(mask2, sigmas[i], seq_len, device)  # [1, seq_len]

        all_v_c: List[torch.Tensor] = [None] * B  # type: ignore[assignment]
        all_v_u: List[torch.Tensor] = [None] * B  # type: ignore[assignment]

        for sb_start in range(0, B, vids_per_sub):
            sb_end = min(sb_start + vids_per_sub, B)
            n = sb_end - sb_start

            # Input list: [cond_0 … cond_{n-1}, uncond_0 … uncond_{n-1}]
            lat_inputs = [latents[j].to(dtype) for j in range(sb_start, sb_end)]
            x_list = lat_inputs + lat_inputs          # 2n elements
            context_list = [ctx_cond[0]] * n + [ctx_null[0]] * n
            ts_batch = ts_single.expand(2 * n, -1)   # [2n, seq_len]

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = dit(x_list, t=ts_batch, context=context_list, seq_len=seq_len)

            for k in range(n):
                all_v_c[sb_start + k] = outputs[k]
                all_v_u[sb_start + k] = outputs[n + k]

            del outputs, lat_inputs, x_list

        # CFG + Euler ODE step for each video
        new_latents = []
        for j in range(B):
            v = (all_v_u[j] + guide_scale * (all_v_c[j] - all_v_u[j])).float()
            next_lat, x0_pred = flow_ode_step(v, latents[j], sigmas, i)
            next_lat = (1.0 - mask2) * z_img.float() + mask2 * next_lat
            new_latents.append(next_lat.detach())
            x0_preds[j] = x0_pred

        latents = new_latents
        del all_v_c, all_v_u

    return x0_preds


# ──────────────────────────────────────────────────────────────────────────────
# NFT: contrastive training step
# ──────────────────────────────────────────────────────────────────────────────

def nft_train_forward(
    dit_ddp, x0, z_img, mask2,
    ctx_cond, seq_len,
    sigmas, step_idx,
    device,
    skip_ref: bool = False,
):
    """Run 2-3 forward passes at one timestep: current (grad), old (no grad), optionally ref (no grad).

    Args:
        dit_ddp    : DDP-wrapped (or raw) PeftModel DiT with "default" and "old" adapters
        x0         : clean latent from rollout (C, F, H, W)
        z_img      : encoded first-frame image (C, 1, H', W')
        mask2      : binary mask (1 = noisy region)
        ctx_cond   : conditional text context
        seq_len    : sequence length for DiT
        sigmas     : sigma schedule
        step_idx   : index into the sigma schedule
        device     : CUDA device
        skip_ref   : if True, skip reference model forward (saves ~33% compute)

    Returns (cur_pred, old_pred, ref_pred_or_None, x_t, t_val) — all on GPU.
    """
    dit_raw = dit_ddp.module if hasattr(dit_ddp, 'module') else dit_ddp
    dtype = _dit_dtype(dit_ddp)
    t = sigmas[step_idx]
    t_val = t.float()

    # Construct noisy latent via flow-matching interpolation
    noise = torch.randn_like(x0.float())
    x_t = (1.0 - t_val) * x0.float() + t_val * noise
    # Apply mask: keep first frame clean
    x_t = (1.0 - mask2) * z_img.float() + mask2 * x_t

    ts = _expand_timestep(mask2, t, seq_len, device)
    lat_in = x_t.to(dtype)

    # ── Old model prediction (no grad) ─────────────────────
    dit_raw.set_adapter("old")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        old_pred = dit_ddp([lat_in], t=ts, context=[ctx_cond[0]], seq_len=seq_len)[0].detach()

    # ── Current model prediction (WITH grad) ───────────────
    dit_raw.set_adapter("default")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        cur_pred = dit_ddp([lat_in], t=ts, context=[ctx_cond[0]], seq_len=seq_len)[0]

    # ── Reference model prediction (no grad, LoRA disabled) ─
    #    Skipped when kl_beta ≈ 0 — the KL term is negligible.
    ref_pred = None
    if not skip_ref:
        with torch.no_grad(), dit_raw.disable_adapter():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ref_pred = dit_ddp([lat_in], t=ts, context=[ctx_cond[0]], seq_len=seq_len)[0].detach()

    # Ensure "default" adapter is active after all forwards
    dit_raw.set_adapter("default")

    return cur_pred, old_pred, ref_pred, x_t, t_val


def nft_compute_loss(
    cur_pred: torch.Tensor,
    old_pred: torch.Tensor,
    ref_pred: torch.Tensor,
    x_t: torch.Tensor,
    x0: torch.Tensor,
    t_val: torch.Tensor,
    mask2: torch.Tensor,
    r: float,
    nft_beta: float,
    kl_beta: float,
    adv_clip_max: float,
    temporal_lambda: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the DiffusionNFT contrastive loss for a single sample at one timestep.

    Args:
        cur_pred   : velocity from current "default" model (with grad)
        old_pred   : velocity from "old" model (detached)
        ref_pred   : velocity from reference/base model (detached)
        x_t        : noisy latent at timestep t
        x0         : clean latent (training target)
        t_val      : scalar timestep value (sigma)
        mask2      : binary mask (1 = noisy region)
        r          : normalised advantage in [0, 1]
        nft_beta   : interpolation coefficient (positive/negative blend)
        kl_beta    : KL divergence penalty coefficient
        adv_clip_max : advantage clip maximum (used as loss scale)
        temporal_lambda : weight for temporal consistency loss (0 = disabled)

    Returns (total_loss, loss_terms_dict).
    """
    # Cast to float for loss computation
    cur_f = cur_pred.float()
    old_f = old_pred.float()
    ref_f = ref_pred.float() if ref_pred is not None else None
    x_t_f = x_t.float()
    x0_f = x0.float()
    mask_f = mask2.float()

    # ── Positive and implicit negative predictions ─────────
    # positive: blend current toward old → more conservative update
    positive_pred = nft_beta * cur_f + (1.0 - nft_beta) * old_f
    # negative: mirror of positive around old
    negative_pred = (1.0 + nft_beta) * old_f - nft_beta * cur_f

    # ── x0 predictions from velocity ──────────────────────
    x0_pos = x_t_f - t_val * positive_pred
    x0_neg = x_t_f - t_val * negative_pred

    # ── Adaptive weighting (stabilises early training) ────
    with torch.no_grad():
        diff_pos = torch.abs(x0_pos - x0_f)
        weight_pos = (
            (diff_pos * mask_f).sum()
            / mask_f.sum().clamp(min=1.0)
        ).clamp(min=1e-5)

        diff_neg = torch.abs(x0_neg - x0_f)
        weight_neg = (
            (diff_neg * mask_f).sum()
            / mask_f.sum().clamp(min=1.0)
        ).clamp(min=1e-5)

    # ── Per-element squared error, weighted and masked ─────
    pos_err = ((x0_pos - x0_f) ** 2 / weight_pos) * mask_f
    positive_loss = pos_err.sum() / mask_f.sum().clamp(min=1.0)

    neg_err = ((x0_neg - x0_f) ** 2 / weight_neg) * mask_f
    negative_loss = neg_err.sum() / mask_f.sum().clamp(min=1.0)

    # ── NFT contrastive policy loss ────────────────────────
    # r ∈ [0, 1]:  r=1 → fully positive (push toward high reward)
    #              r=0 → fully negative (push away from low reward)
    ori_policy_loss = r * positive_loss / nft_beta + (1.0 - r) * negative_loss / nft_beta
    policy_loss = ori_policy_loss * adv_clip_max

    # ── KL divergence penalty (against reference model) ────
    if ref_pred is not None:
        kl_err = ((cur_f - ref_f) ** 2) * mask_f
        kl_div = kl_err.sum() / mask_f.sum().clamp(min=1.0)
    else:
        kl_div = torch.zeros(1, device=cur_pred.device)

    # ── Temporal consistency loss (DreamDojo-inspired) ─────
    # Penalises discrepancies in frame-to-frame changes between predicted
    # and target x0, using the same contrastive structure as the policy loss.
    # Temporal diffs along dim=1 (latent frame axis): (C, F, H, W) → (C, F-1, H, W)
    temporal_loss = torch.zeros(1, device=cur_pred.device)
    if temporal_lambda > 0.0:
        x0_pos_tdiff = x0_pos[:, 1:] - x0_pos[:, :-1]
        x0_neg_tdiff = x0_neg[:, 1:] - x0_neg[:, :-1]
        x0_f_tdiff = x0_f[:, 1:] - x0_f[:, :-1]

        # Temporal mask: both adjacent frames must be unmasked
        mask_t = mask_f[:, 1:] * mask_f[:, :-1]

        # Adaptive weighting (same pattern as spatial loss above)
        with torch.no_grad():
            tdiff_pos = torch.abs(x0_pos_tdiff - x0_f_tdiff)
            weight_tpos = (
                (tdiff_pos * mask_t).sum() / mask_t.sum().clamp(min=1.0)
            ).clamp(min=1e-5)

            tdiff_neg = torch.abs(x0_neg_tdiff - x0_f_tdiff)
            weight_tneg = (
                (tdiff_neg * mask_t).sum() / mask_t.sum().clamp(min=1.0)
            ).clamp(min=1e-5)

        temp_pos_err = ((x0_pos_tdiff - x0_f_tdiff) ** 2 / weight_tpos) * mask_t
        temp_pos_loss = temp_pos_err.sum() / mask_t.sum().clamp(min=1.0)

        temp_neg_err = ((x0_neg_tdiff - x0_f_tdiff) ** 2 / weight_tneg) * mask_t
        temp_neg_loss = temp_neg_err.sum() / mask_t.sum().clamp(min=1.0)

        # Contrastive temporal loss (same r weighting and scaling as policy loss)
        temporal_loss = r * temp_pos_loss / nft_beta + (1.0 - r) * temp_neg_loss / nft_beta
        temporal_loss = temporal_loss * adv_clip_max

    total_loss = policy_loss + kl_beta * kl_div + temporal_lambda * temporal_loss

    loss_terms = {
        "policy_loss": policy_loss.detach(),
        "positive_loss": positive_loss.detach(),
        "negative_loss": negative_loss.detach(),
        "kl_div": kl_div.detach(),
        "temporal_loss": temporal_loss.detach(),
        "total_loss": total_loss.detach(),
        "old_deviate": ((cur_f - old_f) ** 2).mean().detach(),
    }
    return total_loss, loss_terms


# ──────────────────────────────────────────────────────────────────────────────
# NFT: old model decay
# ──────────────────────────────────────────────────────────────────────────────

def return_decay(step: int, decay_type: int) -> float:
    """Compute the EMA decay factor for updating the old model.

    decay_type=0 : no decay (old ← current immediately)
    decay_type=1 : gentle ramp-up (DiffusionNFT default)
    decay_type=2 : delayed + steep ramp-up
    """
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def update_old_model(
    default_params: List[torch.nn.Parameter],
    old_params: List[torch.nn.Parameter],
    decay: float,
):
    """EMA update: old ← old * decay + default * (1 - decay)."""
    with torch.no_grad():
        for src, tgt in zip(default_params, old_params, strict=True):
            tgt.data.copy_(
                tgt.detach().data * decay + src.detach().clone().data * (1.0 - decay)
            )


# ──────────────────────────────────────────────────────────────────────────────
# Advantage computation  (modified for NFT normalisation to [0, 1])
# ──────────────────────────────────────────────────────────────────────────────

def compute_advantages(rewards: torch.Tensor, num_gen: int) -> torch.Tensor:
    """Per-group (per-prompt) z-score normalisation of rewards."""
    adv = torch.zeros_like(rewards)
    n = len(rewards) // num_gen
    for i in range(n):
        s, e = i * num_gen, (i + 1) * num_gen
        g = rewards[s:e]
        adv[s:e] = (g - g.mean()) / (g.std() + 1e-8)
    return adv


def normalise_advantages_to_01(
    advantages: torch.Tensor, adv_clip_max: float,
) -> torch.Tensor:
    """Clip advantages and normalise to [0, 1] for NFT loss weighting.

    Maps [-adv_clip_max, adv_clip_max] → [0, 1].
    """
    clipped = advantages.clamp(-adv_clip_max, adv_clip_max)
    return (clipped / adv_clip_max) / 2.0 + 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Wan2.2 TI2V DiffusionNFT Training")

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

    # sampling / ODE
    p.add_argument("--sample_steps", type=int, default=20)
    p.add_argument("--sample_shift", type=float, default=5.0)
    p.add_argument("--sample_guide_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neg_prompt", type=str, default="")

    # reward (defined in fastvideo/reward/builder.py)
    add_reward_args(p)

    # NFT hyper-parameters
    p.add_argument("--num_generations", type=int, default=4,
                    help="N videos per prompt per training step")
    p.add_argument("--rollout_batch_size", type=int, default=0,
                    help="Max CFG batch size during rollout (0 = auto = 2*N_local). "
                         "Reduce if OOM during rollout phase.")
    p.add_argument("--nft_beta", type=float, default=1.0,
                    help="NFT interpolation beta: positive_pred = beta*cur + (1-beta)*old")
    p.add_argument("--kl_beta", type=float, default=0.0001,
                    help="KL divergence penalty coefficient against reference model")
    p.add_argument("--adv_clip_max", type=float, default=5.0,
                    help="Advantage clipping range (before [0,1] normalisation)")
    p.add_argument("--timestep_fraction", type=float, default=0.5,
                    help="Fraction of denoising steps to train on per sample")
    p.add_argument("--decay_type", type=int, default=1, choices=[0, 1, 2],
                    help="Old model EMA decay schedule (0=instant, 1=gentle, 2=delayed)")
    p.add_argument("--temporal_lambda", type=float, default=0.0,
                    help="Weight for temporal consistency loss (0 = disabled, 0.1 = DreamDojo default)")

    # training
    p.add_argument("--max_train_steps", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=2.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4,
                    help="Accumulate grads over this many prompts before optimizer.step()")
    p.add_argument("--checkpointing_steps", type=int, default=10)
    p.add_argument("--log_every", type=int, default=1)

    # memory optimisations
    p.add_argument("--gradient_checkpointing", type=_str2bool, default=True)
    p.add_argument("--use_8bit_adam", type=_str2bool, default=True)

    # LoRA (required for NFT dual-adapter approach)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_target_modules", type=str, nargs="+", default=None)
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
    dit = model.model  # WanModel (DiT)

    # ── keep T5 / VAE on GPU when VRAM is sufficient ─────────────
    if not args.offload_model:
        main_print("  Moving T5 encoder to GPU permanently (offload_model=false) ...")
        model.text_encoder.model.to(device)
        model.t5_cpu = False  # prevent per-step CPU↔GPU transfers
        torch.cuda.empty_cache()
        main_print("  T5 + VAE will stay on GPU for maximum speed.")

    # ── reward scorer ──────────────────────────────────────────────
    main_print(f"Building reward scorer ({args.reward_backend}) ...")
    scorer = build_reward_scorer(args, device)

    # ── sigma schedule ─────────────────────────────────────────────
    sigmas = get_sigma_schedule(args.sample_steps, args.sample_shift, device)

    # ── dataset ────────────────────────────────────────────────────
    ds = GRPOPromptDataset(args.dataset_json, args.max_samples)
    main_print(f"Dataset: {len(ds)} prompts")

    # ── LoRA with dual adapters (required for NFT) ─────────────────
    main_print("Applying LoRA with dual adapters (default + old) for NFT ...")
    dit, default_params, old_params = _apply_lora_nft(
        dit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        resume_path=args.resume_from_lora_checkpoint,
    )
    model.model = dit

    # ── gradient checkpointing (must be applied before DDP wrapping) ──
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
    dit_raw = dit  # unwrapped model for adapter switching, param access, saving

    # ── optimizer ──────────────────────────────────────────────────
    dit_raw.set_adapter("default")
    trainable_params = [p for p in dit_raw.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    main_print(f"Trainable parameters (default adapter): {n_params / 1e6:.1f} M")

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

    dit_ddp.eval()

    # ── distributed rollout sizing ─────────────────────────────────
    N = args.num_generations
    if use_ddp:
        assert N % world_size == 0, (
            f"num_generations ({N}) must be divisible by world_size ({world_size})")
        N_local = N // world_size
    else:
        N_local = N

    # ── training timestep config ──────────────────────────────────
    S = args.sample_steps
    trainable_steps = S  # can train on any timestep in the schedule
    train_S = max(1, int(trainable_steps * args.timestep_fraction))

    # ── logging ────────────────────────────────────────────────────
    log_path = out_dir / "training_log.jsonl"
    main_print(
        f"\nStarting DiffusionNFT | steps={args.max_train_steps}  N={N}  "
        f"N_local/rank={N_local}  nft_beta={args.nft_beta}  kl_beta={args.kl_beta}  "
        f"lr={args.learning_rate}  sample_steps={args.sample_steps}  "
        f"timestep_frac={args.timestep_fraction}  decay_type={args.decay_type}"
    )

    # ── training loop ──────────────────────────────────────────────
    dl_generator = torch.Generator().manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0,
                    generator=dl_generator)
    data_iter = iter(dl)
    accum_loss = 0.0
    accum_n_fwd = 0
    global_optim_step = 0
    reward_history_steps: List[int] = []
    reward_history_means: List[float] = []

    # ── resume step counter from checkpoint filename ──────────────
    start_step = 0
    if args.resume_from_lora_checkpoint:
        import re
        m = re.search(r"lora_step(\d+)", args.resume_from_lora_checkpoint)
        if m:
            start_step = int(m.group(1))
            main_print(f"Resuming from step {start_step}, will train steps {start_step+1}..{args.max_train_steps}")
            if start_step >= args.max_train_steps:
                raise ValueError(
                    f"resume step ({start_step}) >= max_train_steps ({args.max_train_steps}). "
                    f"Increase --max_train_steps to continue training."
                )
            # Load historical reward data from training_log.jsonl
            if log_path.exists():
                import json as _json
                for line in log_path.read_text().splitlines():
                    try:
                        entry = _json.loads(line)
                        s = entry.get("step")
                        rw = entry.get("mean_reward")
                        if s is not None and rw is not None and s <= start_step:
                            reward_history_steps.append(s)
                            reward_history_means.append(rw)
                    except Exception:
                        continue
                main_print(f"  Loaded {len(reward_history_steps)} historical reward entries from training_log.jsonl")

    for step in range(start_step + 1, args.max_train_steps + 1):
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
            model.text_encoder.model.to(device)
        ctx_c = model.text_encoder([prompt], device)
        ctx_n = model.text_encoder([args.neg_prompt], device)
        if model.t5_cpu:
            model.text_encoder.model.cpu()
            torch.cuda.empty_cache()

        # ─── encode image (VAE) ───────────────────────────
        if args.offload_model:
            model.vae.model.to(device)
        img_pil = Image.open(media_path).convert("RGB")
        img_tensor, oh, ow = _prepare_image(
            model, img_pil, max_area, best_output_size,
        )
        z_img = model.vae.encode([img_tensor])[0]

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

        # DiT is already on GPU (moved before DDP wrapping)

        # ════════════════════════════════════════════════════
        #  PHASE 1a: ROLLOUT — batched ODE with "old" adapter
        #  Batches multiple videos AND CFG (cond+uncond) into
        #  a single DiT forward per ODE step for throughput.
        # ════════════════════════════════════════════════════
        dit_ddp.eval()
        dit_raw.set_adapter("old")  # Use old model for sampling

        scene_suffix = fname_stem[-4:] if len(fname_stem) >= 4 else fname_stem
        step_vid_dir = vid_dir / f"step{step:04d}_{scene_suffix}"
        step_reward_dir = reward_dir / f"step{step:04d}_{scene_suffix}"
        step_vid_dir.mkdir(parents=True, exist_ok=True)
        step_reward_dir.mkdir(parents=True, exist_ok=True)

        # Generate all noise tensors first
        base_seed = args.seed + step * N
        noises: List[torch.Tensor] = []
        video_meta: List[Tuple[int, int]] = []  # (global_idx, seed)
        for g_local in range(N_local):
            g = rank * N_local + g_local if use_ddp else g_local
            seed_g = base_seed + g
            gen = torch.Generator(device=device).manual_seed(seed_g)
            noise = torch.randn(
                latent_shape, dtype=torch.float32,
                generator=gen, device=device,
            )
            noises.append(noise)
            video_meta.append((g, seed_g))

        # Batched rollout: all N_local videos + CFG in one forward per ODE step
        t_rollout_start = time.time()
        local_x0_preds = ode_rollout_batch(
            dit_ddp, noises, z_img, mask2,
            ctx_c, ctx_n, seq_len,
            sigmas, args.sample_guide_scale,
            device,
            max_batch_cfg=args.rollout_batch_size,
        )
        local_x0_preds = [x.detach() for x in local_x0_preds]
        t_rollout_end = time.time()
        main_print(
            f"  Rollout: {N_local} videos in {t_rollout_end - t_rollout_start:.1f}s "
            f"(batch_cfg={min(args.rollout_batch_size or 2*N_local, 2*N_local)})"
        )
        del noises

        # Decode each video and prepare for reward scoring
        pending_scores: List[tuple] = []
        for g_local in range(N_local):
            g, seed_g = video_meta[g_local]
            final_lat = local_x0_preds[g_local].to(device, dtype=model.param_dtype)
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
                dbg_base = f"{fname_stem}_g{g:03d}_s{seed_g}"
                pending_scores.append((vp, dbg_base))
            else:
                pending_scores.append((None, None))

            del final_lat

        dit_raw.set_adapter("default")  # Switch back to default adapter

        # ════════════════════════════════════════════════════
        #  PHASE 1b: REWARD SCORING
        # ════════════════════════════════════════════════════
        local_rws: List[float] = []
        for vp, dbg_base in pending_scores:
            if vp is not None:
                dbg_path = str(step_reward_dir / f"{dbg_base}.jpg")
                try:
                    rd = scorer.score(
                        prompt=prompt,
                        first_frame=None,
                        video_path=vp,
                        debug_save_path=dbg_path,
                    )
                except Exception as exc:
                    rd = {"reward": -10.0, "error": str(exc)}
                rw_val = float(rd["reward"])
                tag = "PASS" if rw_val > 0 else "FAIL"
                # GPT/Gemini debug: save grid image with response text
                grid_bgr = rd.get("_grid_bgr")
                resp_text = rd.get("_response_text")
                if grid_bgr is not None and resp_text is not None:
                    tagged_path = str(step_reward_dir / f"{dbg_base}_{tag}.jpg")
                    try:
                        GPTRewardScorer._save_debug_image(grid_bgr, resp_text, tagged_path)
                    except Exception as e:
                        main_print(f"  [debug img] save failed: {e}")
                # Hallucination debug: log annotated video path
                annotated_vid = rd.get("annotated_video")
                if annotated_vid:
                    main_print(f"  [hall reward] {dbg_base}: {resp_text or ''}")
            else:
                rd = {"reward": -10.0, "error": "video_none"}
                rw_val = float(rd["reward"])
            local_rws.append(rw_val)

        # ─── gather rewards across ranks ──────────────────
        if use_ddp:
            local_rw_t = torch.tensor(local_rws, dtype=torch.float32, device=device)
            gathered = [torch.zeros_like(local_rw_t) for _ in range(world_size)]
            dist.all_gather(gathered, local_rw_t)
            all_rws_global = torch.cat(gathered).tolist()
        else:
            all_rws_global = local_rws

        rw_tensor = torch.tensor(all_rws_global, dtype=torch.float32, device=device)
        advs = compute_advantages(rw_tensor, N)
        local_advs = advs[rank * N_local:(rank + 1) * N_local] if use_ddp else advs

        # Normalise advantages to [0, 1] for NFT
        local_r = normalise_advantages_to_01(local_advs, args.adv_clip_max)

        all_r = normalise_advantages_to_01(advs, args.adv_clip_max)
        step_mean_rw = rw_tensor.mean().item()
        main_print(f"  rewards : {[f'{r:.3f}' for r in all_rws_global]}")
        main_print(f"  mean_rw : {step_mean_rw:.4f}")
        main_print(f"  r (normalised advs) : {[f'{v:.3f}' for v in all_r.cpu().tolist()]}")

        # ─── Check for uniform rewards (all same → zero contrastive signal) ──
        skip_training = (rw_tensor.max() - rw_tensor.min()).item() < 1e-6
        if skip_training:
            main_print(
                f"  ⚠ Uniform rewards detected (all {rw_tensor[0].item():.1f}) — "
                f"skipping training for this step (zero contrastive gradient)."
            )

        # ─── Update reward curve ──────────────────────────
        reward_history_steps.append(step)
        reward_history_means.append(step_mean_rw)
        if rank == 0:
            try:
                save_reward_curve(
                    reward_history_steps,
                    reward_history_means,
                    str(out_dir / "reward_curve.png"),
                )
            except Exception as e:
                main_print(f"  [reward curve] save failed: {e}")

        # ─── Offload VAE / T5 to CPU before training (skip if VRAM is sufficient) ──
        if args.offload_model:
            _offload_vae_t5(model, to_cpu=True)

        # ════════════════════════════════════════════════════
        #  PHASE 2: NFT TRAINING (with grad)
        #  — skipped when rewards are uniform (no contrastive signal)
        # ════════════════════════════════════════════════════
        dit_raw.set_adapter("default")
        dit_ddp.train()

        # Zero grad at start of accumulation window
        accum_idx = (step - 1) % args.gradient_accumulation_steps
        if accum_idx == 0:
            optimizer.zero_grad()

        if not skip_training:
            step_loss_terms: Dict[str, List[float]] = {
                "policy_loss": [], "positive_loss": [], "negative_loss": [],
                "kl_div": [], "temporal_loss": [], "total_loss": [], "old_deviate": [],
            }

            for local_i in range(N_local):
                x0_i = local_x0_preds[local_i].to(device)
                r_i = local_r[local_i].item()

                # Sample random subset of timesteps
                perm = torch.randperm(trainable_steps)[:train_S]

                for ti in perm:
                    ti_idx = int(ti.item())

                    cur_pred, old_pred, ref_pred, x_t, t_val = nft_train_forward(
                        dit_ddp, x0_i, z_img, mask2,
                        ctx_c, seq_len,
                        sigmas, ti_idx,
                        device,
                        skip_ref=(args.kl_beta < 1e-3),
                    )

                    loss, loss_terms = nft_compute_loss(
                        cur_pred, old_pred, ref_pred,
                        x_t, x0_i, t_val, mask2,
                        r=r_i,
                        nft_beta=args.nft_beta,
                        kl_beta=args.kl_beta,
                        adv_clip_max=args.adv_clip_max,
                        temporal_lambda=args.temporal_lambda,
                    )

                    # Normalise by (samples × timesteps × grad_accum_steps)
                    loss = loss / (N_local * train_S * args.gradient_accumulation_steps)

                    loss.backward()
                    accum_loss += loss.detach().item()
                    accum_n_fwd += 1

                    for k, v in loss_terms.items():
                        step_loss_terms[k].append(v.item())

                    del cur_pred, old_pred, ref_pred, x_t, loss

                del x0_i
        else:
            step_loss_terms = {}

        step_loss = accum_loss

        # Optimiser step only every gradient_accumulation_steps
        is_optim_step = (accum_idx == args.gradient_accumulation_steps - 1)
        if is_optim_step:
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in dit_raw.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()
            global_optim_step += 1
            gn_val = gn.item() if isinstance(gn, torch.Tensor) else float(gn)
            main_print(
                f"  [optim step {global_optim_step}] accum_loss={accum_loss:.6f}  "
                f"grad_norm={gn_val:.4f}  n_fwd={accum_n_fwd}"
            )
            accum_loss = 0.0
            accum_n_fwd = 0

            # ── Update old model weights via decay ─────────
            decay = return_decay(global_optim_step, args.decay_type)
            update_old_model(default_params, old_params, decay)
            main_print(f"  [old model update] decay={decay:.6f}")
        else:
            gn_val = 0.0
            main_print(
                f"  [accum {accum_idx+1}/{args.gradient_accumulation_steps}] "
                f"loss_so_far={accum_loss:.6f}  n_fwd={accum_n_fwd}"
            )

        dit_ddp.eval()

        # ─── logging ──────────────────────────────────────
        if rank == 0:
            avg_terms = {k: sum(v) / len(v) if v else 0.0 for k, v in step_loss_terms.items()}
            entry = {
                "step": step,
                "loss": step_loss,
                "grad_norm": gn_val,
                "is_optim_step": is_optim_step,
                "accum_idx": accum_idx,
                "global_optim_step": global_optim_step,
                "mean_reward": rw_tensor.mean().item(),
                "rewards": all_rws_global,
                "advantages": advs.cpu().tolist(),
                "r_normalised": local_r.cpu().tolist(),
                "prompt": prompt[:120],
                **{f"avg_{k}": v for k, v in avg_terms.items()},
            }
            with log_path.open("a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if use_wandb:
                wb_log = {
                    "step": step,
                    "mean_reward": entry["mean_reward"],
                    "loss": entry["loss"],
                    "grad_norm": entry["grad_norm"],
                    "global_optim_step": entry["global_optim_step"],
                }
                for k in ("avg_policy_loss", "avg_positive_loss", "avg_negative_loss",
                          "avg_kl_div", "avg_temporal_loss", "avg_total_loss", "avg_old_deviate"):
                    if k in entry:
                        wb_log[k] = entry[k]
                wandb.log(wb_log, step=step)

        # ─── checkpoint ──────────────────────────────────
        if step % args.checkpointing_steps == 0:
            cp = ckpt_dir / f"lora_step{step:06d}.pt"
            main_print(f"  Saving LoRA checkpoint -> {cp}")
            if rank == 0:
                dit_raw.set_adapter("default")
                _save_lora_checkpoint(dit_raw, str(cp))

        # ─── cleanup ──────────────────────────────────────
        del local_x0_preds, ctx_c, ctx_n, z_img, mask2, img_tensor
        torch.cuda.empty_cache()

    # ── flush leftover accumulated gradients ──────────────────────
    actual_steps = args.max_train_steps - start_step
    leftover = actual_steps % args.gradient_accumulation_steps
    if leftover != 0:
        main_print(f"Flushing leftover {leftover} accumulated steps ...")
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in dit_raw.parameters() if p.requires_grad],
            args.max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad()
        global_optim_step += 1
        gn_val = gn.item() if isinstance(gn, torch.Tensor) else float(gn)
        main_print(f"  [final optim step] accum_loss={accum_loss:.6f}  grad_norm={gn_val:.4f}")

        # Final old model update
        decay = return_decay(global_optim_step, args.decay_type)
        update_old_model(default_params, old_params, decay)

    # ── final save ─────────────────────────────────────────────────
    final_cp = ckpt_dir / "lora_final.pt"
    main_print(f"Saving final LoRA checkpoint -> {final_cp}")
    if rank == 0:
        dit_raw.set_adapter("default")
        _save_lora_checkpoint(dit_raw, str(final_cp))

    # ── clean up reward scorer resources ──────────────────────────
    if hasattr(scorer, 'shutdown'):
        main_print("Shutting down reward scorer ...")
        scorer.shutdown()

    if use_wandb:
        wandb.finish()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    main_print("DiffusionNFT training complete.")


if __name__ == "__main__":
    main()
