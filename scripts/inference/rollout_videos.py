#!/usr/bin/env python3
"""Multi-GPU parallel video rollout using Wan2.2 TI2V.

Extracted from DanceGRPO's multi-card parallel rollout logic.
Reads a JSON dataset, generates N rollout videos per sample with different
random seeds, and saves all videos to disk.

Parallelism strategy (data-parallel, no FSDP):
    - Each GPU loads the full model independently.
    - Data items are distributed across GPUs in round-robin fashion.
    - Each GPU generates all `num_rollouts` videos for its assigned items.
    - No inter-GPU communication needed during rollout — linear scaling.

Usage (4 GPUs, 8 rollouts per data item):
    torchrun --nproc_per_node=4 rollout_videos.py \
        --dataset_json /path/to/robotwin_121.json \
        --output_dir /path/to/output \
        --num_rollouts 8 \
        --ckpt_dir /path/to/Wan2.2-TI2V-5B \
        --pt_dir /path/to/merged_lora.pt

Single-GPU:
    python rollout_videos.py \
        --dataset_json /path/to/robotwin_121.json \
        --output_dir /path/to/output \
        --num_rollouts 8 \
        --ckpt_dir /path/to/Wan2.2-TI2V-5B
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ──────────────────────────────────────────────────────────────────────────────

def init_distributed():
    """Initialize distributed process group (if launched via torchrun)."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def log_rank(msg, rank, target_rank=0):
    """Print only on the specified rank (default: rank 0)."""
    if rank == target_rank:
        print(msg, flush=True)


def log_all(msg, rank):
    """Print on all ranks with rank prefix."""
    print(f"  [Rank {rank}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# SDE / Flow-matching helpers (from DanceGRPO)
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


def expand_timestep(
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


def flow_sde_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One Euler-SDE step for flow-matching diffusion (inference only, no log-prob).

    The flow model predicts  v_theta(x_t, t).
      x0_pred  = x_t - sigma_t * v_theta
      ODE mean = x_t + d_sigma * v_theta          (Euler step)
      SDE      = ODE mean + score correction + N(0, std^2)

    Returns:
        (next_sample, x0_pred)
    """
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    x0_pred = latents - sigma * model_output
    prev_sample_mean = latents + dsigma * model_output

    delta_t = (sigma - sigmas[index + 1]).clamp(min=1e-10)
    std_dev_t = eta * delta_t.sqrt()

    # Langevin-style score correction so that SDE marginals match the ODE
    if eta > 0:
        score = -(latents - x0_pred * (1.0 - sigma)) / (sigma ** 2 + 1e-10)
        prev_sample_mean = prev_sample_mean + (-0.5 * eta ** 2 * score) * dsigma

    # Sample
    prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    return prev_sample, x0_pred


# ──────────────────────────────────────────────────────────────────────────────
# SDE rollout (inference only, torch.no_grad)
# ──────────────────────────────────────────────────────────────────────────────

def sde_rollout(
    dit, noise, z_img, mask2,
    ctx_cond, ctx_null, seq_len,
    sigmas, eta, guide_scale, device,
):
    """Full SDE denoising loop for one video (inference only).

    Args:
        dit        : WanModel (DiT) on GPU
        noise      : initial noise tensor (C, F, H, W)
        z_img      : VAE-encoded first frame (C, 1, H', W')
        mask2      : binary mask (1 = noisy, 0 = image frame)
        ctx_cond   : T5 conditional text embeddings
        ctx_null   : T5 null/negative text embeddings
        seq_len    : flattened sequence length for DiT
        sigmas     : sigma schedule (num_steps+1,)
        eta        : SDE noise level (0 = deterministic ODE)
        guide_scale: classifier-free guidance scale
        device     : target CUDA device

    Returns:
        x0_pred : final clean latent prediction (C, F, H, W) on GPU
    """
    S = len(sigmas) - 1
    dtype = next(dit.parameters()).dtype

    # Initial blend: image at frame-0, noise elsewhere
    latent = ((1.0 - mask2) * z_img + mask2 * noise).to(device).float()
    x0_pred = latent  # placeholder

    for i in range(S):
        ts = expand_timestep(mask2, sigmas[i], seq_len, device)
        lat_in = latent.to(dtype)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Conditional forward
            v_c = dit([lat_in], t=ts, context=[ctx_cond[0]], seq_len=seq_len)[0]
            # Unconditional forward
            v_u = dit([lat_in], t=ts, context=ctx_null, seq_len=seq_len)[0]

        # CFG combination
        v = (v_u + guide_scale * (v_c - v_u)).float()

        next_lat, x0_pred = flow_sde_step(
            v, latent, eta, sigmas, i, mask=mask2,
        )
        # Re-apply mask: keep image frame intact
        next_lat = (1.0 - mask2) * z_img.float() + mask2 * next_lat
        latent = next_lat.detach()

        del v_c, v_u, v, lat_in

    return x0_pred


# ──────────────────────────────────────────────────────────────────────────────
# Image preparation
# ──────────────────────────────────────────────────────────────────────────────

def prepare_image(model, img_pil, max_area, best_output_size_fn):
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


def compute_seq_len(frame_num, oh, ow, vae_stride, patch_size):
    lat_f = (frame_num - 1) // vae_stride[0] + 1
    lat_h = oh // vae_stride[1]
    lat_w = ow // vae_stride[2]
    return lat_f * lat_h * lat_w // (patch_size[1] * patch_size[2])


# ──────────────────────────────────────────────────────────────────────────────
# LoRA helpers
# ──────────────────────────────────────────────────────────────────────────────

WAN_LORA_TARGET_MODULES = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
]


def apply_lora(dit, lora_rank, lora_alpha, target_modules=None,
               resume_path=None):
    """Inject LoRA adapters and optionally load checkpoint weights."""
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = WAN_LORA_TARGET_MODULES

    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha,
        init_lora_weights=True, target_modules=target_modules,
    )
    dit.requires_grad_(False)
    if hasattr(dit, "add_adapter"):
        dit.add_adapter(lora_config)
    else:
        dit = get_peft_model(dit, lora_config)

    if resume_path:
        print(f"  Loading LoRA checkpoint from {resume_path}")
        saved = torch.load(resume_path, map_location="cpu")
        model_dict = dict(dit.named_parameters())
        loaded = 0
        for k, v in saved.items():
            if k in model_dict:
                model_dict[k].data.copy_(v)
                loaded += 1
        print(f"  [LoRA] loaded {loaded}/{len(saved)} tensors")

    return dit


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-GPU Parallel Video Rollout (Wan2.2 TI2V)")

    # vidar / model
    p.add_argument("--vidar_root", type=str, default="",
                    help="Path to vidar repo (optional if wan/ exists in project root)")
    p.add_argument("--task", type=str, default="ti2v-5B")
    p.add_argument("--size", type=str, default="640*736")
    p.add_argument("--frame_num", type=int, default=121)
    p.add_argument("--ckpt_dir", type=str, required=True,
                    help="Path to Wan2.2-TI2V-5B base checkpoint directory")
    p.add_argument("--pt_dir", type=str, default=None,
                    help="Path to fine-tuned DiT weights (.pt/.safetensors)")
    p.add_argument("--convert_model_dtype", action="store_true", default=True,
                    help="Convert DiT to bf16 (saves VRAM)")

    # data
    p.add_argument("--dataset_json", type=str, required=True,
                    help="Path to JSON dataset (list of {prompt, media_path, filename_stem})")
    p.add_argument("--output_dir", type=str, required=True,
                    help="Root directory for saving output videos")

    # sampling / SDE
    p.add_argument("--sample_steps", type=int, default=20,
                    help="Number of denoising steps")
    p.add_argument("--sample_shift", type=float, default=5.0,
                    help="SD3 time-shift factor for sigma schedule")
    p.add_argument("--sample_guide_scale", type=float, default=5.0,
                    help="Classifier-free guidance scale")
    p.add_argument("--eta", type=float, default=1.0,
                    help="SDE noise level. 0 = deterministic ODE, >0 = SDE for diversity")
    p.add_argument("--seed", type=int, default=42,
                    help="Base random seed (each rollout uses seed + rollout_index)")
    p.add_argument("--neg_prompt", type=str, default="",
                    help="Negative prompt (empty = use model default)")
    p.add_argument("--num_rollouts", type=int, default=8,
                    help="Number of rollout videos per data item")

    # LoRA (optional)
    p.add_argument("--use_lora", type=lambda v: v.lower() in ("true", "1", "yes"),
                    default=False, help="Enable LoRA adapter on DiT")
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_checkpoint", type=str, default=None,
                    help="Path to LoRA checkpoint (.pt) to load")

    # misc
    p.add_argument("--skip_existing", action="store_true", default=False,
                    help="Skip rollout if video file already exists (for resuming)")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    # ── Import wan module (fastvideo.models.wan or vidar_root fallback) ──
    try:
        import fastvideo.models.wan as wan
        from fastvideo.models.wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
        from fastvideo.models.wan.utils.utils import save_video, masks_like, best_output_size
    except ImportError:
        if not args.vidar_root:
            raise ImportError(
                "Cannot import 'fastvideo.models.wan' module. Either ensure "
                "fastvideo/models/wan/ exists or pass --vidar_root pointing to "
                "the vidar repository."
            )
        vidar_path = Path(args.vidar_root).resolve()
        if not vidar_path.exists():
            raise FileNotFoundError(f"vidar_root not found: {vidar_path}")
        if str(vidar_path) not in sys.path:
            sys.path.insert(0, str(vidar_path))
        import wan
        from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
        from wan.utils.utils import save_video, masks_like, best_output_size

    cfg = WAN_CONFIGS[args.task]
    if not args.neg_prompt:
        args.neg_prompt = cfg.sample_neg_prompt
    max_area = MAX_AREA_CONFIGS[args.size]

    # ── Load dataset ─────────────────────────────────────────────────────
    dataset_path = Path(args.dataset_json).resolve()
    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Resolve media paths
    for i, row in enumerate(dataset):
        mp = Path(row.get("media_path", ""))
        if not mp.is_file():
            mp = (dataset_path.parent / row.get("media_path", "")).resolve()
        dataset[i]["media_path"] = str(mp)
        if "filename_stem" not in row:
            dataset[i]["filename_stem"] = f"sample_{i:06d}"

    # ── Split data across GPUs (round-robin) ─────────────────────────────
    my_indices = [i for i in range(len(dataset)) if i % world_size == rank]
    my_items = [dataset[i] for i in my_indices]

    log_rank(
        f"{'=' * 70}\n"
        f"  Multi-GPU Parallel Video Rollout\n"
        f"  Total samples : {len(dataset)}\n"
        f"  Num rollouts  : {args.num_rollouts} per sample\n"
        f"  Total videos  : {len(dataset) * args.num_rollouts}\n"
        f"  World size    : {world_size} GPUs\n"
        f"  Sample steps  : {args.sample_steps}\n"
        f"  Guide scale   : {args.sample_guide_scale}\n"
        f"  SDE eta       : {args.eta}\n"
        f"  Frame num     : {args.frame_num}\n"
        f"  Base seed     : {args.seed}\n"
        f"{'=' * 70}",
        rank,
    )
    log_all(f"assigned {len(my_items)} samples (indices: {my_indices})", rank)

    # ── Build model (each GPU loads full model, NO FSDP) ─────────────────
    log_rank("Building Wan2.2 TI2V model (per-GPU, no FSDP) ...", rank)
    model = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        pt_dir=args.pt_dir,
        device_id=local_rank,
        rank=0,           # each GPU is independent, act as rank 0
        t5_fsdp=False,
        dit_fsdp=False,   # no FSDP — full model per GPU
        use_sp=False,
        t5_cpu=True,      # T5 on CPU to save GPU VRAM
        init_on_cpu=True,
        convert_model_dtype=args.convert_model_dtype,
    )
    dit = model.model  # WanModel (DiT)

    # ── LoRA (optional) ──────────────────────────────────────────────────
    if args.use_lora and args.lora_checkpoint:
        log_all("Applying LoRA ...", rank)
        dit = apply_lora(
            dit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            resume_path=args.lora_checkpoint,
        )
        model.model = dit

    dit.eval()
    dit.requires_grad_(False)

    # Move DiT to GPU (was init_on_cpu)
    dit.to(device)
    torch.cuda.empty_cache()
    log_all("DiT loaded on GPU", rank)

    # Move VAE to GPU (keep it there for both encode + decode)
    model.vae.model.to(device)
    log_all("VAE loaded on GPU", rank)

    # ── Sigma schedule ───────────────────────────────────────────────────
    sigmas = get_sigma_schedule(args.sample_steps, args.sample_shift, device)

    # ── Output directory ─────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N = args.num_rollouts
    total_generated = 0
    total_skipped = 0
    t_start_all = time.time()

    # ══════════════════════════════════════════════════════════════════════
    #  ROLLOUT LOOP: iterate over assigned data items
    # ══════════════════════════════════════════════════════════════════════
    for item_idx, item in enumerate(my_items):
        prompt = item["prompt"]
        media_path = item["media_path"]
        fname_stem = item["filename_stem"]
        t_item_start = time.time()

        log_all(
            f"[{item_idx + 1}/{len(my_items)}] {fname_stem}",
            rank,
        )

        # Create per-item output directory
        item_dir = out_dir / fname_stem
        item_dir.mkdir(parents=True, exist_ok=True)

        # Check if all rollouts already exist (for --skip_existing)
        if args.skip_existing:
            existing = sum(
                1 for g in range(N)
                if (item_dir / f"{fname_stem}_g{g:03d}_s{args.seed + g}.mp4").exists()
            )
            if existing == N:
                log_all(f"  all {N} videos exist, skipping", rank)
                total_skipped += N
                continue

        # ── Encode text (T5: move to GPU -> encode -> move back to CPU) ──
        model.text_encoder.model.to(device)
        ctx_c = model.text_encoder([prompt], device)
        ctx_n = model.text_encoder([args.neg_prompt], device)
        model.text_encoder.model.cpu()
        torch.cuda.empty_cache()

        # ── Encode image (VAE is already on GPU) ────────────────────────
        img_pil = Image.open(media_path).convert("RGB")
        img_tensor, oh, ow = prepare_image(
            model, img_pil, max_area, best_output_size,
        )
        z_img = model.vae.encode([img_tensor])[0]  # (C, 1, H', W')

        F = args.frame_num
        seq_len = compute_seq_len(F, oh, ow, model.vae_stride, model.patch_size)
        latent_shape = (
            model.vae.model.z_dim,
            (F - 1) // model.vae_stride[0] + 1,
            oh // model.vae_stride[1],
            ow // model.vae_stride[2],
        )

        # ── Mask (first frame = image, rest = noise) ────────────────────
        noise_tmp = torch.randn(latent_shape, device=device, dtype=torch.float32)
        _, mask2_list = masks_like([noise_tmp], zero=True)
        mask2 = mask2_list[0].to(device)
        del noise_tmp

        # ══════════════════════════════════════════════════════════════════
        #  Generate N rollouts for this data item
        # ══════════════════════════════════════════════════════════════════
        for g in range(N):
            seed_g = args.seed + g
            video_path = item_dir / f"{fname_stem}_g{g:03d}_s{seed_g}.mp4"

            # Skip if already exists
            if args.skip_existing and video_path.exists():
                total_skipped += 1
                continue

            t_rollout_start = time.time()

            # Seed the initial noise with a per-rollout seed
            gen = torch.Generator(device=device).manual_seed(seed_g)
            noise = torch.randn(
                latent_shape, dtype=torch.float32,
                generator=gen, device=device,
            )

            # ── SDE rollout (full denoising loop) ───────────────────────
            x0_pred = sde_rollout(
                dit, noise, z_img, mask2,
                ctx_c, ctx_n, seq_len,
                sigmas, args.eta, args.sample_guide_scale,
                device,
            )

            # ── Decode video (VAE) ──────────────────────────────────────
            final_lat = x0_pred.to(device, dtype=model.param_dtype)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vids = model.vae.decode([final_lat])

            if vids and vids[0] is not None:
                vid = vids[0]
                save_video(
                    tensor=vid[None],
                    save_file=str(video_path),
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                total_generated += 1
                t_rollout_end = time.time()
                log_all(
                    f"  rollout {g + 1}/{N}  seed={seed_g}  "
                    f"time={t_rollout_end - t_rollout_start:.1f}s  "
                    f"-> {video_path.name}",
                    rank,
                )
            else:
                log_all(f"  rollout {g + 1}/{N}  FAILED (decode returned None)", rank)

            del noise, x0_pred, final_lat
            torch.cuda.empty_cache()

        # Cleanup this item
        t_item_end = time.time()
        log_all(
            f"  item done in {t_item_end - t_item_start:.1f}s",
            rank,
        )
        del ctx_c, ctx_n, z_img, mask2, img_tensor
        gc.collect()
        torch.cuda.empty_cache()

    # ── Synchronize and finish ───────────────────────────────────────────
    if world_size > 1:
        dist.barrier()

    t_total = time.time() - t_start_all
    log_all(
        f"Done! generated={total_generated}, skipped={total_skipped}, "
        f"total_time={t_total:.1f}s",
        rank,
    )

    if world_size > 1:
        dist.destroy_process_group()

    log_rank(
        f"\n{'=' * 70}\n"
        f"  All rollouts complete.\n"
        f"  Output directory: {out_dir}\n"
        f"{'=' * 70}",
        rank,
    )


if __name__ == "__main__":
    main()
