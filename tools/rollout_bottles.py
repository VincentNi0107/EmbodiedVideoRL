#!/usr/bin/env python3
"""Rollout-only script for put_bottles_dustbin task.

Loads the Wan2.2 TI2V model (with pre-merged LoRA), generates 8 videos per
sample using deterministic ODE sampling (50 steps), and saves videos to disk.
No training, no reward scoring — pure inference.

Usage:
    python rollout_bottles.py [--sample_steps 50] [--num_generations 8] ...
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _import_vidar_modules(vidar_root: str = ""):
    # Try importing wan from fastvideo.models.wan first, fall back to vidar_root
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


# ──────────────────────────────────────────────────────────────────────────────
# Flow-matching helpers
# ──────────────────────────────────────────────────────────────────────────────

def sd3_time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    return (shift * t) / (1.0 + (shift - 1.0) * t)


def get_sigma_schedule(num_steps: int, shift: float, device: torch.device) -> torch.Tensor:
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    return sd3_time_shift(shift, sigmas)


def flow_ode_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    x0_pred = latents - sigma * model_output
    next_sample = latents + dsigma * model_output
    return next_sample, x0_pred


def _expand_timestep(
    mask2: torch.Tensor,
    sigma_val: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    t_val = sigma_val * 1000.0
    ts = (mask2[0][:, ::2, ::2] * t_val).flatten()
    ts = torch.cat([ts, ts.new_ones(seq_len - ts.size(0)) * t_val])
    return ts.unsqueeze(0).to(device)


def _dit_dtype(dit) -> torch.dtype:
    model = dit.module if hasattr(dit, 'module') else dit
    return next(model.parameters()).dtype


# ──────────────────────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_model(args, wan, cfg, device):
    model = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        pt_dir=args.pt_dir,
        device_id=device.index if device.index is not None else 0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        init_on_cpu=False,
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
# ODE rollout
# ──────────────────────────────────────────────────────────────────────────────

def ode_rollout_single(
    dit, noise, z_img, mask2,
    ctx_cond, ctx_null, seq_len,
    sigmas, guide_scale, device,
):
    S = len(sigmas) - 1
    dtype = _dit_dtype(dit)

    latent = ((1.0 - mask2) * z_img + mask2 * noise).to(device).float()
    x0_pred = latent

    for i in tqdm(range(S), desc="  ODE steps", leave=False):
        ts = _expand_timestep(mask2, sigmas[i], seq_len, device)
        lat_in = latent.to(dtype)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v_c = dit([lat_in], t=ts, context=[ctx_cond[0]], seq_len=seq_len)[0]
            v_u = dit([lat_in], t=ts, context=ctx_null, seq_len=seq_len)[0]

        v = (v_u + guide_scale * (v_c - v_u)).float()
        next_lat, x0_pred = flow_ode_step(v, latent, sigmas, i)

        next_lat = (1.0 - mask2) * z_img.float() + mask2 * next_lat
        latent = next_lat.detach()

        del v_c, v_u, v, lat_in

    return x0_pred


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Rollout-only: generate videos without training")

    # model / vidar
    p.add_argument("--vidar_root", type=str, default="",
                    help="Path to vidar repo (optional if wan/ exists in project root)")
    p.add_argument("--task", type=str, default="ti2v-5B")
    p.add_argument("--size", type=str, default="640*736")
    p.add_argument("--frame_num", type=int, default=121)
    p.add_argument("--ckpt_dir", type=str, default="Wan2.2-TI2V-5B")
    p.add_argument("--pt_dir", type=str, default=None)
    p.add_argument("--convert_model_dtype", action="store_true", default=False)

    # data
    p.add_argument("--dataset_json", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=-1,
                    help="Limit number of samples to process (-1 = all)")
    p.add_argument("--output_dir", type=str, required=True)

    # sampling / ODE
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--sample_shift", type=float, default=5.0)
    p.add_argument("--sample_guide_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neg_prompt", type=str, default="")

    # rollout
    p.add_argument("--num_generations", type=int, default=8,
                    help="Number of videos to generate per sample")
    p.add_argument("--filter_stem", type=str, default=None,
                    help="Only process samples whose filename_stem contains this substring")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # ── vidar imports ─────────────────────────────────────────────
    (wan, WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS,
     save_video, masks_like, best_output_size) = _import_vidar_modules(args.vidar_root)

    cfg = WAN_CONFIGS[args.task]
    if not args.neg_prompt:
        args.neg_prompt = cfg.sample_neg_prompt
    max_area = MAX_AREA_CONFIGS[args.size]

    # ── output dir ────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    vid_dir = out_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    # ── load dataset ──────────────────────────────────────────────
    ds_path = Path(args.dataset_json).resolve()
    with ds_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if 0 < args.max_samples < len(raw):
        raw = raw[:args.max_samples]
    print(f"Dataset: {len(raw)} samples from {ds_path}")

    # ── build model ───────────────────────────────────────────────
    print(f"Building Wan2.2 TI2V model ...")
    model = _build_model(args, wan, cfg, device)
    dit = model.model

    # Move T5 to GPU permanently (single-GPU inference, plenty of VRAM)
    print("  Moving T5 encoder to GPU ...")
    model.text_encoder.model.to(device)
    model.t5_cpu = False
    torch.cuda.empty_cache()

    # ── sigma schedule ────────────────────────────────────────────
    sigmas = get_sigma_schedule(args.sample_steps, args.sample_shift, device)
    print(f"Sigma schedule: {args.sample_steps} steps, shift={args.sample_shift}")

    N = args.num_generations
    print(f"Will generate {N} videos per sample, {len(raw)} samples total")
    print(f"Output: {out_dir}")
    print()

    # ── rollout loop ──────────────────────────────────────────────
    dit.eval()

    for sample_idx, row in enumerate(raw):
        prompt = row["prompt"]
        media_path = row.get("media_path", "")
        fname_stem = row.get("filename_stem", f"sample_{sample_idx:06d}")

        # Filter by stem if specified
        if args.filter_stem and args.filter_stem not in fname_stem:
            continue

        # Resolve media path
        mp = Path(media_path)
        if not mp.is_file():
            mp = (ds_path.parent / media_path).resolve()
        if not mp.is_file():
            print(f"  [SKIP] Image not found: {media_path}")
            continue

        print(f"[{sample_idx + 1}/{len(raw)}] {fname_stem}")

        # ── encode text ──────────────────────────────────────
        ctx_c = model.text_encoder([prompt], device)
        ctx_n = model.text_encoder([args.neg_prompt], device)

        # ── encode image (VAE) ───────────────────────────────
        img_pil = Image.open(str(mp)).convert("RGB")
        img_tensor, oh, ow = _prepare_image(model, img_pil, max_area, best_output_size)
        z_img = model.vae.encode([img_tensor])[0]

        F = args.frame_num
        seq_len = _compute_seq_len(F, oh, ow, model.vae_stride, model.patch_size)
        latent_shape = (
            model.vae.model.z_dim,
            (F - 1) // model.vae_stride[0] + 1,
            oh // model.vae_stride[1],
            ow // model.vae_stride[2],
        )

        # ── mask (first frame = image, rest = noise) ─────────
        noise_tmp = torch.randn(latent_shape, device=device, dtype=torch.float32)
        _, mask2_list = masks_like([noise_tmp], zero=True)
        mask2 = mask2_list[0].to(device)
        del noise_tmp

        # ── create per-sample output dir ─────────────────────
        sample_vid_dir = vid_dir / fname_stem
        sample_vid_dir.mkdir(parents=True, exist_ok=True)

        # ── generate N videos ────────────────────────────────
        base_seed = args.seed + sample_idx * N
        for g in range(N):
            seed_g = base_seed + g
            vp = str(sample_vid_dir / f"{fname_stem}_g{g:03d}_s{seed_g}.mp4")

            # Skip if video already exists
            if Path(vp).is_file():
                print(f"  [SKIP] {g + 1}/{N} already exists: {vp}")
                continue

            gen = torch.Generator(device=device).manual_seed(seed_g)
            noise = torch.randn(
                latent_shape, dtype=torch.float32,
                generator=gen, device=device,
            )

            print(f"  Generating video {g + 1}/{N} (seed={seed_g}) ...")
            x0_pred = ode_rollout_single(
                dit, noise, z_img, mask2,
                ctx_c, ctx_n, seq_len,
                sigmas, args.sample_guide_scale,
                device,
            )

            # Decode latent → video
            final_lat = x0_pred.to(device, dtype=model.param_dtype)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vids = model.vae.decode([final_lat])

            if vids and vids[0] is not None:
                vid = vids[0]
                save_video(
                    tensor=vid[None], save_file=vp,
                    fps=cfg.sample_fps, nrow=1,
                    normalize=True, value_range=(-1, 1),
                )
                print(f"    Saved: {vp}")
            else:
                print(f"    [WARN] VAE decode returned None for g={g}")

            del noise, final_lat, x0_pred
            torch.cuda.empty_cache()

        # Clean up per-sample tensors
        del ctx_c, ctx_n, z_img, mask2
        torch.cuda.empty_cache()
        print()

    print("Done! All videos saved.")


if __name__ == "__main__":
    main()
