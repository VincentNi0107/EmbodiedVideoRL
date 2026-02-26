#!/usr/bin/env python3
"""Inference script for Wan2.2 TI2V models fine-tuned with DiffusionNFT.

Loads a base Wan2.2 TI2V model, applies NFT LoRA checkpoint (peft format),
and generates videos from text prompt + first-frame image pairs.

Usage:
    python infer_nft.py \
        --vidar_root /home/omz1504/code/vidar \
        --ckpt_dir /home/omz1504/code/vidar/Wan2.2-TI2V-5B \
        --pt_dir /path/to/base_dit_weights.pt \
        --nft_lora_path data/outputs/nft_put_object_cabinet/checkpoints/lora_final.pt \
        --lora_rank 64 --lora_alpha 64 \
        --prompt "A robot arm picks up the object and places it in the cabinet." \
        --image /path/to/first_frame.png \
        --output_dir ./infer_output

    # Or batch mode from a dataset JSON:
    python infer_nft.py \
        --vidar_root /home/omz1504/code/vidar \
        --ckpt_dir /home/omz1504/code/vidar/Wan2.2-TI2V-5B \
        --pt_dir /path/to/base_dit_weights.pt \
        --nft_lora_path data/outputs/nft_put_object_cabinet/checkpoints/lora_final.pt \
        --dataset_json /path/to/test.json \
        --output_dir ./infer_output
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _import_vidar(vidar_root: str = ""):
    """Import wan module — tries fastvideo.models.wan first, falls back to vidar_root."""
    try:
        import fastvideo.models.wan as wan
        from fastvideo.models.wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
        from fastvideo.models.wan.utils.utils import save_video
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
        from wan.utils.utils import save_video

    return wan, WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, save_video


def convert_nft_lora_keys(nft_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert peft-format NFT LoRA keys to vidar-compatible format.

    NFT training saves keys like:
        base_model.model.blocks.0.self_attn.q.lora_A.default.weight

    vidar's fuse_lora_to_model expects:
        blocks.0.self_attn.q.lora_A.weight

    This function:
      1. Filters to only "default" adapter keys (ignoring "old" adapter)
      2. Strips 'base_model.model.' prefix
      3. Removes adapter name ('default') from key path
    """
    converted = {}
    for key, value in nft_state_dict.items():
        # Only keep "default" adapter (not "old")
        if ".old." in key:
            continue

        new_key = key
        # Strip peft wrapper prefix
        for prefix in ("base_model.model.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break

        # Remove adapter name: lora_A.default.weight -> lora_A.weight
        new_key = new_key.replace(".lora_A.default.", ".lora_A.")
        new_key = new_key.replace(".lora_B.default.", ".lora_B.")

        converted[new_key] = value

    n_a = sum(1 for k in converted if "lora_A" in k)
    n_b = sum(1 for k in converted if "lora_B" in k)
    logger.info(f"Converted NFT LoRA keys: {len(converted)} total ({n_a} lora_A, {n_b} lora_B)")
    return converted


def load_model(
    vidar_root: str = "",
    ckpt_dir: str = "",
    pt_dir: Optional[str] = None,
    nft_lora_path: Optional[str] = None,
    lora_alpha: float = 1.0,
    device_id: int = 0,
    t5_cpu: bool = False,
    offload_model: bool = False,
    convert_model_dtype: bool = True,
):
    """Load WanTI2V model and apply NFT LoRA checkpoint.

    Returns (model, cfg, wan_module, save_video_fn).
    """
    wan, WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, save_video = _import_vidar(vidar_root)

    cfg = WAN_CONFIGS["ti2v-5B"]

    logger.info(f"Building WanTI2V model (ckpt_dir={ckpt_dir})")
    model = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        pt_dir=pt_dir,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=t5_cpu,
        init_on_cpu=offload_model,
        convert_model_dtype=convert_model_dtype,
    )

    if nft_lora_path:
        logger.info(f"Loading NFT LoRA checkpoint: {nft_lora_path}")
        raw_state = torch.load(nft_lora_path, map_location="cpu", weights_only=True)
        converted_state = convert_nft_lora_keys(raw_state)

        # Save converted state to a temp file and use vidar's load_lora
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            torch.save(converted_state, tmp.name)
            tmp_path = tmp.name
        try:
            updated = model.load_lora(tmp_path, alpha=lora_alpha)
            logger.info(f"NFT LoRA fused into model: {updated} layers updated")
        finally:
            os.unlink(tmp_path)

        # fuse_lora_to_model operates on CPU, which moves fused weight tensors
        # off GPU. Move the entire DiT back to the correct device.
        device = torch.device(f"cuda:{device_id}")
        model.model.to(device)
        logger.info(f"DiT model moved to {device} after LoRA fusion")

    return model, cfg, wan, save_video, SIZE_CONFIGS, MAX_AREA_CONFIGS


def generate_video(
    model,
    cfg,
    save_video_fn,
    SIZE_CONFIGS,
    MAX_AREA_CONFIGS,
    prompt: str,
    image_path: str,
    output_path: str,
    size: str = "640*736",
    frame_num: int = 121,
    shift: float = 5.0,
    sampling_steps: int = 20,
    guide_scale: float = 5.0,
    seed: int = -1,
    offload_model: bool = False,
):
    """Generate a single video from prompt + first-frame image."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(SIZE_CONFIGS[size])

    logger.info(f"Generating video: prompt='{prompt[:80]}...', seed={seed}, steps={sampling_steps}")
    video_tensor = model.generate(
        input_prompt=prompt,
        img=img,
        size=SIZE_CONFIGS[size],
        max_area=MAX_AREA_CONFIGS[size],
        frame_num=frame_num,
        shift=shift,
        sample_solver="unipc",
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        seed=seed,
        offload_model=offload_model,
    )

    # Save video: (C, T, H, W) -> add batch dim for save_video
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_video_fn(
        tensor=video_tensor[None],
        save_file=output_path,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logger.info(f"Video saved to: {output_path}")
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="Wan2.2 TI2V NFT Inference")

    # Model paths
    p.add_argument("--vidar_root", type=str, default="",
                   help="Path to vidar repo (optional if wan/ exists in project root)")
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Path to Wan2.2-TI2V-5B checkpoint directory")
    p.add_argument("--pt_dir", type=str, default=None,
                   help="Path to pre-trained DiT weights (.pt)")
    p.add_argument("--nft_lora_path", type=str, default=None,
                   help="Path to NFT LoRA checkpoint (e.g., lora_final.pt)")
    p.add_argument("--lora_alpha", type=float, default=1.0,
                   help="LoRA scaling factor")

    # Single inference mode
    p.add_argument("--prompt", type=str, default=None,
                   help="Text prompt for video generation")
    p.add_argument("--image", type=str, default=None,
                   help="Path to first-frame image")

    # Batch inference mode
    p.add_argument("--dataset_json", type=str, default=None,
                   help="Path to dataset JSON for batch inference")
    p.add_argument("--max_samples", type=int, default=-1,
                   help="Max number of samples to process (-1 = all)")

    # Generation settings
    p.add_argument("--output_dir", type=str, default="./infer_output",
                   help="Output directory for generated videos")
    p.add_argument("--size", type=str, default="640*736",
                   help="Video resolution (width*height)")
    p.add_argument("--frame_num", type=int, default=121,
                   help="Number of frames to generate")
    p.add_argument("--sampling_steps", type=int, default=20,
                   help="Number of denoising steps")
    p.add_argument("--shift", type=float, default=5.0,
                   help="Noise schedule shift parameter")
    p.add_argument("--guide_scale", type=float, default=5.0,
                   help="Classifier-free guidance scale")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (-1 for random)")
    p.add_argument("--num_videos", type=int, default=1,
                   help="Number of videos per prompt")

    # Device settings
    p.add_argument("--device_id", type=int, default=0,
                   help="CUDA device ID")
    p.add_argument("--t5_cpu", action="store_true", default=False,
                   help="Place T5 on CPU to save VRAM")
    p.add_argument("--offload_model", action="store_true", default=False,
                   help="Offload models to CPU during generation")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, cfg, wan_mod, save_video_fn, SIZE_CONFIGS, MAX_AREA_CONFIGS = load_model(
        vidar_root=args.vidar_root,
        ckpt_dir=args.ckpt_dir,
        pt_dir=args.pt_dir,
        nft_lora_path=args.nft_lora_path,
        lora_alpha=args.lora_alpha,
        device_id=args.device_id,
        t5_cpu=args.t5_cpu,
        offload_model=args.offload_model,
    )

    # Build sample list
    samples = []
    if args.dataset_json:
        dataset_path = Path(args.dataset_json).resolve()
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if 0 < args.max_samples < len(raw):
            raw = raw[:args.max_samples]
        for i, row in enumerate(raw):
            mp = Path(row.get("media_path", ""))
            if not mp.is_file():
                mp = (dataset_path.parent / row.get("media_path", "")).resolve()
            samples.append({
                "prompt": row["prompt"],
                "image": str(mp),
                "stem": row.get("filename_stem", f"sample_{i:06d}"),
            })
    elif args.prompt and args.image:
        samples.append({
            "prompt": args.prompt,
            "image": args.image,
            "stem": Path(args.image).stem,
        })
    else:
        logger.error("Provide either --dataset_json or both --prompt and --image")
        sys.exit(1)

    logger.info(f"Processing {len(samples)} sample(s), {args.num_videos} video(s) each")

    for idx, sample in enumerate(samples):
        for vid_i in range(args.num_videos):
            seed = args.seed + idx * args.num_videos + vid_i
            out_name = f"{sample['stem']}_v{vid_i:02d}_s{seed}.mp4"
            out_path = os.path.join(args.output_dir, out_name)

            try:
                generate_video(
                    model=model,
                    cfg=cfg,
                    save_video_fn=save_video_fn,
                    SIZE_CONFIGS=SIZE_CONFIGS,
                    MAX_AREA_CONFIGS=MAX_AREA_CONFIGS,
                    prompt=sample["prompt"],
                    image_path=sample["image"],
                    output_path=out_path,
                    size=args.size,
                    frame_num=args.frame_num,
                    shift=args.shift,
                    sampling_steps=args.sampling_steps,
                    guide_scale=args.guide_scale,
                    seed=seed,
                    offload_model=args.offload_model,
                )
            except Exception as e:
                logger.error(f"Failed to generate {out_name}: {e}", exc_info=True)

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
