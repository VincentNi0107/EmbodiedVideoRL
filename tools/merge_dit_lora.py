#!/usr/bin/env python3
"""
Merge vidar DiT weights + LoRA into a single checkpoint.

Usage:
    python merge_dit_lora.py \
        --vidar_pt /home/omz1504/code/vidar/vidar_ckpts/vidar.pt \
        --lora_path /home/omz1504/code/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_robotwin_all_121_lora/epoch-3.safetensors \
        --lora_alpha 1.0 \
        --output /home/omz1504/code/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_robotwin_all_121_lora/merged_vidar_lora.pt

After merging, update the training script to use:
    --pt_dir merged_vidar_lora.pt
    (remove --lora_path)
"""
import argparse
import os
import sys


def load_state_dict(file_path, device="cpu"):
    """Load state dict from .pt/.pth or .safetensors file."""
    if file_path.endswith(".safetensors"):
        from safetensors import safe_open
        state_dict = {}
        with safe_open(file_path, framework="pt", device=device) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        return state_dict
    state_dict = __import__("torch").load(file_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and len(state_dict) == 1:
        for wrapper_key in ("state_dict", "module", "model_state"):
            if wrapper_key in state_dict:
                state_dict = state_dict[wrapper_key]
                break
    return state_dict


def convert_lora_keys(lora_sd):
    """Normalize LoRA keys to use lora_A/lora_B naming convention.

    Handles both lora_A/lora_B and lora_down/lora_up styles,
    and strips 'diffusion_model.' prefix if present.
    """
    converted = {}
    for key in lora_sd:
        if ".lora_up." in key:
            lora_a_name, lora_b_name = "lora_down", "lora_up"
        else:
            lora_a_name, lora_b_name = "lora_A", "lora_B"

        if lora_b_name not in key:
            continue

        keys = key.split(".")
        # Remove extra sub-key after lora_B (e.g. lora_B.default.weight -> lora_B.weight)
        if len(keys) > keys.index(lora_b_name) + 2:
            keys.pop(keys.index(lora_b_name) + 1)
        keys.pop(keys.index(lora_b_name))
        if keys[0] == "diffusion_model":
            keys.pop(0)
        keys.pop(-1)  # remove "weight"
        target_name = ".".join(keys)

        converted[target_name + ".lora_B.weight"] = lora_sd[key]
        a_key = key.replace(lora_b_name, lora_a_name)
        converted[target_name + ".lora_A.weight"] = lora_sd[a_key]
    return converted


def main():
    import torch

    parser = argparse.ArgumentParser(description="Merge DiT weights + LoRA into a single checkpoint")
    parser.add_argument("--vidar_pt", type=str, required=True,
                        help="Path to vidar DiT checkpoint (e.g. vidar_ckpts/vidar.pt)")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRA checkpoint (.safetensors or .pt)")
    parser.add_argument("--lora_alpha", type=float, default=1.0,
                        help="LoRA scaling factor (default: 1.0)")
    parser.add_argument("--output", type=str, default="merged_vidar_lora.pt",
                        help="Output path for merged checkpoint")
    parser.add_argument("--base_ckpt_dir", type=str, default=None,
                        help="(Optional) Wan2.2-TI2V-5B checkpoint_dir. Only needed if vidar.pt "
                             "is a partial checkpoint and LoRA targets keys not in vidar.pt.")
    args = parser.parse_args()

    # --- Step 1: Load vidar DiT state dict ---
    print(f"Loading vidar DiT weights from: {args.vidar_pt}")
    dit_sd = load_state_dict(args.vidar_pt, device="cpu")
    print(f"  -> {len(dit_sd)} keys loaded")

    # --- Step 2: Optionally load base checkpoint DiT and merge ---
    if args.base_ckpt_dir is not None:
        print(f"Loading base DiT from checkpoint_dir: {args.base_ckpt_dir}")
        # WanModel.from_pretrained saves in diffusers format
        from diffusers.models.modeling_utils import load_state_dict as load_diffusers_sd
        base_files = [
            os.path.join(args.base_ckpt_dir, "diffusion_pytorch_model.safetensors"),
            os.path.join(args.base_ckpt_dir, "diffusion_pytorch_model.bin"),
        ]
        base_sd = None
        for f in base_files:
            if os.path.exists(f):
                base_sd = load_state_dict(f, device="cpu")
                print(f"  -> {len(base_sd)} keys from {f}")
                break
        if base_sd is not None:
            # vidar.pt overrides base
            base_sd.update(dit_sd)
            dit_sd = base_sd
            print(f"  -> Merged: {len(dit_sd)} total keys")

    # --- Step 3: Load and convert LoRA ---
    print(f"Loading LoRA from: {args.lora_path}")
    lora_raw = load_state_dict(args.lora_path, device="cpu")
    lora_sd = convert_lora_keys(lora_raw)
    lora_layers = sorted(set(k.replace(".lora_B.weight", "") for k in lora_sd if k.endswith(".lora_B.weight")))
    print(f"  -> {len(lora_layers)} LoRA layers found")

    # --- Step 4: Fuse LoRA into DiT state dict ---
    alpha = args.lora_alpha
    fused = 0
    missing = []
    for name in lora_layers:
        weight_key = name + ".weight"
        up = lora_sd[name + ".lora_B.weight"]
        down = lora_sd[name + ".lora_A.weight"]

        if weight_key not in dit_sd:
            missing.append(weight_key)
            continue

        base_weight = dit_sd[weight_key]
        if up.dim() == 4:
            delta = alpha * torch.mm(
                up.squeeze(3).squeeze(2).float(),
                down.squeeze(3).squeeze(2).float()
            ).unsqueeze(2).unsqueeze(3)
        else:
            delta = alpha * torch.mm(up.float(), down.float())

        dit_sd[weight_key] = (base_weight.float() + delta).to(base_weight.dtype)
        fused += 1

    print(f"  -> Fused {fused}/{len(lora_layers)} LoRA layers (alpha={alpha})")
    if missing:
        print(f"  WARNING: {len(missing)} LoRA target keys not found in DiT state dict:")
        for m in missing[:10]:
            print(f"    - {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
        print("  Consider providing --base_ckpt_dir to include base model weights.")

    # --- Step 5: Save merged checkpoint ---
    print(f"Saving merged checkpoint to: {args.output}")
    torch.save(dit_sd, args.output)
    file_size = os.path.getsize(args.output) / (1024 ** 3)
    print(f"  -> Done! File size: {file_size:.2f} GB")
    print()
    print("Usage after merging:")
    print(f"  --ckpt_dir <Wan2.2-TI2V-5B>   (VAE + T5 only)")
    print(f"  --pt_dir {args.output}          (merged DiT)")
    print(f"  (no --lora_path needed)")


if __name__ == "__main__":
    main()
