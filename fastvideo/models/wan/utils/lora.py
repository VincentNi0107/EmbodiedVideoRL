import os
from typing import Dict, Tuple

import torch


def _load_state_dict(file_path: str, device: str = "cpu", dtype: torch.dtype = None) -> Dict[str, torch.Tensor]:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"LoRA file not found: {file_path}")
    if file_path.endswith(".safetensors"):
        try:
            from safetensors import safe_open
        except Exception as exc:
            raise ImportError(
                "safetensors is required to load .safetensors LoRA files. "
                "Please install it in the runtime environment."
            ) from exc
        state_dict = {}
        with safe_open(file_path, framework="pt", device=str(device)) as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if dtype is not None:
                    t = t.to(dtype)
                state_dict[k] = t
        return state_dict
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and len(state_dict) == 1:
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict:
            state_dict = state_dict["module"]
        elif "model_state" in state_dict:
            state_dict = state_dict["model_state"]
    if dtype is not None:
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.to(dtype)
    return state_dict


def _build_name_dict(lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple[str, str]]:
    name_dict = {}
    for key in lora_state_dict:
        if ".lora_up." in key:
            lora_a_key = "lora_down"
            lora_b_key = "lora_up"
        else:
            lora_a_key = "lora_A"
            lora_b_key = "lora_B"
        if lora_b_key not in key:
            continue
        keys = key.split(".")
        if len(keys) > keys.index(lora_b_key) + 2:
            keys.pop(keys.index(lora_b_key) + 1)
        keys.pop(keys.index(lora_b_key))
        if keys[0] == "diffusion_model":
            keys.pop(0)
        keys.pop(-1)
        target_name = ".".join(keys)
        name_dict[target_name] = (key, key.replace(lora_b_key, lora_a_key))
    return name_dict


def _convert_state_dict(lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    name_dict = _build_name_dict(lora_state_dict)
    converted = {}
    for name, (b_key, a_key) in name_dict.items():
        converted[name + ".lora_B.weight"] = lora_state_dict[b_key]
        converted[name + ".lora_A.weight"] = lora_state_dict[a_key]
    return converted


def fuse_lora_to_model(model: torch.nn.Module, lora_path: str, alpha: float = 1.0, device: str = "cpu", dtype: torch.dtype = None) -> int:
    lora_state_dict = _load_state_dict(lora_path, device=device, dtype=dtype)
    lora_state_dict = _convert_state_dict(lora_state_dict)
    lora_layer_names = set([k.replace(".lora_B.weight", "") for k in lora_state_dict if k.endswith(".lora_B.weight")])
    updated = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if name not in lora_layer_names:
                continue
            weight_up = lora_state_dict[name + ".lora_B.weight"].to(device=device, dtype=dtype)
            weight_down = lora_state_dict[name + ".lora_A.weight"].to(device=device, dtype=dtype)
            if weight_up.dim() == 4:
                weight_up = weight_up.squeeze(3).squeeze(2)
                weight_down = weight_down.squeeze(3).squeeze(2)
                weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_lora = alpha * torch.mm(weight_up, weight_down)
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                module.weight.data = module.weight.data.to(device=device, dtype=dtype) + weight_lora
                updated += 1
    return updated
