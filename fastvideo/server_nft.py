#!/usr/bin/env python3
"""FastAPI server for Wan2.2 TI2V inference with DiffusionNFT LoRA.

Serves the NFT fine-tuned model as an HTTP API, compatible with the
vidar server protocol. Supports both video generation and IDM inference.

Launch via:
    uvicorn fastvideo.server_nft:api --host 0.0.0.0 --port 25400 --workers 1

Or use the provided scripts/inference/start_server_nft.sh.

Environment variables:
    VIDAR_ROOT      - path to vidar repo (default: /home/omz1504/code/vidar)
    CKPT_DIR        - path to Wan2.2-TI2V-5B checkpoint dir
    PT_DIR          - path to base DiT weights (optional)
    NFT_LORA_PATH   - path to NFT LoRA checkpoint (lora_final.pt)
    LORA_ALPHA      - LoRA scaling factor (default: 1.0)
    DEVICE_ID       - CUDA device ID (default: 0)
    T5_CPU          - place T5 on CPU (default: false)
    FRAME_NUM       - number of frames (default: 121)
    SAMPLING_STEPS  - denoising steps (default: 20)
    GUIDE_SCALE     - CFG scale (default: 5.0)
    SHIFT           - noise schedule shift (default: 5.0)
    SIZE            - resolution string (default: 640*736)
    SAVE_VIDEO      - save generated videos to disk (default: false)
    SAVE_VIDEO_DIR  - directory for saved videos (default: /tmp/nft_server_videos)
"""

import base64
import io
import json as _json
import logging
import os
import time
import uuid
from pathlib import Path

import torch
import torchvision
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Request / Response schemas ────────────────────────────────────────────────

class GenerateRequest(PydanticModel):
    """Request schema compatible with vidar server protocol."""
    prompt: str
    imgs: list  # list of base64-encoded images
    num_conditional_frames: int = 1
    num_new_frames: int = 120
    seed: int = 42
    num_sampling_step: int = 20
    guide_scale: float = 5.0
    shift: float = 5.0
    return_imgs: bool = False
    return_video: bool = False
    password: str = ""


class SimpleGenerateRequest(PydanticModel):
    """Simplified request for direct video generation."""
    prompt: str
    image_base64: str  # base64-encoded first-frame image
    frame_num: int = 121
    seed: int = 42
    sampling_steps: int = 20
    guide_scale: float = 5.0
    shift: float = 5.0


class PlanRequest(PydanticModel):
    """Generate a video plan for closed-loop IDM execution."""
    prompt: str
    imgs: list  # list of base64-encoded images (last one used as first frame)
    num_new_frames: int = 120
    seed: int = 42
    num_sampling_step: int = 20
    guide_scale: float = 5.0
    shift: float = 5.0
    password: str = ""
    return_video: bool = False


class IDMStepRequest(PydanticModel):
    """Per-step closed-loop IDM inference."""
    session_id: str
    frame_index: int
    observation_b64: str  # base64-encoded JPEG of current SAPIEN observation


# ── Globals ───────────────────────────────────────────────────────────────────

api = FastAPI(title="DanceGRPO NFT Inference Server")
wan_model = None
wan_cfg = None
save_video_fn = None
SIZE_CONFIGS = None
MAX_AREA_CONFIGS = None
idm = None
idm_device = None
processor = None
mask_processor = None
gc_idm = None  # Goal-conditioned IDM (GoalConditionedMask)

# Session store: maps session_id -> {"goal_frames": Tensor(T,C,H,W), "created": float}
_video_store: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("1", "true", "yes", "y", "on")


def _decode_image(b64_str: str):
    """Decode a base64 image string to PIL Image."""
    from PIL import Image
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _tensor_to_video_bytes(video_tensor: torch.Tensor, fps: int = 16) -> bytes:
    """Convert a (C, T, H, W) video tensor to MP4 bytes."""
    video = video_tensor.clamp(-1, 1).add(1).mul(127.5).to(torch.uint8)
    video = video.permute(1, 2, 3, 0).cpu()  # (T, H, W, C)
    buf = io.BytesIO()
    torchvision.io.write_video(buf, video, fps=fps, video_codec="libx264")
    buf.seek(0)
    return buf.read()


def _save_video_to_disk(video_tensor: torch.Tensor, request, out_dir: str):
    """Save generated video tensor to disk."""
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    prompt_short = request.prompt[:8].replace(" ", "_")
    out_path = os.path.join(out_dir, f"nft_{ts}_s{request.seed}_{prompt_short}.mp4")

    save_video_fn(
        tensor=video_tensor[None],
        save_file=out_path,
        fps=wan_cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logger.info(f"Saved video to {out_path}")
    return out_path


def batch_tensor_to_jpeg_b64(tensor: torch.Tensor):
    """Convert (B, C, H, W) uint8 tensor to list of base64 JPEG strings."""
    tensor = (tensor * 255).to(torch.uint8).cpu()
    results = []
    for i in range(tensor.shape[0]):
        jpeg_tensor = torchvision.io.encode_jpeg(tensor[i])
        results.append(base64.b64encode(jpeg_tensor.numpy().tobytes()).decode("utf-8"))
    return results


# ── Model initialization ─────────────────────────────────────────────────────

def init():
    """Load model from environment variables on server startup."""
    global wan_model, wan_cfg, save_video_fn, SIZE_CONFIGS, MAX_AREA_CONFIGS
    global idm, idm_device, processor, mask_processor

    from fastvideo.infer_nft import load_model

    ckpt_dir = os.getenv("CKPT_DIR", "ckpts/Wan2.2-TI2V-5B")
    pt_dir = os.getenv("PT_DIR", None)
    nft_lora_path = os.getenv("NFT_LORA_PATH", "")
    lora_alpha = float(os.getenv("LORA_ALPHA", "1.0"))
    device_id = int(os.getenv("DEVICE_ID", "0"))
    t5_cpu = _env_bool("T5_CPU", False)
    offload_model = _env_bool("OFFLOAD_MODEL", False)

    logger.info(f"Initializing NFT server (PID={os.getpid()})")
    logger.info(f"  CKPT_DIR={ckpt_dir}")
    logger.info(f"  PT_DIR={pt_dir}")
    logger.info(f"  NFT_LORA_PATH={nft_lora_path}")
    logger.info(f"  LORA_ALPHA={lora_alpha}")
    logger.info(f"  DEVICE_ID={device_id}")

    wan_model, wan_cfg, wan_mod, save_video_fn, SIZE_CONFIGS, MAX_AREA_CONFIGS = load_model(
        ckpt_dir=ckpt_dir,
        pt_dir=pt_dir,
        nft_lora_path=nft_lora_path if nft_lora_path else None,
        lora_alpha=lora_alpha,
        device_id=device_id,
        t5_cpu=t5_cpu,
        offload_model=offload_model,
    )

    # Optionally load IDM model (for vidar-compatible endpoint)
    idm_path = os.getenv("IDM_PATH", "")
    if idm_path and os.path.isfile(idm_path):
        from server.idm import IDM

        idm_device = torch.device(f"cuda:{device_id}")
        if _env_bool("IDM_CPU"):
            idm_device = torch.device("cpu")

        if idm_path.endswith("_out.pt"):
            output_dim = int(idm_path.split("_out.pt")[0].split("_")[-1])
            idm = IDM(model_name="mask", output_dim=output_dim).to(idm_device)
        else:
            idm = IDM(model_name="mask", output_dim=14).to(idm_device)

        loaded_dict = torch.load(idm_path, map_location=idm_device, weights_only=False)
        idm.load_state_dict(loaded_dict["model_state_dict"])
        idm.eval()
        logger.info(f"IDM loaded from {idm_path}")

        processor = torchvision.transforms.Compose([
            torchvision.transforms.Resize((518, 518)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        mask_processor = torchvision.transforms.Resize((736, 640))
    else:
        logger.info("IDM not loaded (IDM_PATH not set or file missing)")

    # Optionally load goal-conditioned IDM (for closed-loop endpoint)
    gc_idm_path = os.getenv("GC_IDM_PATH", "")
    if gc_idm_path and os.path.isfile(gc_idm_path):
        from server.idm import IDM as IDMModule

        if idm_device is None:
            idm_device = torch.device(f"cuda:{device_id}")
            if _env_bool("IDM_CPU"):
                idm_device = torch.device("cpu")

        gc_idm = IDMModule(model_name="goal_conditioned_mask", output_dim=14).to(idm_device)
        gc_loaded = torch.load(gc_idm_path, map_location=idm_device, weights_only=False)
        gc_idm.load_state_dict(gc_loaded["model_state_dict"])
        gc_idm.eval()
        logger.info(f"Goal-conditioned IDM loaded from {gc_idm_path}")

        if processor is None:
            processor = torchvision.transforms.Compose([
                torchvision.transforms.Resize((518, 518)),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    else:
        logger.info("Goal-conditioned IDM not loaded (GC_IDM_PATH not set or file missing)")

    logger.info("Server initialization complete.")


def idm_pred(imgs: torch.Tensor, return_imgs: bool = False):
    """Run IDM prediction if available."""
    import json as _json
    if idm is None:
        return {"actions": "[]"}

    imgs_dev = imgs.to(idm_device)
    with torch.no_grad():
        actions, masks = idm(processor(imgs_dev), return_mask=return_imgs)
    pred = {"actions": _json.dumps(actions.cpu().numpy().tolist())}
    if return_imgs:
        pred["imgs"] = batch_tensor_to_jpeg_b64(imgs)
        masks = mask_processor(masks)
        pred["masks"] = batch_tensor_to_jpeg_b64(torch.where(masks >= 0.5, imgs, 1))
    return pred


# ── Initialize on import ─────────────────────────────────────────────────────
init()


# ── API endpoints ─────────────────────────────────────────────────────────────

@api.get("/")
async def health():
    """Health check endpoint."""
    return {"message": "DanceGRPO NFT Inference Server is running."}


@api.get("/info")
async def info():
    """Return server configuration info."""
    return {
        "model": "Wan2.2-TI2V-5B + NFT LoRA",
        "nft_lora_path": os.getenv("NFT_LORA_PATH", ""),
        "lora_alpha": float(os.getenv("LORA_ALPHA", "1.0")),
        "device": os.getenv("DEVICE_ID", "0"),
    }


@api.post("/")
async def predict_vidar(request: GenerateRequest):
    """vidar-compatible prediction endpoint.

    Accepts the same request format as vidar/server/stand_worker.py.
    Returns IDM actions (and optionally images/masks).
    """
    import hashlib
    expected_hash = "d43e76d9cad30d53805246aa72cc25b04ce2cbe6c7086b53ac6fb5028c48d307"
    if hashlib.sha256(request.password.encode("utf-8")).hexdigest() != expected_hash:
        return {}

    logger.info(f"[vidar] prompt='{request.prompt[:60]}', seed={request.seed}, "
                f"frames={request.num_conditional_frames}+{request.num_new_frames}")

    size_key = os.getenv("SIZE", "640*736")
    frame_num = request.num_conditional_frames + request.num_new_frames

    img = _decode_image(request.imgs[-1])
    img = img.resize(SIZE_CONFIGS[size_key])

    # offload_model=True (same as original stand_worker.py default):
    # after generation, DiT/VAE are offloaded to CPU to free VRAM for IDM.
    video_tensor = wan_model.generate(
        input_prompt=request.prompt,
        img=img,
        size=SIZE_CONFIGS[size_key],
        max_area=MAX_AREA_CONFIGS[size_key],
        frame_num=frame_num,
        shift=request.shift,
        sample_solver="unipc",
        sampling_steps=request.num_sampling_step,
        guide_scale=request.guide_scale,
        seed=request.seed,
        offload_model=True,
    )

    if _env_bool("SAVE_VIDEO"):
        _save_video_to_disk(video_tensor, request,
                            os.getenv("SAVE_VIDEO_DIR", "/tmp/nft_server_videos"))

    # Process for IDM: (C, T, H, W) -> frames grid -> IDM
    imgs_for_idm = video_tensor[None].clamp(-1, 1)
    imgs_for_idm = torch.stack([
        torchvision.utils.make_grid(u, nrow=8, normalize=True, value_range=(-1, 1))
        for u in imgs_for_idm.unbind(2)
    ], dim=1).permute(1, 0, 2, 3)  # (T, C, H, W)

    pred = idm_pred(imgs_for_idm, return_imgs=request.return_imgs)

    if request.return_video:
        video_b64 = base64.b64encode(
            _tensor_to_video_bytes(video_tensor, fps=wan_cfg.sample_fps)
        ).decode("utf-8")
        pred["video_base64"] = video_b64

    return pred


@api.post("/generate")
async def generate(request: SimpleGenerateRequest):
    """Simple video generation endpoint.

    Returns the generated video as base64-encoded MP4.
    """
    logger.info(f"[generate] prompt='{request.prompt[:60]}', seed={request.seed}, "
                f"frames={request.frame_num}, steps={request.sampling_steps}")

    size_key = os.getenv("SIZE", "640*736")
    img = _decode_image(request.image_base64)
    img = img.resize(SIZE_CONFIGS[size_key])

    video_tensor = wan_model.generate(
        input_prompt=request.prompt,
        img=img,
        size=SIZE_CONFIGS[size_key],
        max_area=MAX_AREA_CONFIGS[size_key],
        frame_num=request.frame_num,
        shift=request.shift,
        sample_solver="unipc",
        sampling_steps=request.sampling_steps,
        guide_scale=request.guide_scale,
        seed=request.seed,
        offload_model=True,
    )

    if _env_bool("SAVE_VIDEO"):
        _save_video_to_disk(video_tensor, request,
                            os.getenv("SAVE_VIDEO_DIR", "/tmp/nft_server_videos"))

    video_bytes = _tensor_to_video_bytes(video_tensor, fps=wan_cfg.sample_fps)
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")

    return {
        "video_base64": video_b64,
        "seed": request.seed,
        "frame_num": request.frame_num,
    }


# ── Closed-loop IDM endpoints ───────────────────────────────────────────────


def _video_tensor_to_goal_frames(video_tensor: torch.Tensor) -> torch.Tensor:
    """Convert (C, T, H, W) video tensor to (T, C, H, W) goal frames for IDM.

    Produces the same 8-column grid + normalise transform used by the original
    open-loop ``idm_pred``, so that the goal-conditioned IDM sees the same
    visual format.
    """
    imgs = video_tensor[None].clamp(-1, 1)
    frames = torch.stack([
        torchvision.utils.make_grid(u, nrow=8, normalize=True, value_range=(-1, 1))
        for u in imgs.unbind(2)
    ], dim=1).permute(1, 0, 2, 3)  # (T, C, H, W)
    return frames


@api.post("/generate_plan")
async def generate_plan(request: PlanRequest):
    """Generate a video plan and cache goal frames for closed-loop IDM.

    Returns a session_id that the client uses in subsequent /idm_step calls.
    """
    import hashlib
    expected_hash = "d43e76d9cad30d53805246aa72cc25b04ce2cbe6c7086b53ac6fb5028c48d307"
    if hashlib.sha256(request.password.encode("utf-8")).hexdigest() != expected_hash:
        return {}

    frame_num = 1 + request.num_new_frames
    logger.info(f"[generate_plan] prompt='{request.prompt[:60]}', seed={request.seed}, "
                f"frames={frame_num}")

    size_key = os.getenv("SIZE", "640*736")
    img = _decode_image(request.imgs[-1])
    img = img.resize(SIZE_CONFIGS[size_key])

    video_tensor = wan_model.generate(
        input_prompt=request.prompt,
        img=img,
        size=SIZE_CONFIGS[size_key],
        max_area=MAX_AREA_CONFIGS[size_key],
        frame_num=frame_num,
        shift=request.shift,
        sample_solver="unipc",
        sampling_steps=request.num_sampling_step,
        guide_scale=request.guide_scale,
        seed=request.seed,
        offload_model=True,
    )

    # Cache goal frames on the IDM device
    goal_frames = _video_tensor_to_goal_frames(video_tensor)  # (T, C, H, W)
    goal_frames = goal_frames.to(idm_device)

    session_id = uuid.uuid4().hex[:12]
    _video_store[session_id] = {
        "goal_frames": goal_frames,
        "created": time.time(),
    }
    logger.info(f"[generate_plan] session={session_id}, cached {goal_frames.shape[0]} goal frames")

    result = {
        "session_id": session_id,
        "frame_count": goal_frames.shape[0],
    }

    if request.return_video:
        video_b64 = base64.b64encode(
            _tensor_to_video_bytes(video_tensor, fps=wan_cfg.sample_fps)
        ).decode("utf-8")
        result["video_base64"] = video_b64

    return result


@api.post("/idm_step")
async def idm_step(request: IDMStepRequest):
    """Per-step goal-conditioned IDM inference.

    Takes the current observation + a goal frame index from a cached session,
    runs the GoalConditionedIDM, and returns a single 14-DOF action.
    """
    if gc_idm is None:
        raise HTTPException(status_code=503, detail="Goal-conditioned IDM not loaded (set GC_IDM_PATH)")

    session = _video_store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")

    goal_frames = session["goal_frames"]  # (T, C, H, W)
    if request.frame_index < 0 or request.frame_index >= goal_frames.shape[0]:
        raise HTTPException(
            status_code=400,
            detail=f"frame_index {request.frame_index} out of range [0, {goal_frames.shape[0]})"
        )

    # Decode current observation from base64 JPEG
    obs_pil = _decode_image(request.observation_b64)
    obs_tensor = torchvision.transforms.functional.to_tensor(obs_pil)  # (3, H, W), [0,1]
    obs_tensor = obs_tensor.unsqueeze(0).to(idm_device)  # (1, 3, H, W)

    # Get the goal frame for this step
    goal_tensor = goal_frames[request.frame_index].unsqueeze(0)  # (1, 3, H, W)

    # Preprocess both through the same transform (resize + ImageNet normalise)
    obs_proc = processor(obs_tensor)
    goal_proc = processor(goal_tensor)

    with torch.no_grad():
        actions, _ = gc_idm(obs_proc, goal_proc)  # (1, 14)

    return {
        "action": actions[0].cpu().tolist(),
    }


@api.post("/cleanup_session")
async def cleanup_session(session_id: str):
    """Free a cached video session."""
    if session_id in _video_store:
        del _video_store[session_id]
        return {"status": "ok"}
    return {"status": "not_found"}
