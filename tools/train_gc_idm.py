#!/usr/bin/env python3
"""Train the Goal-Conditioned IDM (GoalConditionedMask).

Reads directly from vidar-robotwin demo HDF5 files. Each file contains an
expert trajectory with JPEG-encoded camera observations and 14-DOF actions.

Training pairs are constructed from the same trajectory:
  - observation = composite of 3 cameras at timestep t
  - goal_frame  = composite of 3 cameras at timestep t + goal_horizon
  - action      = 14-DOF qpos at timestep t

The goal_horizon is randomised within [goal_min, goal_max] to teach the model
to handle varying temporal distances between current state and target.

Usage::

    # Train on put_object_cabinet demos (20 episodes)
    python tools/train_gc_idm.py \
        --data_dir vidar-robotwin/data/put_object_cabinet/demo_clean_ep20_vidar/data \
        --pretrained_idm ckpts/vidar_ckpts/idm.pt \
        --output_dir data/outputs/gc_idm_cabinet \
        --epochs 100 --lr 1e-4

    # Train on ALL 50 tasks (1000 episodes)
    python tools/train_gc_idm.py \
        --data_dir vidar-robotwin/data/*/demo_clean_ep20_vidar/data \
        --pretrained_idm ckpts/vidar_ckpts/idm.pt \
        --output_dir data/outputs/gc_idm_all \
        --epochs 200

    # Resume training
    python tools/train_gc_idm.py \
        --data_dir vidar-robotwin/data/put_object_cabinet/demo_clean_ep20_vidar/data \
        --resume data/outputs/gc_idm_cabinet/checkpoint_latest.pt \
        --output_dir data/outputs/gc_idm_cabinet \
        --epochs 200
"""

import argparse
import glob
import json
import logging
import os
import time

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────────


def _decode_jpeg(buf):
    """Decode a JPEG byte-string from HDF5 to BGR numpy array."""
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _composite_obs(head_bgr, left_bgr, right_bgr):
    """Reproduce deploy_policy.py::encode_obs — composite 3 cameras.

    Returns (720, 640, 3) uint8 RGB array.
    """
    h, w = head_bgr.shape[:2]  # 480, 640
    new_h, new_w = h // 2, w // 2  # 240, 320
    left_resized = cv2.resize(left_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    bottom = np.concatenate([left_resized, right_resized], axis=1)  # (240, 640, 3)
    combined = np.zeros((h + new_h, w, 3), dtype=head_bgr.dtype)   # (720, 640, 3)
    combined[:h] = head_bgr
    combined[h:] = bottom
    # BGR → RGB
    return combined[:, :, ::-1].copy()


class VidarDemoDataset(Dataset):
    """Goal-conditioned IDM dataset from vidar-robotwin HDF5 demo files.

    Constructs (obs_t, goal_{t+k}, action_t) tuples where k is sampled
    from [goal_min, goal_max].  Both obs and goal are composited from
    the same trajectory's 3-camera views.
    """

    def __init__(
        self,
        h5_paths: list,
        transform=None,
        goal_min: int = 1,
        goal_max: int = 20,
    ):
        self.transform = transform
        self.goal_min = goal_min
        self.goal_max = goal_max

        # Build index: (h5_path, timestep, T) per valid sample
        self.index = []
        for path in sorted(h5_paths):
            with h5py.File(path, "r") as f:
                T = f["joint_action/vector"].shape[0]
            # Only include timesteps where at least goal_min future frames exist
            for t in range(T - goal_min):
                self.index.append((path, t, T))

        logger.info(f"VidarDemoDataset: {len(self.index)} samples from {len(h5_paths)} episodes")

    def __len__(self):
        return len(self.index)

    def _read_frame(self, f, t):
        """Read and composite the 3-camera observation at timestep t."""
        head = _decode_jpeg(f["observation/head_camera/rgb"][t])
        left = _decode_jpeg(f["observation/left_camera/rgb"][t])
        right = _decode_jpeg(f["observation/right_camera/rgb"][t])
        return _composite_obs(head, left, right)  # (720, 640, 3) RGB uint8

    def __getitem__(self, idx):
        h5_path, t, T = self.index[idx]

        # Sample goal horizon
        k = np.random.randint(self.goal_min, min(self.goal_max, T - t - 1) + 1)
        t_goal = t + k

        with h5py.File(h5_path, "r") as f:
            obs_rgb = self._read_frame(f, t)          # (720, 640, 3) uint8
            goal_rgb = self._read_frame(f, t_goal)     # (720, 640, 3) uint8
            action = f["joint_action/vector"][t]       # (14,) float64

        # HWC uint8 → CHW float [0, 1]
        obs = torch.from_numpy(obs_rgb).permute(2, 0, 1).float() / 255.0
        goal = torch.from_numpy(goal_rgb).permute(2, 0, 1).float() / 255.0
        action = torch.from_numpy(action.astype(np.float32))

        if self.transform is not None:
            obs = self.transform(obs)
            goal = self.transform(goal)

        return obs, goal, action


# ── Training ─────────────────────────────────────────────────────────────────


def build_transform():
    """Standard IDM preprocessing: resize to 518x518 + ImageNet normalise."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((518, 518)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def collect_h5_paths(data_dir_patterns):
    """Expand glob patterns and collect all .hdf5 files."""
    paths = []
    for pattern in data_dir_patterns:
        expanded = glob.glob(pattern)
        for p in expanded:
            if os.path.isdir(p):
                paths.extend(sorted(glob.glob(os.path.join(p, "*.hdf5"))))
            elif p.endswith(".hdf5"):
                paths.append(p)
    return sorted(set(paths))


def train(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build model ──
    from server.idm import IDM, GoalConditionedMask

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model = IDM(model_name="goal_conditioned_mask", output_dim=14)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
    elif args.pretrained_idm:
        logger.info(f"Initialising from pretrained IDM: {args.pretrained_idm}")
        old_ckpt = torch.load(args.pretrained_idm, map_location="cpu", weights_only=False)
        old_sd = old_ckpt["model_state_dict"]
        # Extract the inner Mask state dict (strip "model." prefix)
        mask_sd = {k.replace("model.", ""): v for k, v in old_sd.items() if k.startswith("model.")}
        gc_mask = GoalConditionedMask.from_pretrained_mask(mask_sd, output_dim=14)
        model = IDM(model_name="goal_conditioned_mask", output_dim=14)
        model.model = gc_mask
        start_epoch = 0
    else:
        logger.info("Training from scratch")
        model = IDM(model_name="goal_conditioned_mask", output_dim=14)
        start_epoch = 0

    model = model.to(device)
    model.train()
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Dataset ──
    h5_paths = collect_h5_paths(args.data_dir)
    if not h5_paths:
        raise FileNotFoundError(f"No HDF5 files found in: {args.data_dir}")
    logger.info(f"Found {len(h5_paths)} HDF5 files")

    transform = build_transform()
    dataset = VidarDemoDataset(
        h5_paths,
        transform=transform,
        goal_min=args.goal_min,
        goal_max=args.goal_max,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimiser ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.resume and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if args.resume and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # ── Training loop ──
    best_loss = float("inf")
    log_path = os.path.join(args.output_dir, "training_log.jsonl")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (obs, goal, action_gt) in enumerate(loader):
            obs = obs.to(device)
            goal = goal.to(device)
            action_gt = action_gt.to(device)

            # Forward: model.model returns raw (normalised-space) output
            action_pred, _ = model.model(obs, goal)

            # Loss in normalised action space (balanced across DOFs)
            action_gt_norm = model.normalize(action_gt)
            loss = F.mse_loss(action_pred, action_gt_norm)

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": elapsed,
            "n_batches": n_batches,
        }
        logger.info(f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.6f}  "
                     f"lr={log_entry['lr']:.2e}  time={elapsed:.1f}s  batches={n_batches}")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Save checkpoint
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch + 1,
            "loss": avg_loss,
        }
        torch.save(state, os.path.join(args.output_dir, "checkpoint_latest.pt"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(state, os.path.join(args.output_dir, "checkpoint_best.pt"))

        if (epoch + 1) % args.save_every == 0:
            torch.save(state, os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1:04d}.pt"))

    logger.info(f"Training complete. Best loss: {best_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train Goal-Conditioned IDM")
    parser.add_argument("--data_dir", type=str, nargs="+", required=True,
                        help="Directories or glob patterns containing vidar HDF5 demo files. "
                             "E.g. 'vidar-robotwin/data/put_object_cabinet/demo_clean_ep20_vidar/data' "
                             "or 'vidar-robotwin/data/*/demo_clean_ep20_vidar/data'")
    parser.add_argument("--output_dir", type=str, default="data/outputs/gc_idm",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--pretrained_idm", type=str, default="",
                        help="Path to existing single-input IDM checkpoint for init")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to a GC-IDM checkpoint to resume training from")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=int, default=0)

    # Goal frame sampling
    parser.add_argument("--goal_min", type=int, default=1,
                        help="Minimum future-frame offset for goal (default: 1)")
    parser.add_argument("--goal_max", type=int, default=20,
                        help="Maximum future-frame offset for goal (default: 20)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
