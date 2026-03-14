# vidar-robotwin Evaluation Environment Setup

This guide explains how to set up the vidar-robotwin evaluation pipeline on a new machine, using the **wanx** conda environment for both the video generation server and the SAPIEN simulation client.

## Prerequisites

- **GPU**: 1x A100-80GB (or equivalent, ~30GB for model + ~2GB for SAPIEN/curobo)
- **Conda**: miniconda or anaconda
- **Existing wanx env**: with PyTorch, Wan2.2 model code, PEFT, flash-attn

## Step 1: Install SAPIEN + Simulation Dependencies

These packages are needed for the SAPIEN simulation client (on top of the existing wanx env):

```bash
conda run -n wanx pip install \
    sapien==3.0.0b1 \
    mplib==0.2.1 \
    gymnasium==0.29.1 \
    transforms3d==0.4.2 \
    "pyglet<2" \
    zarr h5py termcolor

# open3d is optional (point cloud saving only, NOT needed for eval)
# open3d does NOT support Python 3.12 — skip if your env is py3.12
# pip install open3d==0.18.0  # only for py3.10/3.11
```

## Step 2: Install Server Dependencies

```bash
conda run -n wanx pip install uvicorn fastapi
```

## Step 3: Install ffmpeg

```bash
conda install -n wanx -y -c conda-forge ffmpeg
```

## Step 4: Apply Patches

SAPIEN and mplib require small patches for compatibility:

```bash
cd vidar-robotwin

# 1. Fix sapien urdf_loader.py encoding (adds utf-8 encoding to file reads)
SAPIEN_LOC=$(conda run -n wanx pip show sapien | grep Location | awk '{print $2}')/sapien
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $SAPIEN_LOC/wrapper/urdf_loader.py

# 2. Fix mplib planner.py (remove collision check that causes false failures)
MPLIB_LOC=$(conda run -n wanx pip show mplib | grep Location | awk '{print $2}')/mplib
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $MPLIB_LOC/planner.py
```

## Step 5: Install curobo (if not already installed)

curobo (CUDA Robot Optimization) is vendored in `envs/curobo/`:

```bash
cd vidar-robotwin/envs/curobo
conda run -n wanx pip install -e . --no-build-isolation
cd ../../..
```

## Step 6: Download Assets

```bash
cd vidar-robotwin
bash script/_download_assets.sh
```

This downloads robot models, object meshes, and textures to `assets/`.

## Verification

Test that SAPIEN + curobo work correctly:

```bash
conda run -n wanx --cwd /path/to/EmbodiedVideoRL \
    python scripts/eval/test_sapien_env.py
```

Expected output:
```
setup_demo: SUCCESS
play_once: SUCCESS
```

## Running Evaluation

See `scripts/eval/eval_vidar_put_object_cabinet.sh` for the full pipeline.

```bash
# On a GPU node:
cd /path/to/EmbodiedVideoRL
bash scripts/eval/eval_vidar_put_object_cabinet.sh
```

This script:
1. Starts the video generation server (Wan2.2 + IDM) on port 25400
2. Runs the SAPIEN client for 10 episodes of put_object_cabinet
3. Reports success rate to `vidar-robotwin/eval_result/ar/vidar_ckpt_test/put_object_cabinet/_result.txt`

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PT_DIR` | `ckpts/vidar_ckpt/merged_vidar_lora.pt` | Video generation model weights |
| `IDM_PATH` | `ckpts/vidar_ckpt/idm.pt` | Inverse dynamics model weights |
| `NFT_LORA_PATH` | (empty) | Optional NFT LoRA checkpoint |
| `CUDA_DEV` | `0` | GPU device index |
| `PORT` | `25400` | Server port |
| `PREFIX` | `vidar_ckpt_test` | Output subdirectory name |

## Path Portability

CuRobo YAML configs (`assets/embodiments/*/curobo*.yml`) use `${VIDAR_ROOT}` placeholders for all absolute paths. These are resolved at runtime by `envs/robot/robot.py::_resolve_curobo_yml()` using the vidar-robotwin root directory (computed from `envs/_GLOBAL_CONFIGS.py`). No hardcoded absolute paths — the configs work on any machine as long as the relative directory structure is preserved.

## Troubleshooting

### Vulkan ICD warning
```
UserWarning: Failed to find Vulkan ICD file.
```
This is normal on headless GPU nodes. SAPIEN falls back to its built-in ICD — rendering still works.

### `ModuleNotFoundError: No module named 'open3d'`
open3d is not needed for evaluation. If you see this error, it means `open3d` is imported at module level somewhere. The imports in `envs/utils/save_file.py` and `envs/camera/camera.py` have been made optional (`try/except`).

### `PermissionError` on curobo YAML paths
If you see paths like `/gpfs/projects/p33048/...`, the YAML files have old hardcoded paths. Check that all `curobo*.yml` files use `${VIDAR_ROOT}` placeholders:
```bash
grep -r "gpfs/projects" assets/embodiments/*/curobo*.yml
# Should return nothing — all should use ${VIDAR_ROOT}
```

### `missing pytorch3d`
pytorch3d is optional (used for farthest point sampling). The code falls back gracefully. To install:
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
