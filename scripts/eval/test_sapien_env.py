#!/usr/bin/env python3
"""Test SAPIEN environment for put_object_cabinet (no model server needed)."""
import sys
import os
import traceback

# Must run from vidar-robotwin directory
os.chdir(os.path.join(os.path.dirname(__file__), "../../vidar-robotwin"))
sys.path.insert(0, ".")
sys.path.insert(0, "./policy")

import yaml
from envs import CONFIGS_PATH
from envs.put_object_cabinet import put_object_cabinet

with open("task_config/hd_clean.yml") as f:
    args = yaml.safe_load(f)
args["task_name"] = "put_object_cabinet"
args["task_config"] = "hd_clean"
args["ckpt_setting"] = None
args["eval_mode"] = True

with open(CONFIGS_PATH + "_embodiment_config.yml") as f:
    emb = yaml.safe_load(f)
robot_file = emb["aloha-vidar"]["file_path"]
args["left_robot_file"] = robot_file
args["right_robot_file"] = robot_file
args["dual_arm_embodied"] = True
with open(robot_file + "/config.yml") as f:
    emb_cfg = yaml.safe_load(f)
args["left_embodiment_config"] = emb_cfg
args["right_embodiment_config"] = emb_cfg
with open(CONFIGS_PATH + "_camera_config.yml") as f:
    cam = yaml.safe_load(f)
args["head_camera_h"] = cam["Large_D435"]["h"]
args["head_camera_w"] = cam["Large_D435"]["w"]

print("Creating put_object_cabinet environment...")
env = put_object_cabinet()
try:
    env.setup_demo(now_ep_num=0, seed=1234100000, is_test=True, **args)
    print("setup_demo: SUCCESS")
    info = env.play_once()
    print(f"play_once: SUCCESS, info: {info['info']}")
    success = env.check_success()
    print(f"check_success: {success}")
except Exception:
    traceback.print_exc()
finally:
    env.close_env()
    print("Environment closed.")
