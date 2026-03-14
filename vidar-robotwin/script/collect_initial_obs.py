import os
import sys
import json
import argparse
import importlib
import traceback
from pathlib import Path
from datetime import datetime
import base64
from typing import Any
import csv

import yaml
import numpy as np
import cv2


sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from description.utils.generate_episode_instructions import generate_episode_descriptions


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.float32:
                dtype = "float32"
            elif obj.dtype == np.float64:
                dtype = "float64"
            elif obj.dtype == np.int32:
                dtype = "int32"
            elif obj.dtype == np.int64:
                dtype = "int64"
            else:
                dtype = str(obj.dtype)
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": dtype,
                "shape": obj.shape,
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    return json.dumps(data, cls=NumpyEncoder, ensure_ascii=False)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit("No Task")
    return env_instance


def get_camera_config(camera_type):
    camera_config_path = os.path.join(os.path.dirname(__file__), "../task_config/_camera_config.yml")
    if not os.path.isfile(camera_config_path):
        raise FileNotFoundError("task config file is missing")
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    if camera_type not in args:
        raise ValueError(f"camera {camera_type} is not defined")
    return args[camera_type]




def encode_obs(observation):
    obs = observation

    head_rgbs = np.array(obs['observation']["head_camera"]["rgb"])
    left_rgbs = np.array(obs['observation']["left_camera"]["rgb"])
    right_rgbs = np.array(obs['observation']["right_camera"]["rgb"])

    head_rgbs = head_rgbs[..., ::-1]
    left_rgbs = left_rgbs[..., ::-1]
    right_rgbs = right_rgbs[..., ::-1]

    head_img = head_rgbs
    left_img = left_rgbs
    right_img = right_rgbs

    h, w, _ = head_img.shape
    new_h, new_w = h // 2, w // 2

    left_resized = cv2.resize(left_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    bottom_row = np.concatenate([left_resized, right_resized], axis=1)
    assert bottom_row.shape[1] == w

    final_h = h + new_h
    combined_img = np.zeros((final_h, w, 3), dtype=head_img.dtype)

    combined_img[:h, :w] = head_img
    combined_img[h:, :w] = bottom_row
    combined_img = combined_img[:, :, ::-1].copy()
    return combined_img


def _format_instruction(instruction: str) -> str:
    if not instruction:
        return instruction
    return instruction[0].lower() + instruction[1:]


def build_prompt_from_instruction(instruction: str) -> str:
    system_prompt = (
        "The whole scene is in a realistic, industrial art style with three views: "
        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
        "The aloha robot is currently performing the following task: "
    )
    return system_prompt + _format_instruction(instruction)

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    parser.add_argument("--start_seed", type=int, default=1234)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--per_task", type=int, default=10)
    parser.add_argument("--max_attempts", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="initial_obs")
    parser.add_argument("--out_json", type=str, default="/home/omz1504/code/vidar/data/test/robotwin_collected.json")
    parser.add_argument("--image_dir", type=str, default="/home/omz1504/code/vidar/data/test")
    parser.add_argument(
        "--success_csv",
        type=str,
        default="/home/omz1504/code/vidar-robotwin/eval_result/ar/single_test/success_rates.csv",
    )
    parser.add_argument("--success_threshold", type=float, default=0.6)
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Comma-separated task names. If set, override success-rate based task selection.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except Exception:
                pass
            override_dict[key] = value
        return override_dict

    config["start_seed"] = args.start_seed
    config["num_seeds"] = args.num_seeds
    config["per_task"] = args.per_task
    config["max_attempts"] = args.max_attempts
    config["save_dir"] = args.save_dir
    config["out_json"] = args.out_json
    config["image_dir"] = args.image_dir
    config["success_csv"] = args.success_csv
    config["success_threshold"] = args.success_threshold
    config["tasks"] = args.tasks
    
    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        # Keep overrides as the highest-priority source.
        config.update(overrides)

    return config


def load_low_success_tasks(csv_path: str, threshold: float) -> list:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"success_rates.csv not found: {csv_path}")
    tasks = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = (row.get("task") or "").strip()
            rate_str = (row.get("success_rate") or "").strip()
            if not task or not rate_str:
                continue
            try:
                rate = float(rate_str)
            except ValueError:
                continue
            if rate < threshold:
                tasks.append(task)
    return tasks


def main(usr_args):
    task_name = usr_args.get("task_name", "")
    task_config = usr_args["task_config"]

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = usr_args.get("ckpt_setting", "")

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["eval_mode"] = True

    save_dir = Path(usr_args["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    out_json = Path(usr_args["out_json"])
    image_dir = Path(usr_args["image_dir"])
    image_dir.mkdir(parents=True, exist_ok=True)

    instruction_type = usr_args.get("instruction_type", "seen")


    task_arg = str(usr_args.get("tasks", "")).strip()
    if task_arg:
        tasks = [task.strip() for task in task_arg.split(",") if task.strip()]
    else:
        tasks = load_low_success_tasks(usr_args["success_csv"], float(usr_args["success_threshold"]))
    print(tasks)
    
    start_seed = int(usr_args.get("seed", usr_args.get("start_seed", 1234)))
    base_seed = 100000 * (1 + start_seed)
    per_task = int(usr_args.get("per_task", 10))
    max_attempts = int(usr_args.get("max_attempts", 200))

    records = []

    for task_name in tasks:
        args["task_name"] = task_name
        TASK_ENV = class_decorator(task_name)
        task_image_dir = image_dir / task_name
        task_image_dir.mkdir(parents=True, exist_ok=True)
        collected = 0
        attempts = 0
        now_seed = base_seed

        while collected < per_task and attempts < max_attempts:
            try:
                TASK_ENV.setup_demo(now_ep_num=collected, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                print(f"[WARN] UnStableError at seed {now_seed}: {e}")
                now_seed += 1
                attempts += 1
                continue
            except Exception:
                print(f"[WARN] Error at seed {now_seed}:")
                print(traceback.format_exc())
                now_seed += 1
                attempts += 1
                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass
                continue

            if not (getattr(TASK_ENV, "plan_success", False) and TASK_ENV.check_success()):
                now_seed += 1
                attempts += 1
                continue

            try:
                TASK_ENV.setup_demo(now_ep_num=collected, seed=now_seed, is_test=True, **args)
                episode_info_list = [episode_info["info"]]
                results = generate_episode_descriptions(task_name, episode_info_list, max_descriptions=10)
                instruction_pool = results[0].get(instruction_type, [])
                if not instruction_pool:
                    raise RuntimeError(f"No instructions for type '{instruction_type}'")
                instruction = np.random.choice(instruction_pool)
                TASK_ENV.set_instruction(instruction=instruction)
                instruction = TASK_ENV.full_instruction
                print(instruction)
                observation = TASK_ENV.get_obs()
                obs_img = encode_obs(observation)
                prompt = build_prompt_from_instruction(instruction)

                filename_stem = f"robotwin_{task_name}_{now_seed}"
                image_path = task_image_dir / f"{filename_stem}.png"
                cv2.imwrite(str(image_path), obs_img[:, :, ::-1])
                records.append({
                    "prompt": prompt,
                    "filename_stem": filename_stem,
                    "media_path": str(image_path),
                })
                collected += 1
                print(f"[INFO] Saved initial image: {image_path}")
            except Exception:
                print(f"[WARN] Error at seed {now_seed} when building prompt/obs:")
                print(traceback.format_exc())
            finally:
                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass

            now_seed += 1
            attempts += 1

        print(f"[INFO] Task {task_name}: collected {collected}/{per_task} (attempts={attempts})")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote metadata: {out_json}")


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()
    usr_args = parse_args_and_config()
    main(usr_args)
