# -- coding: UTF-8
import numpy as np
import json
import requests
import cv2
import urllib3
from base64 import b64encode, b64decode
import os
import multiprocessing
import subprocess
import logging
import torch
import torchvision
import time
from datetime import datetime


def save_video(ffmpeg_cmd, images):
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    for image in images:
        img = cv2.imdecode(np.frombuffer(b64decode(image), np.uint8), cv2.IMREAD_COLOR)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()


def save_videos(videos, width, height, fps=8):
    workers = []
    for k, v in videos.items():
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', '-c:v', 'libx264', '-preset', 'veryslow',
            '-crf', '10', '-threads', '1', '-pix_fmt', 'yuv420p',
            '-loglevel', 'error', k
        ]
        workers.append(multiprocessing.Process(target=save_video, args=(ffmpeg_cmd, v)))
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


def worker(port, headers, data, verify):
    resp = requests.post(
        f"http://localhost:{port}",
        headers=headers,
        data=json.dumps(data),
        verify=verify,
        timeout=600,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Server HTTP {resp.status_code}. Body (first 1000 chars): {resp.text[:1000]}"
        )
    try:
        response = resp.json()
    except Exception as e:
        raise RuntimeError(
            f"Server returned non-JSON response (first 1000 chars): {resp.text[:1000]}"
        ) from e
    assert len(response) > 0, "password error"
    return response


class AR:
    def __init__(self, usr_args=None):
        if usr_args is None:
            usr_args = {}
        self.usr_args = usr_args
        self.policy_name = usr_args["policy_name"]
        self.task_name = usr_args["task_name"]
        self.task_config = usr_args["task_config"]
        self.num_new_frames = usr_args["num_new_frames"]
        self.num_sampling_step = usr_args["num_sampling_step"]
        self.max_steps = usr_args["max_steps"]
        self.seed = usr_args["seed"]
        self.port = usr_args["port"]
        self.save_dir = usr_args["save_dir"]
        self.obs_cache = []
        self.prompt = None
        self.episode_id = 0
        self.timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_imgs = []
        self.out_masks = []
        self.video_ffmpeg = None
        self.num_conditional_frames = None
        self.rollout_bound = usr_args["rollout_bound"]
        self.rollout_prefill_num = usr_args["rollout_prefill_num"]
        self.guide_scale = usr_args["guide_scale"]
        os.makedirs(self.save_dir, exist_ok=True)

        # Closed-loop state
        self.closed_loop = usr_args.get("closed_loop", False)
        self.session_id = None
        self.plan_frame_count = 0
        self.replan_interval = usr_args.get("replan_interval", 0)  # 0 = disabled

    def set_ffmpeg(self, save_path):
        self.video_ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    "640x736",
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{save_path}",
                ],
                stdin=subprocess.PIPE,
            )

    def close_ffmpeg(self):
        if self.video_ffmpeg:
            self.video_ffmpeg.stdin.close()
            self.video_ffmpeg.wait()
            del self.video_ffmpeg

    def reset(self):
        """Resets the internal state of the model."""
        self.obs_cache = []
        self.prompt = ""
        self.out_imgs = []
        self.out_masks = []
        self.num_conditional_frames = 1
        # Clean up closed-loop session
        if self.session_id is not None:
            try:
                requests.post(
                    f"http://localhost:{self.port}/cleanup_session",
                    params={"session_id": self.session_id},
                    timeout=5,
                )
            except Exception:
                pass
            self.session_id = None
            self.plan_frame_count = 0
        print("AR model has been reset.")

    def update_obs(self, obs):
        """Updates the model with the latest observation."""
        img = torch.tensor(obs, dtype=torch.uint8).permute(2, 0, 1)
        img = torchvision.io.encode_jpeg(img)
        img = b64encode(img.numpy().tobytes()).decode("utf-8")
        self.obs_cache.append(img)

    def set_instruction(self, instruction):
        """Sets the task instruction for the policy."""

        system_prompt = "The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: "
        if instruction:
            instruction = instruction[0].lower() + instruction[1:]
        self.prompt = system_prompt + instruction

    def set_demo_instruction(self, instruction):
        """Sets the task instruction for the policy."""

        system_prompt = "The whole scene is in a realistic, industrial art style with three views: a fixed front camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: "
        self.prompt = system_prompt + instruction

    def set_task_name(self, task_name):
        """Sets the task name for the policy."""
        self.task_name = task_name

    def set_episode_id(self, episode_id):
        """Sets the episode ID for the current run."""
        self.episode_id = episode_id

    def modify_actions(self, actions):
        """
        以第一帧为基准，对夹爪角度的变化量做线性变换后施加回去，并将夹角限制在[0, 5]区间
        :param actions: shape (N, 14)
        :param gripper_strengthen_factor: 收紧趋势加强系数
        :param bias: 偏置
        :return: 修改后的actions
        """
        actions = np.array(actions)
        for dim in [6, 13]:
            smoothed = actions[:, dim].copy()
            for i in range(2, len(smoothed)-2):
                smoothed[i] = (actions[i-2, dim] + actions[i-1, dim] + actions[i, dim] + actions[i+1, dim] + actions[i+2, dim]) / 5
            actions[:, dim] = smoothed
            # if action is decreasing and action < 3, then set to 0
            diffs = actions[1:, dim] - actions[:-1, dim]
            mask = diffs < 0.1
            # append one more element to mask
            mask = np.concatenate(([False], mask))
            actions[:, dim] = np.where(mask & (actions[:, dim] < 0.3), np.clip(actions[:, dim],None,0), actions[:, dim])
            actions[:, dim] = np.where(actions[:, dim] > 0.7, np.clip(actions[:, dim], 1, None), actions[:, dim])
        return actions.tolist()

    def get_actions(self):
        if len(self.obs_cache) >= self.max_steps:
            return []
        headers = {
            "Content-Type": "application/json",
        }
        port, seed = self.port, self.seed
        t = time.time()
        if self.num_conditional_frames + self.num_new_frames > self.rollout_bound:
            self.num_conditional_frames = self.rollout_prefill_num
            obs_cache = self.obs_cache[-self.num_conditional_frames:]
            clean_cache = True
        else:
            obs_cache = self.obs_cache[-self.num_new_frames:]
            clean_cache = False
        data = {"prompt": self.prompt, "imgs": obs_cache, "num_conditional_frames": self.num_conditional_frames, "num_new_frames": self.num_new_frames, "seed": seed, "num_sampling_step": self.num_sampling_step, "guide_scale": self.guide_scale, "password": "r49h8fieuwK", "return_imgs": True, "clean_cache": clean_cache}
        
        response = worker(port, headers, data, False)
        print(f"Inference done with time usage {time.time() - t}")
        actions = json.loads(response["actions"])
        print(f"Actions length: {len(actions)}")
        # actions = self.modify_actions(actions)
        if "imgs" in response:
            self.out_imgs += response["imgs"]
        if "masks" in response:
            self.out_masks += response["masks"]
        self.num_conditional_frames += self.num_new_frames
        return actions

    def save_videos(self):
        if self.out_imgs:
            self.set_ffmpeg(os.path.join(self.save_dir, f"episode{self.episode_id}_pred_{len(self.out_imgs)}.mp4"))
            for i, v in enumerate(self.out_imgs):
                img = cv2.imdecode(np.frombuffer(b64decode(v), np.uint8), cv2.IMREAD_COLOR)
                # cv2.imwrite(os.path.join(self.save_dir, f"episode{self.episode_id}_pred_{i}.jpg"), img)
                self.video_ffmpeg.stdin.write(img[:, :, ::-1].tobytes())
            self.close_ffmpeg()
        if self.out_masks:
            self.set_ffmpeg(os.path.join(self.save_dir, f"episode{self.episode_id}_mask_{len(self.out_masks)}.mp4"))
            for i, v in enumerate(self.out_masks):
                img = cv2.imdecode(np.frombuffer(b64decode(v), np.uint8), cv2.IMREAD_COLOR)
                # cv2.imwrite(os.path.join(self.save_dir, f"episode{self.episode_id}_mask_{i}.jpg"), img)
                self.video_ffmpeg.stdin.write(img[:, :, ::-1].tobytes())
            self.close_ffmpeg()

    # ── Closed-loop methods ──────────────────────────────────────────────────

    def generate_plan(self):
        """Generate a video plan and cache goal frames on the server.

        Sends the most recent observation to POST /generate_plan,
        receives a session_id for subsequent per-step IDM queries.
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": self.prompt,
            "imgs": [self.obs_cache[-1]],
            "num_new_frames": self.num_new_frames,
            "seed": self.seed,
            "num_sampling_step": self.num_sampling_step,
            "guide_scale": self.guide_scale,
            "password": "r49h8fieuwK",
        }
        t = time.time()
        resp = requests.post(
            f"http://localhost:{self.port}/generate_plan",
            headers=headers,
            data=json.dumps(data),
            timeout=600,
        )
        resp.raise_for_status()
        result = resp.json()
        assert len(result) > 0, "password error"

        self.session_id = result["session_id"]
        self.plan_frame_count = result["frame_count"]
        print(f"Plan generated: session={self.session_id}, "
              f"frames={self.plan_frame_count}, time={time.time() - t:.1f}s")

    def get_single_action(self, obs_b64, frame_index):
        """Query the goal-conditioned IDM for one action.

        Args:
            obs_b64: base64-encoded JPEG of the current observation
            frame_index: which goal frame to condition on (0-indexed)

        Returns:
            list of 14 floats (joint positions + grippers)
        """
        data = {
            "session_id": self.session_id,
            "frame_index": frame_index,
            "observation_b64": obs_b64,
        }
        resp = requests.post(
            f"http://localhost:{self.port}/idm_step",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["action"]

    def should_replan(self, frame_index):
        """Check whether to re-generate the video plan.

        Returns True when periodic re-planning is enabled and the current
        frame_index has reached the re-plan interval boundary.
        """
        if self.replan_interval <= 0:
            return False
        return (frame_index > 0) and (frame_index % self.replan_interval == 0)
