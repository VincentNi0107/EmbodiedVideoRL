import cv2
import numpy as np

from .ar import AR


def encode_obs(observation):  # Post-Process Observation
    obs = observation

    head_rgbs = np.array(obs['observation']["head_camera"]["rgb"])
    left_rgbs = np.array(obs['observation']["left_camera"]["rgb"])
    right_rgbs = np.array(obs['observation']["right_camera"]["rgb"])

    # rgb to bgr
    head_rgbs = head_rgbs[..., ::-1]
    left_rgbs = left_rgbs[..., ::-1]
    right_rgbs = right_rgbs[..., ::-1]
        
    # This logic assumes a single frame observation, not a sequence.
    # The original logic seemed to expect a sequence (len(head_rgbs)).
    # We'll process a single frame observation dict.
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


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    """Initializes and returns the Vidar policy model."""
    model = AR(usr_args=usr_args)
    return model


def eval(TASK_ENV, model, observation):
    """
    The AR model now handles the full observation-to-action pipeline.
    Supports both open-loop (original) and closed-loop (goal-conditioned IDM) modes.
    """
    TASK_ENV.step_lim = model.max_steps
    obs = encode_obs(observation)

    # Set instruction and update observation for the model
    model.set_episode_id(TASK_ENV.ep_num)
    if model.task_config.startswith("demo"):
        instruction = TASK_ENV.instruction
        model.set_demo_instruction(instruction)
    else:
        instruction = TASK_ENV.full_instruction
        model.set_instruction(instruction)
    model.update_obs(obs)
    print(f"Instruction: {model.prompt}")

    if model.closed_loop:
        _eval_closed_loop(TASK_ENV, model, obs)
    else:
        _eval_open_loop(TASK_ENV, model)

    model.save_videos()


def _eval_open_loop(TASK_ENV, model):
    """Original open-loop execution: generate all actions at once, execute sequentially."""
    actions = model.get_actions()
    action_idx = 0
    while action_idx < len(actions):
        action = actions[action_idx]
        TASK_ENV.take_action(action, action_type='qpos')
        if TASK_ENV.eval_success:
            break
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)
        action_idx += 1


def _eval_closed_loop(TASK_ENV, model, obs):
    """Closed-loop execution: goal-conditioned IDM queries the server per step."""
    from base64 import b64encode as _b64encode
    import torch
    import torchvision

    def _obs_to_b64(obs_array):
        """Convert HWC uint8 numpy observation to base64 JPEG."""
        img = torch.tensor(obs_array, dtype=torch.uint8).permute(2, 0, 1)
        jpeg_bytes = torchvision.io.encode_jpeg(img).numpy().tobytes()
        return _b64encode(jpeg_bytes).decode("utf-8")

    # Phase 1: generate video plan
    model.generate_plan()
    obs_b64 = _obs_to_b64(obs)

    # Phase 2: closed-loop execution
    frame_idx = 0
    while frame_idx < model.plan_frame_count:
        action = model.get_single_action(obs_b64, frame_idx)
        TASK_ENV.take_action(action, action_type='qpos')
        if TASK_ENV.eval_success:
            break

        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)
        obs_b64 = _obs_to_b64(obs)
        frame_idx += 1

        # Re-plan if needed (MPC-style sliding window)
        if model.should_replan(frame_idx):
            print(f"Re-planning at frame {frame_idx}")
            model.generate_plan()
            frame_idx = 0


def reset_model(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset()
