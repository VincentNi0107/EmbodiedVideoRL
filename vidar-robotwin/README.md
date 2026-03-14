# Robotwin Eval code for Vidar/Vidarc

## Overview
We utilize a client-server architecture for evaluation. This repository serves as the **client**, responsible for managing the server, sending requests, and executing evaluations upon receiving actions.
Before proceeding, please ensure that the server-side environment and code are properly set up.

## Env Setup
Please refer to [READMEEnv.md](READMEEnv.md).

## Evaluation
This module provides a unified evaluation script based on `torch.distributed` (DDP), designed to simplify the multi-GPU/multi-task evaluation workflow in `Client-Server` mode.

## Features

1.  **Unified Architecture**: Uses `torchrun` to launch a single Python script.
2.  **Automatic Task Distribution**: Automatically splits task lists leveraging DDP's `rank` and `world_size`, removing the need for manual assignment.
3.  **Robust Process Management**: Utilizes Python Context Manager to manage the Server lifecycle, ensuring that the Server and its subprocesses are cleanly terminated regardless of normal completion or abnormal exit.
4.  **Decoupled Design**: All paths (Server script, model, task descriptions) are passed via arguments rather than being hardcoded.
5.  **Resumable Execution**: Automatically skips tasks that already have existing logs.

## Dependencies

- PyTorch (for `torch.distributed`)
- Existing `vidar` Server script
- Existing `script/eval_policy.py` Client script

## Usage
bash collect_data.sh beat_block_hammer demo_clean 0 20

```bash
conda activate RoboTwin-hb

# eval with vidarc
bash run_eval_ddp_causal.sh

# eval with vidar
bash run_eval_ddp.sh 
```
all_tasks=(
  # adjust_bottle
  # beat_block_hammer
  # blocks_ranking_rgb
  blocks_ranking_size
  # click_alarmclock
  # click_bell
  dump_bin_bigbin
  # grab_roller
  handover_block
  handover_mic
  hanging_mug
  # lift_pot
  # move_can_pot
  # move_pillbottle_pad
  # move_playingcard_away
  move_stapler_pad
  open_laptop
  open_microwave
  pick_diverse_bottles
  # pick_dual_bottles
  # place_a2b_left
  # place_a2b_right
  # place_bread_basket
  # place_bread_skillet
  place_burger_fries
  # place_can_basket
  place_cans_plasticbox
  # place_container_plate
  place_dual_shoes
  # place_empty_cup
  place_fan
  # place_mouse_pad
  place_object_basket
  place_object_scale
  # place_object_stand
  # place_phone_stand
  # place_shoe
  # press_stapler
  put_bottles_dustbin
  put_object_cabinet
  # rotate_qrcode
  scan_object
  # shake_bottle_horizontally
  # shake_bottle
  stack_blocks_three
  # stack_blocks_two
  # stack_bowls_three
  # stack_bowls_two
  stamp_seal
  # turn_switch
)

all_tasks=(
  handover_block
  handover_mic
  hanging_mug
  scan_object
)

all_tasks=(
  move_stapler_pad
  open_laptop
  open_microwave
  put_object_cabinet
)

all_tasks=(
  pick_diverse_bottles
  place_burger_fries
  place_cans_plasticbox
  stack_blocks_three
  stamp_seal
)

all_tasks=(
  place_dual_shoes
  place_fan
  place_object_basket
  place_object_scale
  put_bottles_dustbin
)

### Parameters

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--server_script` | Path to the Server startup script (Required) | - |
| `--model` | Path to the model (Required) | - |
| `--idm` | Path to the Inverse Dynamics Model | - |
| `--prefix` | Prefix for the output directory (Required) | "debug" |
| `--task_dir` | Directory containing task description files | "./description/task_instruction" |
| `--server_cwd` | Working directory for the Server script | "../cosmos-predict2" |
| `--base_port` | Starting port number (Rank 0 uses base, Rank 1 uses base+1...) | 25400 |
