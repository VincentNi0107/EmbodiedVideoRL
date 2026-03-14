# Robotwin Eval code for Vidar/Vidarc

## 概览
我们使用client-server模式来进行评估, 本仓库为client端, 负责管理server, 并向server发起请求, 获得action后执行评估。
在此之前, 请保证server端的环境和代码已经准备好。

## 环境搭建
请查看 [READMEEnv.md](READMEEnv.md)。


## 进行评估
本模块提供了一个基于 `torch.distributed` (DDP) 的统一评估脚本，旨在简化 `Client-Server` 模式下的多卡/多任务评估流程。

## 功能特点

1.  **统一架构**: 统一使用 `torchrun` 启动单一 Python 脚本。
2.  **自动任务分发**: 利用 DDP 的 `rank` 和 `world_size` 自动切分任务列表，无需手动分配。
3.  **健壮的进程管理**: 使用 Python Context Manager 管理 Server 生命周期，确保无论正常结束还是异常退出，Server 及其子进程都会被清理干净。
4.  **解耦设计**: 所有路径（Server 脚本、模型、任务描述）均通过参数传入，不再硬编码。
5.  **断点续传**: 自动跳过已存在日志的任务。

## 依赖

- PyTorch (用于 `torch.distributed`)
- 现有的 `vidar` Server 脚本
- 现有的 `script/eval_policy.py` Client 脚本

## 使用方法

```bash
conda activate RoboTwin-hb

# eval with vidarc
bash run_eval_ddp_causal.sh

# eval with vidar
bash run_eval_ddp.sh 
```

### 2. 参数说明

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--server_script` | Server 启动脚本路径 (必需) | - |
| `--model` | 模型路径 (必需) | - |
| `--idm` | 逆动力学模型路径 |
| `--prefix` | 输出目录的前缀 (必需) | "debug" |
| `--task_dir` | 任务描述文件所在目录 | "./description/task_instruction" |
| `--server_cwd` | Server 脚本的工作目录 | "../cosmos-predict2" |
| `--base_port` | 起始端口号 (Rank 0 使用 base, Rank 1 使用 base+1...) | 25400 |


