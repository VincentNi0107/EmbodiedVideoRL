import argparse
import logging
import os
import signal
import socket
import subprocess
import time
from typing import Optional
import torch.distributed as dist

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | Rank %(process)d | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ServerManager:
    """Server 进程管理器 (Context Manager)"""
    def __init__(self, script_path: str, model: str, idm: str, port: int, device_id: int, cwd: str = "."):
        self.script_path = os.path.abspath(script_path)
        self.cmd = [
            self.script_path, model, idm, str(port), str(device_id), "localhost"
        ]
        self.port = port
        self.cwd = cwd
        self.process: Optional[subprocess.Popen] = None

    def _wait_for_port(self, timeout: int = 300) -> bool:
        """轮询检测端口是否开启"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            # check process is running, if faild, print std out and std err
            if self.process.poll() is not None:
                logger.error(f"Server process failed to start: {self.process.poll()=}")
                return False
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                if sock.connect_ex(('localhost', self.port)) == 0:
                    return True
            time.sleep(2)
        return False

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, stopping server...")
        raise SystemExit(f"Received signal {signum}")

    def __enter__(self):
        assert os.path.exists(self.script_path), f"Server working directory not found: {self.script_path}"

        # Register signal handlers
        self.old_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        self.old_sigint = signal.signal(signal.SIGINT, self._signal_handler)

        try:
            logger.info(f"Starting Server on Port {self.port}...")
            # 使用 preexec_fn=os.setsid 创建新的进程组，方便后续杀掉整个进程树
            logger.info(' '.join(self.cmd))
            self.process = subprocess.Popen(
                self.cmd, cwd=self.cwd,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )

            if self._wait_for_port():
                logger.info("Server is READY.")
                return self
            
            raise RuntimeError("Server failed to start within timeout.")

        except (Exception, SystemExit):
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except Exception as e:
                logger.error(f"Failed to kill server process {self.process.pid}: {e}")
            

def run_client_task(task_name: str, args, port: int, output_dir_base: str, local_rank: int):
    """运行单个 Client 任务"""
    task_out_dir = os.path.join(output_dir_base, args.prefix, task_name)
    log_file = os.path.join(task_out_dir, "log.txt")

    # 自动跳过
    if os.path.exists(log_file):
        logger.info(f"Task {task_name} seems already done. Skipping.")
        return

    os.makedirs(task_out_dir, exist_ok=True)
    
    cmd = [
        "python", "script/eval_policy.py",
        "--config", "policy/AR/deploy_policy.yml",
        "--overrides",
        "--task_name", task_name,
        "--task_config", args.task_config,
        "--port", str(port),
        "--seed", str(args.seed),
        "--policy_name", "AR",
        "--num_new_frames", str(args.num_new_frames),
        "--num_sampling_step", str(args.num_sampling_step),
        "--guide_scale", str(args.cfg),
        "--rollout_bound", str(args.rollout_bound),
        "--rollout_prefill_num", str(args.rollout_prefill_num),
        "--save_dir", task_out_dir
    ]
    logger.info(" ".join(cmd))
    logger.info(f"Running Task: {task_name}")
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
    env['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    env['PYTHONUNBUFFERED'] = '1'
    
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_script", type=str, required=True, help="Server script path")
    parser.add_argument("--server_cwd", type=str, default="../", help="Server working directory")
    parser.add_argument("--task_dir", type=str, default="./description/task_instruction", help="Task instruction directory")
    parser.add_argument("--output_dir", type=str, default="eval_result/ar")
    parser.add_argument("--model", type=str, required=True, help="Model path for server")
    parser.add_argument("--idm", type=str, default="0418_3e-3_60000.pt", help="IDM path for server")
    parser.add_argument("--task_config", type=str, default="hd_clean", help="Task configuration")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prefix", type=str, default="debug", help="Prefix for output directory")
    parser.add_argument("--num_new_frames", type=int, default=80)
    parser.add_argument("--num_sampling_step", type=int, default=5, help="Number of sampling steps")
    parser.add_argument("--cfg", type=float, default=5.0, help="Guide scale")
    parser.add_argument("--rollout_prefill_num", type=int, default=33)
    parser.add_argument("--rollout_bound", type=int, default=60)
    parser.add_argument("--base_port", type=int, default=25400)
    args = parser.parse_args()
    
    
    args.model = os.path.abspath(args.model)
    args.idm = os.path.abspath(args.idm)
    assert os.path.exists(args.model), f"Model path not found: {args.model}"
    assert os.path.exists(args.idm), f"IDM path not found: {args.idm}"
    # DDP 初始化
    rank, world_size, local_rank = 0, 1, 0
    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        logger.info("Running in single process mode (Non-DDP).")

    # 获取并切分任务
    all_tasks = sorted([i.split(".")[0] for i in os.listdir(args.task_dir)])
    all_tasks = ['stack_bowls_two', 'place_cans_plasticbox', 'beat_block_hammer', 'pick_dual_bottles', 'click_alarmclock', 'click_bell', 'shake_bottle_horizontally', 'open_laptop','turn_switch', 'press_stapler', 'shake_bottle', 'place_bread_basket', 'grab_roller', 'place_burger_fries', 'place_phone_stand', 'place_object_stand', 'place_container_plate', 'place_a2b_right', 'place_empty_cup', 'adjust_bottle', 'dump_bin_bigbin']
    my_tasks = all_tasks[rank::world_size]
    
    logger.info(f"Rank {rank} {local_rank=} assigned {len(my_tasks)} tasks")

    if not my_tasks:
        return

    # 启动 Server 并运行 Client
    try:
        with ServerManager(
            args.server_script, args.model, args.idm, 
            args.base_port + rank, local_rank, args.server_cwd
        ):
            for task in my_tasks:
                run_client_task(task, args, args.base_port + rank, args.output_dir, local_rank)
    except Exception as e:
        logger.error(f"Error in Rank {rank}: {e}")
        if world_size > 1:
            raise e



if __name__ == "__main__":
    main()
