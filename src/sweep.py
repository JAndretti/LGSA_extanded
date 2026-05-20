import os
import random
import subprocess
import sys
from itertools import product
from multiprocessing import Process, Queue
from time import sleep

import torch
import yaml
from loguru import logger
from src.utils import setup_logger

# ─────────────────────────────────────────────────────────────
#  Global Parameters
# ─────────────────────────────────────────────────────────────

SWEEP_MODE = "random"  # "grid" | "random"
TRAINING_MODULE = "src.train"  # run as: python -m src.train
HYPERPARAMETERS_PATH = "src/HP/HP_sweep.yaml"


# ─────────────────────────────────────────────────────────────
#  Config Parsing
# ─────────────────────────────────────────────────────────────

def read_YAML_hyperparameters_sweep(hp_path: str) -> tuple[list[str], list[tuple]]:
    with open(hp_path, "r") as f:
        documents = list(yaml.safe_load_all(f))

    grids = [doc for doc in documents if doc is not None]
    if not grids:
        raise ValueError(f"No valid YAML documents found in {hp_path}")
    grids = [grids[0]] if len(grids) == 1 and isinstance(grids[0], dict) else grids

    all_configs: list[tuple] = []
    keys: list[str] = []

    for i, grid in enumerate(grids):
        current_keys = sorted(grid.keys())
        if i == 0:
            keys = current_keys
        elif current_keys != keys:
            raise ValueError(
                f"Grid #{i} has different parameters than the first grid. "
                "All groups must share the same keys."
            )

        values_list = []
        for k in keys:
            val = grid[k]
            if not isinstance(val, list):
                val = [val]
            values_list.append(val)

        all_configs.extend(product(*values_list))

    return keys, all_configs


# ─────────────────────────────────────────────────────────────
#  Worker Logic
# ─────────────────────────────────────────────────────────────

def run_training_script(gpu_id: int, hp_names: list[str], hp_values: tuple):
    cmd = [sys.executable, "-m", TRAINING_MODULE]
    for name, val in zip(hp_names, hp_values):
        cmd += [f"--{name}", str(val)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    subprocess.run(cmd, env=env)


def execute_process(gpu_queue: Queue, hp_names: list[str], hp_queue: Queue):
    while True:
        if not gpu_queue.empty():
            gpu_id = gpu_queue.get()

            if not hp_queue.empty():
                hp_values = hp_queue.get()
                logger.info(f"Starting training on GPU {gpu_id}")
                for name, value in zip(hp_names, hp_values):
                    logger.info(f"  {name}: {value}")
                try:
                    run_training_script(gpu_id, hp_names, hp_values)
                except Exception as e:
                    print(f"GPU {gpu_id} error:\n{e}\n")

                gpu_queue.put(gpu_id)
            else:
                break

        sleep(1)


# ─────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logger()

    num_gpus = torch.cuda.device_count()
    gpu_availables = list(range(num_gpus)) if num_gpus > 0 else [0]

    gpu_queue: Queue = Queue()
    for gpu_id in gpu_availables:
        gpu_queue.put(gpu_id)

    hp_names, hp_list = read_YAML_hyperparameters_sweep(HYPERPARAMETERS_PATH)
    logger.info(f"Number of configurations to test: {len(hp_list)}")

    if SWEEP_MODE == "random":
        random.shuffle(hp_list)

    hp_queue: Queue = Queue()
    for config in hp_list:
        hp_queue.put(config)

    processes = [
        Process(target=execute_process, args=(gpu_queue, hp_names, hp_queue))
        for _ in gpu_availables
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
