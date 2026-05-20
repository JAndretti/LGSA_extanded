import ast
import datetime
import heapq
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from loguru import logger

import wandb

# ─────────────────────────────────────────────────────────────
#  Logger Setup
# ─────────────────────────────────────────────────────────────


def setup_logger(log_dir: str | None = None) -> Path | None:
    """
    Configure loguru. Call once at the start of a script.

    Console: coloured, INFO and above (always enabled).
    File:    all levels, one file per run named by timestamp.
             Pass log_dir to enable; omit (or pass None) to skip.

    Returns the log file path, or None if no file sink was created.
    """
    logger.remove()  # drop the default stderr sink

    # Console sink — coloured, INFO+
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}:{line}</cyan> — <level>{message}</level>"
        ),
    )

    if log_dir is None:
        return None

    # File sink — all levels, one file per run
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"run_{timestamp}.log"

    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        encoding="utf-8",
    )

    logger.info(f"Logging to {log_file}")
    return log_file


# ─────────────────────────────────────────────────────────────
#  Script Argument Parsing
# ─────────────────────────────────────────────────────────────


def get_script_arguments(keys: list[str] | None = None) -> dict:
    """Parse --key value pairs from sys.argv.

    If keys is provided, only those keys are returned.
    If keys is None, all --key value pairs are parsed.
    Keys may use dotted paths (e.g. "optimizer.lr") for nested configs.

    Values are parsed as Python literals (ints, floats, bools, None, lists, ...).
    Lowercase `true`/`false` are also accepted as booleans (since shells typically
    pass them in that form).
    """
    args = {}
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg.startswith("--") and i + 1 < len(argv):
            name = arg[2:]
            if keys is None or name in keys:
                value_str = argv[i + 1]
                if value_str in ("true", "false"):
                    value = value_str == "true"
                else:
                    try:
                        value = ast.literal_eval(value_str)
                    except (ValueError, SyntaxError):
                        value = value_str
                args[name] = value
    return args


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply dotted-path overrides to a nested config dict in-place.

    Example:
        apply_overrides(cfg, {"optimizer.lr": 1e-4, "training.seed": 1337})
        # sets cfg["optimizer"]["lr"] = 1e-4 and cfg["training"]["seed"] = 1337
    """
    for key, value in overrides.items():
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            if part not in node:
                raise KeyError(
                    f"Override key '{key}': section '{part}' not found in config."
                )
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(
                f"Override key '{key}': field '{parts[-1]}' not found in config."
            )
        node[parts[-1]] = value
    return cfg


# ─────────────────────────────────────────────────────────────
#  Top-K Checkpoint Manager
# ─────────────────────────────────────────────────────────────


class TopKCheckpointManager:
    """Keeps the K best model checkpoints, tracked via a min-heap.

    The heap root is always the WORST kept checkpoint so eviction is O(log k):
      mode="min": score = -value  → root = most negative = largest loss  = worst
      mode="max": score = +value  → root = smallest value = smallest reward = worst

    In both cases a new checkpoint enters the top-K when score > heap[0] (it beats
    the current worst), and the worst is evicted to make room.
    """

    def __init__(self, top_k: int, mode: str = "min"):
        self.top_k = top_k
        self.mode = mode
        self.heap: list[tuple[float, str]] = []
        self.save_dir: Path | None = None
        self._best_score: float | None = None  # best score seen across all updates

    def setup(self):
        assert wandb.run is not None, "WandB was not initialized!"
        self.save_dir = Path(wandb.run.dir) / "top_checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[TopKCheckpoint] Saving to: {self.save_dir}")

    def _score(self, value: float) -> float:
        # Higher score always means better checkpoint.
        # min mode: lower loss is better → negate so lower loss → higher score
        # max mode: higher reward is better → keep as-is
        return -value if self.mode == "min" else value

    def update(
        self,
        epoch: int,
        value: float,
        *,
        models: dict[str, nn.Module],
        optimizers: dict[str, torch.optim.Optimizer] | None = None,
        schedulers: dict[str, "torch.optim.lr_scheduler.LRScheduler"] | None = None,
        extra: dict | None = None,
    ) -> bool:
        if self.save_dir is None:
            raise RuntimeError("Call TopKCheckpointManager.setup() before update().")
        score = self._score(value)
        if self._best_score is None or score > self._best_score:
            self._best_score = score

        tag = "loss" if self.mode == "min" else "reward"
        filename = f"epoch_{epoch:04d}_{tag}_{value:.4f}.pt"
        filepath = self.save_dir / filename

        def _save():
            save_checkpoint(
                str(filepath),
                models=models, optimizers=optimizers,
                schedulers=schedulers, extra=extra, epoch=epoch,
            )

        if len(self.heap) < self.top_k:
            _save()
            heapq.heappush(self.heap, (score, str(filepath)))
            logger.info(
                f"[TopKCheckpoint] Saved ({len(self.heap)}/{self.top_k}): {filename}"
            )
            return True

        worst_score, worst_path = self.heap[0]
        if score > worst_score:
            _save()
            os.remove(worst_path)
            heapq.heapreplace(self.heap, (score, str(filepath)))
            logger.info(f"[TopKCheckpoint] Updated top-{self.top_k}: {filename}")
            return True

        return False

    def best_value(self) -> float | None:
        """Return the best value seen across all calls to update()."""
        if self._best_score is None:
            return None
        return -self._best_score if self.mode == "min" else self._best_score

    def log_to_wandb(self):
        artifact = wandb.Artifact(
            name="top-k-checkpoints",
            type="model",
            description=f"Top-{self.top_k} checkpoints (mode={self.mode})",
        )
        artifact.add_dir(str(self.save_dir))
        assert wandb.run is not None, "WandB was not initialized!"
        wandb.run.log_artifact(artifact)
        logger.info(f"[TopKCheckpoint] Logged artifact with {len(self.heap)} files.")


# ─────────────────────────────────────────────────────────────
#  Config Loading
# ─────────────────────────────────────────────────────────────


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).parent / "HP" / "HP.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
#  Reproducibility & Device
# ─────────────────────────────────────────────────────────────


def set_seed(seed: int, train: bool = True):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if train:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    # If your environment has its own RNG, seed it here too.
    # e.g. for a custom env:  env.seed(seed)  or  env.reset(seed=seed)


def save_checkpoint(
    path: str,
    *,
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
    schedulers: dict[str, "torch.optim.lr_scheduler.LRScheduler"] | None = None,
    extra: dict | None = None,
    epoch: int,
):
    """Save a full training checkpoint.

    Args:
        path: file to write.
        models: name -> nn.Module bundle (e.g. {"actor": actor, "critic": critic}).
        optimizers: name -> Optimizer bundle.
        schedulers: name -> LRScheduler bundle.
        extra: name -> any object exposing `.state_dict()`/`.load_state_dict()`
            (e.g. AdaptiveKLController). Stored as a dict of state dicts.
        epoch: last completed epoch.

    RNG states (torch, numpy, python, cuda) are always saved so resume is exact.
    """
    import random
    import numpy as np

    payload = {
        "epoch": epoch,
        "models": {name: m.state_dict() for name, m in models.items()},
        "optimizers": {n: o.state_dict() for n, o in (optimizers or {}).items()},
        "schedulers": {n: s.state_dict() for n, s in (schedulers or {}).items()},
        "extra": {n: e.state_dict() for n, e in (extra or {}).items()},
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    *,
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
    schedulers: dict[str, "torch.optim.lr_scheduler.LRScheduler"] | None = None,
    extra: dict | None = None,
    device: torch.device,
) -> int:
    """Restore a checkpoint saved by `save_checkpoint`. Returns the resume epoch."""
    import random
    import numpy as np

    # weights_only=False required: checkpoint contains Python RNG state objects.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for name, m in models.items():
        m.load_state_dict(ckpt["models"][name])
    for name, o in (optimizers or {}).items():
        o.load_state_dict(ckpt["optimizers"][name])
    for name, s in (schedulers or {}).items():
        s.load_state_dict(ckpt["schedulers"][name])
    for name, e in (extra or {}).items():
        e.load_state_dict(ckpt["extra"][name])
    if "rng_torch" in ckpt:
        torch.set_rng_state(ckpt["rng_torch"])
        np.random.set_state(ckpt["rng_numpy"])
        random.setstate(ckpt["rng_python"])
    if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    logger.info(f"Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


def get_device(request: str | None = None) -> torch.device:
    """Return a device.

    If request is set (e.g. from HP.yaml training.device), honour it directly.
    Otherwise auto-detect: cuda → mps → cpu.
    """
    if request and request != "auto":
        return torch.device(request)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """Return the appropriate autocast dtype for a given device."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device.type == "mps":
        return torch.float16
    else:
        return None
