"""Loader for pre-generated CVRP problem files.

Files are produced by `scripts/generate_problems.py` and live in a directory
(default: `generated_problems/`). Each file holds `node_coords`, `demands`,
`capacity` for N pre-sampled instances. The loader slices the first
`n_problems` rows so the user can request a smaller eval set.
"""
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from tensordict import TensorDict


def _candidate_path(instances_dir: Path, dim: int, capacity: int) -> Path:
    return instances_dir / f"gen_dim{dim}_load{capacity}.pt"


def load_eval_instances(
    cfg: dict,
    device: torch.device,
    n_problems: Optional[int] = None,
) -> Optional[TensorDict]:
    """Return a TensorDict matching `cfg.eval.dim` and `cfg.problem.max_load`.

    Returns None if no matching file is found, so the caller can fall back to
    on-the-fly generation. If `n_problems` is None, uses `cfg.eval.n_problems`.
    """
    instances_dir = Path(cfg["eval"].get("instances_dir", "generated_problems"))
    dim = cfg["eval"]["dim"]
    max_load = cfg["problem"]["max_load"]
    n = n_problems if n_problems is not None else cfg["eval"]["n_problems"]

    path = _candidate_path(instances_dir, dim, max_load)
    if not path.exists():
        logger.info(
            f"No pre-generated eval file at {path} — falling back to random generation."
        )
        return None

    blob = torch.load(path, map_location="cpu", weights_only=False)
    available = blob["node_coords"].shape[0]
    if n > available:
        logger.warning(
            f"Requested {n} eval problems but {path} only holds {available}. "
            f"Using all {available}."
        )
        n = available

    coords = blob["node_coords"][:n].to(device)
    demands = blob["demands"][:n].to(device).long()
    capacity = blob["capacity"][:n].to(device).long()

    logger.info(
        f"Loaded {n} eval problems from {path} (dim={dim}, capacity={max_load})."
    )
    return TensorDict(
        {"coords": coords, "demands": demands, "capacity": capacity},
        batch_size=[n],
        device=device,
    )
