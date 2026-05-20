"""Filesystem layer for the eval harness.

Discovers W&B runs, loads their saved configs and checkpoints, writes the
final CSV. Pure I/O — no torch model logic.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from loguru import logger


# ─────────────────────────────────────────────────────────────
#  Flatten + diff helpers (shared across handlers)
# ─────────────────────────────────────────────────────────────


def _flatten_cfg(d: dict, parent: str = "", sep: str = ".") -> dict:
    """Flatten a nested cfg to dotted-path keys. Lists become tuples (hashable)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten_cfg(v, key, sep))
        elif isinstance(v, list):
            out[key] = tuple(v)
        else:
            out[key] = v
    return out


def _differing_keys(cfgs: list[dict]) -> set[str]:
    """Keys whose flattened values differ across cfgs."""
    if not cfgs:
        return set()
    flat = [_flatten_cfg(c) for c in cfgs]
    keys = set().union(*flat)
    diff = set()
    for k in keys:
        vals = {f.get(k) for f in flat}
        if len(vals) > 1:
            diff.add(k)
    return diff


def _hp_row(cfg: dict, keys: set[str]) -> dict:
    """Project a cfg onto a key set (flatten then select)."""
    flat = _flatten_cfg(cfg)
    return {k: flat.get(k) for k in keys}


# ─────────────────────────────────────────────────────────────
#  Run discovery
# ─────────────────────────────────────────────────────────────


def find_run_dirs(project: str, group: str) -> list[Path]:
    """Return wandb/<project>/<group>/wandb/{run,offline-run}-*/files/ dirs.

    Both online (run-*) and offline (offline-run-*) dirs are included. Dirs
    lacking config.yaml are filtered with a warning. Dirs lacking
    top_checkpoints/ are kept — the per-run loop falls back to
    latest_checkpoint.pt and warns there if both are missing.
    """
    group_dir = Path("wandb") / project / group / "wandb"
    if not group_dir.is_dir():
        return []

    candidates = sorted(
        list(group_dir.glob("run-*/files"))
        + list(group_dir.glob("offline-run-*/files"))
    )
    valid: list[Path] = []
    for c in candidates:
        if not (c / "config.yaml").exists():
            logger.warning(f"Skipping {c.parent.name}: no config.yaml")
            continue
        valid.append(c)
    return valid


def load_run_config(run_files_dir: Path) -> dict:
    """Read config.yaml saved by wandb.init and unwrap the wandb format.

    wandb wraps each section in {value: V} and adds a _wandb metadata key.
    Returns a dict shaped exactly like HP.yaml.
    """
    with open(run_files_dir / "config.yaml") as f:
        raw = yaml.safe_load(f) or {}
    unwrapped: dict[str, Any] = {}
    for k, v in raw.items():
        if k == "_wandb":
            continue
        if isinstance(v, dict) and "value" in v and len(v) == 1:
            unwrapped[k] = v["value"]
        else:
            unwrapped[k] = v
    return unwrapped


# ─────────────────────────────────────────────────────────────
#  Checkpoint discovery + actor state loader
# ─────────────────────────────────────────────────────────────


_CKPT_RE = re.compile(
    r"^epoch_(\d{4})_(loss|reward)_(\d+(?:\.\d+)?)\.pt$"
)


def _parsed_ckpt_value(path: Path) -> tuple[float, str] | None:
    """Return (value, tag) parsed from `epoch_NNNN_{tag}_V.pt` or None if unparseable."""
    m = _CKPT_RE.match(path.name)
    if m is None:
        return None
    return float(m.group(3)), m.group(2)


def best_checkpoint_path(
    run_files_dir: Path, mode: str | None = None
) -> Path:
    """Return the best checkpoint for a run.

    Looks in <run>/files/top_checkpoints/ first. `mode` is read from
    the run's config when None (defaults to "min" if absent).
      mode="min" — pick the smallest parsed value (loss).
      mode="max" — pick the largest parsed value (reward).
    Falls back to <run>/files/latest_checkpoint.pt if top_checkpoints/ is
    empty or all filenames are unparseable. Raises FileNotFoundError if
    neither exists.
    """
    if mode is None:
        try:
            cfg = load_run_config(run_files_dir)
            mode = cfg.get("checkpointing", {}).get("mode", "min")
        except FileNotFoundError:
            mode = "min"

    top_dir = run_files_dir / "top_checkpoints"
    parsed: list[tuple[float, Path]] = []
    if top_dir.is_dir():
        for p in top_dir.glob("epoch_*.pt"):
            parsed_val = _parsed_ckpt_value(p)
            if parsed_val is not None:
                parsed.append((parsed_val[0], p))

    if parsed:
        key = min if mode == "min" else max
        return key(parsed, key=lambda x: x[0])[1]

    latest = run_files_dir / "latest_checkpoint.pt"
    if latest.exists():
        return latest
    raise FileNotFoundError(
        f"No checkpoint in {run_files_dir} (neither parseable top_checkpoints/* "
        f"nor latest_checkpoint.pt)"
    )


def load_actor_state(ckpt_path: Path, device: torch.device) -> dict:
    """Load the actor state_dict from a bundle checkpoint.

    Expects the new format written by src.utils.save_checkpoint:
      ckpt["models"]["actor"] -> state_dict

    Uses `weights_only=False` because `save_checkpoint` also serialises the
    numpy + python RNG state (under `rng_numpy` / `rng_python`), which pulls
    in numpy dtype globals (UInt32DType, etc.) plus tuples of built-ins —
    not reachable via torch's default safe-globals. The checkpoint format
    is owned by this codebase and not consumed from untrusted sources, so
    arbitrary-pickle execution is not a real risk here. Raises KeyError
    with a clear message if the format is unrecognised.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        return ckpt["models"]["actor"]
    except (KeyError, TypeError) as e:
        raise KeyError(
            f"Checkpoint {ckpt_path} does not have ckpt['models']['actor']; "
            f"got top-level keys {list(ckpt) if isinstance(ckpt, dict) else type(ckpt)}"
        ) from e


# ─────────────────────────────────────────────────────────────
#  Results CSV writer
# ─────────────────────────────────────────────────────────────


def save_results_csv(
    df: pd.DataFrame,
    project: str,
    group: str,
    filename_stem: str,
    base_dir: Path = Path("results"),
) -> Path:
    """Write df to <base_dir>/<project>/<group>/<stem>.csv with auto-versioning.

    If <stem>.csv exists, writes <stem>_2.csv, then <stem>_3.csv, ... — never
    overwrites. Creates parent dirs. Returns the path written.
    """
    out_dir = base_dir / project / group
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{filename_stem}.csv"
    if path.exists():
        i = 2
        while (out_dir / f"{filename_stem}_{i}.csv").exists():
            i += 1
        path = out_dir / f"{filename_stem}_{i}.csv"
    df.to_csv(path, index=False)
    return path
