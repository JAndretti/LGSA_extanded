"""Inference-side helpers: seed control, actor construction from a saved run,
SA wrapper, D4 augmentation, and a uniform-sampling baseline actor.

Pure inference — no training-time logic touched.
"""
from __future__ import annotations

import contextlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.actor import CVRPActor
from src.model import build_actor
from src.model.base import apply_mask_c1, apply_mask_c2
from src.sa.inference import sa_infer

from eval.eval_io import load_actor_state


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warmup_cuda() -> None:
    if torch.cuda.is_available():
        torch.empty(1, device="cuda").uniform_()
        torch.cuda.synchronize()


def autocast_for(device: torch.device, dtype: torch.dtype):
    """Return a context manager: autocast(device, dtype) or nullcontext."""
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def build_actor_from_config(
    run_cfg: dict,
    ckpt_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> CVRPActor:
    """Construct a CVRPActor for a saved run and load its best checkpoint.

    Reads run_cfg["model"] + run_cfg["problem"]["update_method"] to size the
    net (via src.model.build_actor), then loads the actor state from
    ckpt_path. Returns the actor in eval() mode, cast to `dtype`.
    """
    actor = build_actor(run_cfg, device)
    state_dict = load_actor_state(ckpt_path, device)
    actor.load_state_dict(state_dict, strict=True)
    actor = actor.to(dtype)
    actor.eval()
    return actor


# ─────────────────────────────────────────────────────────────
#  D4 augmentation (8 isometries of the unit square)
# ─────────────────────────────────────────────────────────────

_D4 = [
    lambda p: p,
    lambda p: torch.stack([1 - p[..., 0],     p[..., 1]], dim=-1),  # flip x
    lambda p: torch.stack([    p[..., 0], 1 - p[..., 1]], dim=-1),  # flip y
    lambda p: torch.stack([1 - p[..., 0], 1 - p[..., 1]], dim=-1),  # flip both
    lambda p: torch.stack([    p[..., 1],     p[..., 0]], dim=-1),  # swap
    lambda p: torch.stack([1 - p[..., 1],     p[..., 0]], dim=-1),
    lambda p: torch.stack([    p[..., 1], 1 - p[..., 0]], dim=-1),
    lambda p: torch.stack([1 - p[..., 1], 1 - p[..., 0]], dim=-1),
]


def augment_coords(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Apply the k-th D4 isometry. k ∈ [0..7]."""
    return _D4[k](coords)


# ─────────────────────────────────────────────────────────────
#  Uniform baseline actor (matches CVRPActor I/O contract)
# ─────────────────────────────────────────────────────────────


class UniformActor(nn.Module):
    """Drop-in replacement for CVRPActor that samples uniformly over the valid
    mask. Used by the baseline arm. Writes the same out_keys as CVRPActor.

    Matches the masking regime currently wired in train.py: no
    `get_action_mask_fn` (so method='valid' falls through to 'free' semantics
    in apply_mask_c2).

    No silent fallback: if a row has zero valid c1 or c2 positions, the
    underlying `torch.multinomial` will raise, surfacing the upstream bug
    (degenerate state) rather than emitting an infeasible action.
    """

    def __init__(self, method: str):
        super().__init__()
        self.method = method

    def forward(self, td: TensorDict, greedy: bool = False) -> TensorDict:
        obs = td["observation"]
        x = obs[..., :1].long()
        B, L, _ = obs.shape
        device = obs.device

        c1_logits = torch.zeros(B, L, device=device)
        c1_logits, valid1 = apply_mask_c1(c1_logits, x, self.method)
        probs1 = torch.softmax(c1_logits, dim=-1)
        c1 = torch.multinomial(probs1, 1).squeeze(-1)

        c2_logits = torch.zeros(B, L, device=device)
        c2_logits, valid2 = apply_mask_c2(
            c2_logits, c1, self.method, B, external_mask=None
        )
        probs2 = torch.softmax(c2_logits, dim=-1)
        c2 = torch.multinomial(probs2, 1).squeeze(-1)

        action = torch.stack([c1.long(), c2.long()], dim=-1)
        n_valid1 = valid1.sum(dim=-1).clamp(min=1).float()
        n_valid2 = valid2.sum(dim=-1).clamp(min=1).float()
        log_prob = -(n_valid1.log() + n_valid2.log())

        td.set("action", action)
        td.set("sample_log_prob", log_prob)
        td.set("entropy", n_valid1.log() + n_valid2.log())
        td.set("action_mask", valid2)
        return td


def strip_enrichment(static_td: TensorDict) -> TensorDict:
    """Return a clone of `static_td` with derived geometry keys removed so
    `init_state` recomputes them. Used by the D4 augmentation loop in
    handler_random when the underlying coords change.
    """
    out = static_td.clone()
    for k in ("distance_matrix", "angles", "mean_dist_k",
              "dist_to_depot", "demand_normalized"):
        if k in out.keys():
            del out[k]
    return out


def run_sa(
    actor: nn.Module,
    state: TensorDict,
    cfg: dict,
    *,
    outer_steps: int,
    greedy: bool,
    metropolis: bool,
    autocast_ctx=None,
) -> dict:
    """Thin wrapper around src.sa.inference.sa_infer. Returns {best_solution, best_cost}."""
    from src.sa.temperature import make_schedule

    sched = make_schedule(
        cfg["sa"]["schedule"],
        T_max=cfg["sa"]["init_temp"],
        T_min=cfg["sa"]["stop_temp"],
        step_max=outer_steps,
    )
    return sa_infer(
        actor, state, sched, cfg,
        total_steps=outer_steps,
        greedy=greedy, metropolis=metropolis,
        autocast_ctx=autocast_ctx,
    )
