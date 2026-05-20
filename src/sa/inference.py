"""Inference-only SA outer loop — stripped for speed.

Same Metropolis dynamics as `sa.loop.sa_collect`, but everything that exists
only to feed training is removed:
  - no reward computation
  - no transition collection / cloning
  - no per-step metric aggregation (n_accepted, valid_pct, reward sums)
  - no old_best_cost / best_step tracking

The hot per-step path keeps only what's strictly needed to advance the
Metropolis chain and maintain the per-problem best. Wrap calls in
`torch.inference_mode()` for the lowest overhead.
"""

from contextlib import nullcontext
from typing import Callable

import torch
from tensordict import TensorDict

from src.cvrp.features import build_features
from src.cvrp.heuristics import apply_move
from src.cvrp.state import cost, cost_with_solution, update_state


def _scale_to_unit(value: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    span = vmax - vmin
    if span == 0:
        return torch.zeros_like(value)
    return (value - vmin) / span


def _metropolis_accept_mask(
    cost_improvement: torch.Tensor, temp: torch.Tensor
) -> torch.Tensor:
    """Boolean accept mask. Drops the `actual_improvement` return (unused here)."""
    accept_prob = torch.minimum(
        torch.exp(cost_improvement / temp.squeeze(-1).clamp(min=1e-10)),
        torch.ones_like(cost_improvement),
    )
    return torch.rand_like(accept_prob) < accept_prob


def sa_infer(
    actor,
    state: TensorDict,
    temperature_schedule: Callable[[int], torch.Tensor],
    cfg: dict,
    total_steps: int,
    *,
    greedy: bool = True,
    metropolis: bool = True,
    autocast_ctx=None,
) -> dict:
    """Run `total_steps` of SA on `state` and return only the per-problem best.

    Args:
        actor: CVRPActor (or any compatible TensorDictModule).
        state: enriched state TD; mutated in-place.
        temperature_schedule: step -> scalar temperature tensor.
        cfg: full config dict (heuristic, update_method, feature_flags, init/stop temp).
        total_steps: number of SA steps to run.
        greedy: if True, actor takes the argmax instead of sampling.
        metropolis: if False, every proposed move is accepted (pure descent
            when combined with greedy=True).
        autocast_ctx: optional autocast context for the actor forward.

    Returns:
        {"best_solution": (B, L, 1) int64, "best_cost": (B,) fp32}
    """
    device = state.device
    init_temp = cfg["sa"]["init_temp"]
    stop_temp = cfg["sa"]["stop_temp"]
    heuristic = cfg["problem"]["heuristic"]
    update_method = cfg["problem"]["update_method"]
    flags = cfg["model"]["feature_flags"]
    n_problems = state.batch_size[0]
    actor_ctx = autocast_ctx if autocast_ctx is not None else nullcontext()

    state.set("temp", torch.full((n_problems, 1), 1.0, device=device))
    state.set("progress", torch.full((n_problems, 1), 1.0, device=device))
    state.set("observation", build_features(state, flags))

    best_solution = state["solution"].clone()
    best_cost = cost(state).float()
    current_cost = best_cost.clone()

    for step in range(total_steps):
        with actor_ctx:
            actor(state, greedy=greedy)

        new_solution, valid = apply_move(
            state, state["action"], heuristic, update_method
        )
        new_solution = torch.where(
            (valid == 1).view(-1, 1, 1).expand_as(new_solution),
            new_solution,
            state["solution"],
        ).long()

        new_cost = cost_with_solution(state, new_solution).float()
        if metropolis:
            improvement = (current_cost - new_cost).float()
            accepted = _metropolis_accept_mask(improvement, state["temp"].float())
        else:
            accepted = torch.ones(n_problems, dtype=torch.bool, device=device)

        accepted_e = accepted.view(-1, 1, 1).expand_as(new_solution)
        accepted_sol = torch.where(accepted_e, new_solution, state["solution"])
        update_state(state, accepted_sol.long())
        current_cost = torch.where(accepted, new_cost, current_cost)

        is_imp = current_cost < best_cost
        best_solution = torch.where(
            is_imp.view(-1, 1, 1).expand_as(accepted_sol),
            accepted_sol,
            best_solution,
        ).long()
        best_cost = torch.minimum(current_cost, best_cost)

        next_temp = temperature_schedule(step).to(device).float()
        next_temp_v = (
            next_temp.expand(n_problems) if next_temp.dim() == 0 else next_temp
        )
        state.set("temp", _scale_to_unit(next_temp_v, stop_temp, init_temp).view(-1, 1))
        state.set(
            "progress",
            torch.full(
                (n_problems, 1), 1.0 - step / max(1, total_steps), device=device
            ),
        )
        state.set("observation", build_features(state, flags))

    return {"best_solution": best_solution, "best_cost": best_cost}
