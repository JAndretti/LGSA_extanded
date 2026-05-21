"""SA outer loop with Metropolis acceptance.

Drives the actor through `total_steps` of move-propose / accept-or-reject.
When `train=True`, accumulates per-step transition TDs in a list and returns
the stacked `(T, B, ...)` TensorDict under `results["transitions"]` for the
PPO update (no torchrl replay buffer involved).

Numerically sensitive parts (Metropolis, cost, reward) run in fp32 regardless
of the surrounding autocast context.
"""

from contextlib import nullcontext
from typing import Callable, Tuple

import torch
from tensordict import TensorDict

from src.cvrp.features import build_features
from src.cvrp.heuristics import apply_move
from src.cvrp.state import cost, cost_with_solution, update_state
from src.sa.reward import compute_reward


def metropolis_accept(
    cost_improvement: torch.Tensor, temp: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Metropolis acceptance. Inputs must be fp32."""
    accept_prob = torch.minimum(
        torch.exp(cost_improvement / temp.squeeze(-1).clamp(min=1e-10)),
        torch.ones_like(cost_improvement),
    )
    rand = torch.rand_like(accept_prob)
    accepted = (rand < accept_prob).long()
    actual_improvement = cost_improvement * accepted
    return accepted, actual_improvement


def _scale_to_unit(value: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    span = vmax - vmin
    if span == 0:
        return torch.zeros_like(value)
    return (value - vmin) / span


def sa_collect(
    actor,
    state: TensorDict,
    temperature_schedule: Callable[[int], torch.Tensor],
    cfg: dict,
    total_steps: int,
    *,
    train: bool,
    epoch: int = 0,
    greedy: bool = False,
    autocast_ctx=None,
) -> dict:
    """Run `total_steps` of SA on the given state.

    Args:
        actor: CVRPActor.
        state: enriched state TD; mutated in-place.
        temperature_schedule: callable step -> scalar temperature tensor.
        cfg: full config dict (heuristic, update_method, feature_flags, reward args).
        total_steps: number of SA steps to run.
        train: if True, the returned results include a 'transitions' TD of
            shape (T, B, ...) containing the per-step transitions for PPO.
        epoch: outer training epoch (used by curriculum reward modes).
        greedy: if True, actor uses argmax (eval mode).
        autocast_ctx: optional autocast context for the actor forward.
    """
    device = state.device
    init_temp = cfg["sa"]["init_temp"]
    stop_temp = cfg["sa"]["stop_temp"]
    heuristic = cfg["problem"]["heuristic"]
    update_method = cfg["problem"]["update_method"]
    flags = cfg["model"]["feature_flags"]
    n_problems = state.batch_size[0]
    actor_ctx = autocast_ctx if autocast_ctx is not None else nullcontext()

    raw_temp = torch.full((n_problems,), float(init_temp), device=device)
    state.set("temp", torch.full((n_problems, 1), 1.0, device=device))
    state.set("progress", torch.full((n_problems, 1), 1.0, device=device))
    state.set("observation", build_features(state, flags))

    best_solution = state["solution"].clone()
    best_cost = cost(state).float()
    initial_cost = best_cost.clone()
    current_cost = best_cost.clone()
    best_step = torch.zeros(n_problems, dtype=torch.long, device=device)

    n_accepted = torch.zeros(n_problems, device=device)
    valid_sum = 0.0
    rewards_sum = torch.zeros(n_problems, device=device)
    transitions: list[TensorDict] = []

    for step in range(total_steps):
        with actor_ctx, torch.no_grad():
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
        improvement = (current_cost - new_cost).float()
        if cfg["sa"]["metropolis"]:
            accepted, actual_improvement = metropolis_accept(
                improvement, raw_temp
            )
        else:
            accepted = torch.ones_like(improvement, dtype=torch.long)
            actual_improvement = improvement

        accepted_e = accepted.view(-1, 1, 1).expand_as(new_solution)
        accepted_sol = torch.where(accepted_e == 1, new_solution, state["solution"])
        update_state(state, accepted_sol.long())
        current_cost = torch.where(accepted == 1, new_cost, current_cost)

        is_imp = current_cost < best_cost
        old_best_cost = best_cost.clone()
        best_solution = torch.where(
            is_imp.view(-1, 1, 1).expand_as(accepted_sol),
            accepted_sol,
            best_solution,
        ).long()
        best_cost = torch.minimum(current_cost, best_cost)
        best_step = torch.where(is_imp, torch.full_like(best_step, step + 1), best_step)

        next_temp = temperature_schedule(step).to(device).float()
        raw_temp = (
            next_temp.expand(n_problems) if next_temp.dim() == 0 else next_temp
        )
        state.set("temp", _scale_to_unit(raw_temp, stop_temp, init_temp).view(-1, 1))
        state.set(
            "progress",
            torch.full(
                (n_problems, 1), 1.0 - step / max(1, total_steps), device=device
            ),
        )
        next_obs = build_features(state, flags)

        reward = compute_reward(
            mode=cfg["sa"]["reward_mode"],
            actual_improvement=actual_improvement.float(),
            accepted=accepted,
            valid=valid,
            best_cost=best_cost,
            old_best_cost=old_best_cost,
            current_cost=current_cost,
            initial_cost=initial_cost,
            normalize_reward=cfg["sa"]["normalize_reward"],
            hybrid_alpha=cfg["sa"]["hybrid_alpha"],
            reward_scale=cfg["sa"]["reward_scale"],
            epoch=epoch,
            n_epochs_for_curriculum=cfg["training"]["num_epochs"],
            step=step,
            total_steps=total_steps,
            last_step=(step + 1 == total_steps),
        )

        if train:
            is_last = step + 1 == total_steps
            transitions.append(
                TensorDict(
                    {
                        "observation": state["observation"].clone(),
                        "action": state["action"].clone(),
                        "action_mask": state["action_mask"].clone(),
                        "sample_log_prob": state["sample_log_prob"].clone(),
                        "next": TensorDict(
                            {
                                "observation": next_obs.clone(),
                                "reward": reward.clone(),
                                "done": torch.full(
                                    (n_problems, 1),
                                    is_last,
                                    dtype=torch.bool,
                                    device=device,
                                ),
                            },
                            batch_size=[n_problems],
                        ),
                    },
                    batch_size=[n_problems],
                )
            )

        state.set("observation", next_obs)
        n_accepted = n_accepted + accepted.float()
        valid_sum += valid.float().mean().item()
        rewards_sum = rewards_sum + reward.squeeze(-1)

    results = {
        "best_solution": best_solution,
        "best_cost": best_cost,
        "init_cost": initial_cost,
        "n_accepted": n_accepted,
        "n_valid_pct": valid_sum / max(1, total_steps),
        "avg_reward": rewards_sum.mean(),
        "best_step": best_step.float(),
    }
    if train and transitions:
        results["transitions"] = torch.stack(transitions, dim=0)
    return results
