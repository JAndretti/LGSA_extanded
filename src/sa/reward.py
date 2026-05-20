"""Reward modes for the SA-PPO hybrid.

Each mode is a private function. All operate on FP32 tensors and return (B, 1).
Ported from LGSA_OLD/src/sa/sa_train.py::calculate_reward.
"""

import torch


_MAX_REL_IMPROVEMENT = 0.2


def _normalize(
    actual_improvement: torch.Tensor, initial_cost: torch.Tensor
) -> torch.Tensor:
    return torch.clamp(
        actual_improvement / initial_cost / _MAX_REL_IMPROVEMENT, -1.0, 1.0
    )


def _immediate(*, actual_improvement, valid, initial_cost, normalize_reward, **_):
    v = (
        _normalize(actual_improvement, initial_cost)
        if normalize_reward
        else actual_improvement
    )
    return torch.where(
        ~valid.bool().squeeze(-1),
        torch.full_like(v, -1.5),
        torch.where(actual_improvement == 0, torch.zeros_like(v), v),
    ).view(-1, 1)


def _sa_aligned(
    *, actual_improvement, accepted, valid, initial_cost, normalize_reward, **_
):
    v = (
        _normalize(actual_improvement, initial_cost)
        if normalize_reward
        else actual_improvement
    )
    acc = accepted.bool().view(-1)
    sa_r = torch.where(
        acc & (actual_improvement > 0),
        v,
        torch.where(
            acc & (actual_improvement <= 0),
            torch.zeros_like(v),
            torch.full_like(v, -0.1),
        ),
    )
    return torch.where(
        ~valid.bool().squeeze(-1), torch.full_like(sa_r, -1.5), sa_r
    ).view(-1, 1)


def _global_best(
    *, best_cost, old_best_cost, initial_cost, normalize_reward, reward_scale=1.0, **_
):
    delta = old_best_cost - best_cost
    v = _normalize(delta, initial_cost) if normalize_reward else delta
    r = torch.where(delta > 0, v, torch.zeros_like(delta)).view(-1, 1)
    return r * reward_scale


def _hybrid(
    *,
    actual_improvement,
    valid,
    best_cost,
    old_best_cost,
    initial_cost,
    normalize_reward,
    hybrid_alpha,
    **_,
):
    imm = _immediate(
        actual_improvement=actual_improvement,
        valid=valid,
        initial_cost=initial_cost,
        normalize_reward=normalize_reward,
    )
    glob = _global_best(
        best_cost=best_cost,
        old_best_cost=old_best_cost,
        initial_cost=initial_cost,
        normalize_reward=normalize_reward,
        reward_scale=1.0,
    )
    return hybrid_alpha * imm + (1 - hybrid_alpha) * glob


def _curriculum(
    *,
    epoch,
    n_epochs_for_curriculum,
    actual_improvement,
    accepted,
    valid,
    best_cost,
    old_best_cost,
    initial_cost,
    normalize_reward,
    **_,
):
    progress = (epoch - 1) / max(1, n_epochs_for_curriculum - 1)
    sa_part = _sa_aligned(
        actual_improvement=actual_improvement,
        accepted=accepted,
        valid=valid,
        initial_cost=initial_cost,
        normalize_reward=normalize_reward,
    )
    glob = _global_best(
        best_cost=best_cost,
        old_best_cost=old_best_cost,
        initial_cost=initial_cost,
        normalize_reward=normalize_reward,
        reward_scale=1.0,
    )
    return (1 - progress) * sa_part + progress * glob


def _curriculum_terminal(
    *,
    epoch,
    n_epochs_for_curriculum,
    actual_improvement,
    accepted,
    valid,
    best_cost,
    initial_cost,
    normalize_reward,
    last_step,
    **_,
):
    progress = (epoch - 1) / max(1, n_epochs_for_curriculum - 1)
    sa_part = _sa_aligned(
        actual_improvement=actual_improvement,
        accepted=accepted,
        valid=valid,
        initial_cost=initial_cost,
        normalize_reward=normalize_reward,
    )
    terminal = torch.zeros_like(sa_part)
    if last_step:
        terminal = ((initial_cost - best_cost) / initial_cost).view(-1, 1)
    return (1 - progress) * sa_part + progress * terminal


def _step_penalty_init(*, best_cost, initial_cost, **_):
    return -(best_cost / initial_cost).view(-1, 1)


def _step_penalty_16(*, best_cost, **_):
    return -(best_cost / 16.0).view(-1, 1)


def _terminal(*, best_cost, initial_cost, last_step, **_):
    if not last_step:
        return torch.zeros_like(best_cost).view(-1, 1)
    return ((initial_cost - best_cost) / initial_cost).view(-1, 1)


_REGISTRY = {
    "immediate": _immediate,
    "sa_aligned": _sa_aligned,
    "global_best": _global_best,
    "dact": _global_best,  # alias from old code
    "hybrid": _hybrid,
    "curriculum": _curriculum,
    "curriculum_terminal": _curriculum_terminal,
    "terminal": _terminal,
    "step_penalty_init": _step_penalty_init,
    "step_penalty_16": _step_penalty_16,
}


def compute_reward(mode: str, **kwargs) -> torch.Tensor:
    """Dispatch to one of the registered reward modes. Returns (B, 1)."""
    try:
        fn = _REGISTRY[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown reward mode: {mode}") from exc
    return fn(**kwargs)
