"""Curriculum schedules and multi-init activation.

Ported from LGSA_OLD/src/main.py::calculate_curriculum_steps_sig / _lin
and the multi-init slicing in train_ppo.
"""
import math


def curriculum_pretrain_steps(step: int, cfg: dict) -> int:
    """Number of deterministic improvement steps T_init for this epoch.

    Sigmoid or linear schedule depending on cfg.curriculum.type.
    """
    xi_cl = cfg["curriculum"]["max_outer_steps"]
    E = cfg["curriculum"]["max_prob_step"]
    if cfg["curriculum"]["type"] == "sig":
        kappa_base = cfg["curriculum"]["steepness"]
        kappa = kappa_base * (200 / max(1, E))

        def sigmoid(x):
            e = max(-20.0, min(20.0, -kappa * (x - E / 2.0)))
            return 1.0 / (1.0 + math.exp(e))

        s0, sE, se = sigmoid(0), sigmoid(E), sigmoid(step)
        denom = (sE - s0)
        progress = (se - s0) / denom if denom != 0 else 1.0
    else:
        progress = step / max(1, E)
    t_init = int(progress * xi_cl)
    return max(0, min(t_init, xi_cl))


def active_init_methods(step: int, cfg: dict) -> list[str]:
    """Multi-init expansion schedule: start with 1 method, linearly add up to all."""
    init_list = cfg["sa"]["init_list"]
    if not cfg["sa"]["multi_init"]:
        return [cfg["sa"]["init"]]
    total = len(init_list)
    target = cfg["sa"].get("multi_init_step", 0)
    if target <= 0:
        return init_list[:]
    progress = min(1.0, step / target)
    num_active = 1 + int(progress * (total - 1))
    return init_list[:num_active]
