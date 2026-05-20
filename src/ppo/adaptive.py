"""Adaptive KL / LR / β controller for PPO.

Tracks β_KL and adapts it + the actor LR based on observed KL divergence.
Logic ported verbatim from LGSA_OLD/src/ppo/ppo.py.
"""
from dataclasses import dataclass

import torch


@dataclass
class AdaptiveKLController:
    target_kl: float
    beta_kl: float
    beta_min: float = 1e-3
    beta_max: float = 10.0
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    beta_inc: float = 2.0
    beta_dec: float = 0.5
    lr_dec: float = 0.95
    lr_inc: float = 1.05

    def update(self, mean_kl: float, actor_opt: torch.optim.Optimizer) -> None:
        if mean_kl > 1.5 * self.target_kl:
            self.beta_kl = min(self.beta_kl * self.beta_inc, self.beta_max)
            for g in actor_opt.param_groups:
                g["lr"] = max(g["lr"] * self.lr_dec, self.lr_min)
        elif mean_kl < 0.5 * self.target_kl:
            self.beta_kl = max(self.beta_kl * self.beta_dec, self.beta_min)
            for g in actor_opt.param_groups:
                g["lr"] = min(g["lr"] * self.lr_inc, self.lr_max)

    def state_dict(self) -> dict:
        return {"beta_kl": self.beta_kl}

    def load_state_dict(self, sd: dict) -> None:
        self.beta_kl = sd["beta_kl"]
