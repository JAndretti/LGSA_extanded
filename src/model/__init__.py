"""Factory for building actor + critic from a config dict."""
from typing import Callable, Optional

import torch

from src.cvrp.features import input_dim
from src.model.actor import CVRPActor
from src.model.critic import CVRPCritic


def build_actor(
    cfg: dict,
    device: torch.device,
    get_action_mask_fn: Optional[Callable] = None,
) -> CVRPActor:
    flags = cfg["model"]["feature_flags"]
    c = input_dim(flags)
    return CVRPActor(
        c=c,
        embed_dim=cfg["model"]["embed_dim"],
        num_hidden_layers=cfg["model"]["num_hidden_layers"],
        method=cfg["problem"]["update_method"],
        device=device,
        get_action_mask_fn=get_action_mask_fn,
    )


def build_critic(cfg: dict, device: torch.device) -> CVRPCritic:
    flags = cfg["model"]["feature_flags"]
    c = input_dim(flags)
    return CVRPCritic(
        c=c,
        embed_dim=cfg["model"]["embed_dim"],
        num_hidden_layers=cfg["model"]["num_hidden_layers"],
        device=device,
    )
