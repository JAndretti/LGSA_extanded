"""Shared building blocks for the actor and critic."""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def build_mlp(
    input_dim: int,
    embed_dim: int,
    num_hidden_layers: int,
    device: str | torch.device,
    output_dim: int = 1,
) -> nn.Sequential:
    """Linear+LeakyReLU entry, N hidden layers (Linear+LeakyReLU), Linear output."""
    layers: list[nn.Module] = [
        nn.Linear(input_dim, embed_dim, bias=True, device=device),
        nn.LeakyReLU(),
    ]
    for _ in range(num_hidden_layers):
        layers.extend([
            nn.Linear(embed_dim, embed_dim, bias=True, device=device),
            nn.LeakyReLU(),
        ])
    layers.append(nn.Linear(embed_dim, output_dim, bias=False, device=device))
    return nn.Sequential(*layers).to(device)


def init_orthogonal(m: nn.Module) -> None:
    """Orthogonal init for Linear weights, zero bias."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def sample_from_logits(
    logits: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    greedy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample an action per row. Returns (action_idx (B,), log_prob (B, 1))."""
    probs = torch.softmax(logits, dim=-1)
    if greedy:
        idx = torch.argmax(probs, dim=-1)
    else:
        idx = torch.multinomial(probs, 1, generator=generator)[..., 0]
    taken = probs.gather(1, idx.view(-1, 1))
    return idx, torch.log(taken.clamp(min=1e-20))


def apply_mask_c1(
    logits: torch.Tensor, x: torch.Tensor, method: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask city-1 candidates. Returns (logits, valid_mask) with valid_mask: (B, L) bool.

    x: (B, L, 1) — solution index column. Depot (x==0) and last index are forbidden
    unless method='rm_depot' (compacted-TSP mode).
    """
    valid = torch.ones(logits.shape, dtype=torch.bool, device=logits.device)
    if method != "rm_depot":
        non_depot = (x != 0).squeeze(-1)
        logits = logits.masked_fill(~non_depot, float("-inf"))
        logits = logits.clone()
        logits[:, 0] = float("-inf")
        logits[:, -1] = float("-inf")
        valid = non_depot.clone()
        valid[:, 0] = False
        valid[:, -1] = False
    return logits, valid


def apply_mask_c2(
    logits: torch.Tensor,
    c1: torch.Tensor,
    method: str,
    n_problems: int,
    external_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask city-2 candidates. external_mask used only when method='valid'."""
    valid = torch.ones_like(logits, dtype=torch.bool)
    if method == "valid" and external_mask is not None:
        logits = logits.masked_fill(~external_mask, float("-inf"))
        valid = external_mask
    else:
        idx = torch.arange(n_problems, device=logits.device)
        logits = logits.clone()
        logits[idx, c1] = float("-inf")
        valid = valid.clone()
        valid[idx, c1] = False
        if method == "free":
            logits[:, 0] = float("-inf")
            logits[:, -1] = float("-inf")
            valid[:, 0] = False
            valid[:, -1] = False
    return logits, valid
