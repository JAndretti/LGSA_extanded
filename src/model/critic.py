"""CVRP critic — per-node MLP, mean-pooled to a scalar value."""
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from .base import build_mlp, init_orthogonal


class CVRPCritic(TensorDictModuleBase):
    in_keys = ["observation"]
    out_keys = ["state_value"]

    def __init__(
        self, c: int, embed_dim: int, num_hidden_layers: int, device: str | torch.device
    ):
        super().__init__()
        self.q_func = build_mlp(c, embed_dim, num_hidden_layers, device)
        if str(device) != "mps":
            self.q_func.apply(init_orthogonal)
            last = self.q_func[-1]
            if isinstance(last, nn.Linear):
                nn.init.orthogonal_(last.weight, gain=1.0)
                if last.bias is not None:
                    nn.init.constant_(last.bias, 0.0)

    def forward(self, td: TensorDict) -> TensorDict:
        obs = td["observation"]
        per_node = self.q_func(obs[..., 1:])                       # (..., L, 1)
        value = per_node.squeeze(-1).mean(dim=-1, keepdim=True)    # (..., 1)
        td.set("state_value", value)
        return td
