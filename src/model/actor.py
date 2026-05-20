"""CVRP actor — two-stage factorized policy (sample c1, then c2 | c1).

Wraps the underlying _ActorNet as a TensorDictModuleBase. The `forward` is
used during SA collection; `evaluate` is called from LGSAPPOLoss to
recompute log-prob + entropy on a stored action.
"""
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from .base import (
    apply_mask_c1, apply_mask_c2, build_mlp, init_orthogonal, sample_from_logits,
)


class _ActorNet(nn.Module):
    """Two MLPs: one scores c1 candidates, one scores c2 candidates given c1.

    Per-node feature dim is `c` (without the leading index column). The c2 net
    consumes 2c-2 features: a node's features concatenated with the chosen-c1
    features (minus the duplicated meta channels).
    """

    def __init__(
        self, c: int, embed_dim: int, num_hidden_layers: int, device: str | torch.device
    ):
        super().__init__()
        self.city1_net = build_mlp(c, embed_dim, num_hidden_layers, device)
        self.city2_net = build_mlp(2 * c - 2, embed_dim, num_hidden_layers, device)
        if str(device) != "mps":
            self.city1_net.apply(init_orthogonal)
            self.city2_net.apply(init_orthogonal)
            for net in (self.city1_net, self.city2_net):
                last = net[-1]
                if isinstance(last, nn.Linear):
                    nn.init.orthogonal_(last.weight, gain=0.01)
                    if last.bias is not None:
                        nn.init.constant_(last.bias, 0.0)

    def c1_logits(self, c_state: torch.Tensor) -> torch.Tensor:
        return self.city1_net(c_state)[..., 0]

    def c2_logits(self, c2_state: torch.Tensor) -> torch.Tensor:
        return self.city2_net(c2_state)[..., 0]


def _safe_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax(logits), safe under masked (-inf) logits.

    Two failure modes to guard:

    1. Partially-masked rows: `-(p * log p).sum` over a row with some -inf logits
       produces NaN gradients at those positions because `0 * (-inf)` is NaN and
       its backward propagates NaN even when the forward is wrapped in
       `nan_to_num`. Fix: replace `log p` with 0 where p == 0 before multiplying.

    2. Fully-masked rows (all -inf — reachable via pathological multi-init or
       rm_depot states): `softmax(all -inf)` is NaN with undefined gradient. Fix:
       substitute zero-logits for these rows before softmax so the math is
       well-defined, then zero the resulting entropy. The `torch.where` blocks
       the gradient from flowing back through the substituted positions.
    """
    all_masked = torch.isinf(logits).all(dim=-1, keepdim=True)
    safe_logits = torch.where(all_masked, torch.zeros_like(logits), logits)
    probs = torch.softmax(safe_logits, dim=-1)
    log_probs = torch.log_softmax(safe_logits, dim=-1)
    safe_lp = torch.where(probs > 0, log_probs, torch.zeros_like(log_probs))
    ent = -(probs * safe_lp).sum(dim=-1)
    return torch.where(all_masked.squeeze(-1), torch.zeros_like(ent), ent)


def _prepare_features_city2(c1_state: torch.Tensor, c1: torch.Tensor) -> torch.Tensor:
    """Build per-node c2 features: concat (chosen-c1 features) with (this node's
    features minus the last 2 meta channels). Returns (B, L, 2c-2)."""
    B, _, c = c1_state.shape
    arange = torch.arange(B, device=c1_state.device)
    c1_vec = c1_state[arange, c1]                              # (B, c)
    base = c1_vec[:, None, :].expand(-1, c1_state.size(1), -1)  # (B, L, c)
    trunc = c1_state[:, :, :-2]                                 # (B, L, c-2)
    return torch.cat([base, trunc], dim=-1)


class CVRPActor(TensorDictModuleBase):
    """Two-stage factorized policy wrapped as a TensorDictModuleBase."""

    in_keys = ["observation", "solution"]
    out_keys = ["action", "sample_log_prob", "entropy", "action_mask"]

    def __init__(
        self,
        c: int,
        embed_dim: int,
        num_hidden_layers: int,
        method: str,
        device: str | torch.device,
        get_action_mask_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.net = _ActorNet(c, embed_dim, num_hidden_layers, device)
        self.method = method
        self.get_action_mask_fn = get_action_mask_fn
        self._generator = torch.Generator(device=str(device))

    def manual_seed(self, seed: int) -> None:
        self._generator = torch.Generator(device=self._generator.device).manual_seed(seed)

    def forward(self, td: TensorDict, greedy: bool = False) -> TensorDict:
        obs = td["observation"]
        x = obs[..., :1].long()
        c_state = obs[..., 1:]
        B = obs.size(0)

        logits1 = self.net.c1_logits(c_state)
        logits1, _ = apply_mask_c1(logits1, x, self.method)
        c1, log_p_c1 = sample_from_logits(logits1, self._generator, greedy)

        c2_state = _prepare_features_city2(c_state, c1)
        logits2 = self.net.c2_logits(c2_state)
        ext_mask = None
        if self.method == "valid" and self.get_action_mask_fn is not None:
            ext_mask = self.get_action_mask_fn(td, c1)
        logits2, mask = apply_mask_c2(logits2, c1, self.method, B, ext_mask)
        c2, log_p_c2 = sample_from_logits(logits2, self._generator, greedy)

        action = torch.stack([c1.long(), c2.long()], dim=-1)
        log_prob = (log_p_c1 + log_p_c2).squeeze(-1)

        # Distribution-level entropy. Use the masked-safe form so masked
        # positions (logits == -inf, probs == 0) contribute 0 with zero gradient.
        ent_c1 = _safe_entropy(logits1)
        ent_c2 = _safe_entropy(logits2)

        td.set("action", action)
        td.set("sample_log_prob", log_prob)
        td.set("entropy", ent_c1 + ent_c2)
        td.set("action_mask", mask)
        return td

    def evaluate(self, td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recompute log_prob + entropy for a stored action. Called from LGSAPPOLoss.

        The c1 distribution is reproduced exactly: `apply_mask_c1` depends only on
        the (deterministic) `x` column of the observation, which is identical to
        sampling time. For c2 the mask depends on c1 *and*, in `method="valid"`
        mode, on the dynamic capacity state. Re-deriving it here would require
        carrying segment_ids/ordered_demands in the replay TD. Instead we read
        back the mask captured at sampling time (which already incorporates the
        `get_action_mask` `force_no_op` fallback), giving identical logits.
        """
        obs = td["observation"]
        x = obs[..., :1].long()
        c_state = obs[..., 1:]
        B = obs.size(0)
        action = td["action"]
        stored_mask = td["action_mask"]
        c1, c2 = action[..., 0].long(), action[..., 1].long()

        logits1 = self.net.c1_logits(c_state)
        logits1, _ = apply_mask_c1(logits1, x, self.method)
        lp1_all = torch.log_softmax(logits1, dim=-1)

        c2_state = _prepare_features_city2(c_state, c1)
        logits2 = self.net.c2_logits(c2_state)
        ext = stored_mask if self.method == "valid" else None
        logits2, _ = apply_mask_c2(logits2, c1, self.method, B, ext)
        lp2_all = torch.log_softmax(logits2, dim=-1)

        log_prob = (
            lp1_all.gather(1, c1.view(-1, 1)).squeeze(-1)
            + lp2_all.gather(1, c2.view(-1, 1)).squeeze(-1)
        )
        ent = _safe_entropy(logits1) + _safe_entropy(logits2)
        return log_prob, ent
