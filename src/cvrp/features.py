"""Build the per-node feature tensor for the model.

Each feature group is a separate function. Selected groups are concatenated
along the last dim according to feature_flags from HP.yaml.

The first column of the returned tensor is the leading node-index column (x);
the rest are the engineered features. The actor reads x for masking; the
critic strips it. `input_dim(feature_flags)` is the c (without leading index).
"""
from typing import Mapping
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .geometry import calculate_detour_features, repeat_to


GROUP_WIDTHS = {
    "static":     6,
    "topology":   4,
    "density5":   1,
    "density10":  1,
    "density33":  1,
    "detour":     1,
    "centroid":   1,
    "route_pct":  1,
    "slack":      1,
    "node_pct":   1,
    "meta":       2,
}


def input_dim(feature_flags: Mapping[str, bool]) -> int:
    """Total c (per-node feature dim, excluding the leading index column)."""
    return sum(GROUP_WIDTHS[k] for k, v in feature_flags.items() if v)


def _padded_coords(state: TensorDict, x: torch.Tensor) -> torch.Tensor:
    coords = state["coords"]
    pad = max(0, x.size(1) - coords.size(1))
    coords_p = F.pad(coords, (0, 0, 0, pad))
    return coords_p.gather(1, x.expand(-1, -1, 2))


def static_features(state: TensorDict, x: torch.Tensor) -> torch.Tensor:
    coords_p = _padded_coords(state, x)
    is_depot = (x == 0).float()
    angles = state["angles"].gather(1, x)
    dist_dep = state["dist_to_depot"].gather(1, x)
    demand_n = state["demand_normalized"].gather(1, x)
    return torch.cat([coords_p, is_depot, angles, dist_dep, demand_n], dim=-1)


def topology_features(state: TensorDict, x: torch.Tensor) -> torch.Tensor:
    coords_p = _padded_coords(state, x)
    return torch.cat([
        torch.roll(coords_p, shifts=1, dims=1),
        torch.roll(coords_p, shifts=-1, dims=1),
    ], dim=-1)


def density_features(state: TensorDict, x: torch.Tensor, which: int) -> torch.Tensor:
    """which: 0 → k=5, 1 → k=10%, 2 → k=33%."""
    return state["mean_dist_k"][:, :, which:which + 1].gather(1, x)


def detour_features(state: TensorDict, x: torch.Tensor) -> torch.Tensor:
    return calculate_detour_features(x, state["distance_matrix"])


def centroid_features(state: TensorDict, x: torch.Tensor) -> torch.Tensor:
    """Per-node distance to its route's centroid."""
    segment_ids = state["segment_ids"]
    coords = state["coords"].gather(1, x.expand(-1, -1, 2))
    B, L = segment_ids.shape
    # Use static upper bound L+1 to avoid a host-device sync.
    num_routes = L + 1
    sums = torch.zeros(B, num_routes, 2, device=state.device, dtype=coords.dtype)
    counts = torch.zeros(B, num_routes, 1, device=state.device, dtype=coords.dtype)
    seg_e = segment_ids.unsqueeze(-1).expand(-1, -1, 2)
    sums.scatter_add_(1, seg_e, coords)
    counts.scatter_add_(
        1, segment_ids.unsqueeze(-1),
        torch.ones(B, L, 1, device=state.device, dtype=coords.dtype),
    )
    centroids = sums / counts.clamp(min=1.0)
    node_centroids = centroids.gather(1, seg_e)
    return (coords - node_centroids).norm(p=2, dim=-1, keepdim=True)


def _percentage_demands(state: TensorDict):
    """Returns (node_pct, route_pct, slack)."""
    nd = state["ordered_demands"]
    capacity = state["capacity"]
    total = torch.zeros_like(nd)
    total.scatter_add_(1, state["segment_ids"], nd)
    route_load = total.gather(1, state["segment_ids"])
    node_frac = torch.nan_to_num(nd.float() / route_load.float().clamp(min=1.0), nan=0.0)
    load_frac = route_load.float() / capacity.float()
    remaining = (capacity.float() - route_load.float()) / capacity.float()
    return node_frac.unsqueeze(-1), load_frac.unsqueeze(-1), remaining.unsqueeze(-1)


def meta_features(state: TensorDict, x: torch.Tensor) -> torch.Tensor:
    """Two channels: normalized temperature and progress, broadcast per node."""
    temp = repeat_to(state["temp"].squeeze(-1), x.float())
    progress = repeat_to(state["progress"].squeeze(-1), x.float())
    return torch.cat([temp, progress], dim=-1)


def build_features(
    state: TensorDict, feature_flags: Mapping[str, bool]
) -> torch.Tensor:
    """Return (B, L, 1 + sum(enabled_group_widths)) feature tensor.

    Leading column is the node-index x (kept as float for concat).
    """
    x = state["solution"]
    components = [x.float()]

    if feature_flags.get("static", False):
        components.append(static_features(state, x))
    if feature_flags.get("topology", False):
        components.append(topology_features(state, x))
    if feature_flags.get("density5", False):
        components.append(density_features(state, x, 0))
    if feature_flags.get("density10", False):
        components.append(density_features(state, x, 1))
    if feature_flags.get("density33", False):
        components.append(density_features(state, x, 2))
    if feature_flags.get("detour", False):
        components.append(detour_features(state, x))
    if feature_flags.get("centroid", False):
        components.append(centroid_features(state, x))

    if any(feature_flags.get(k, False) for k in ("route_pct", "slack", "node_pct")):
        node_pct, route_pct, slack = _percentage_demands(state)
        if feature_flags.get("route_pct", False):
            components.append(route_pct)
        if feature_flags.get("slack", False):
            components.append(slack)
        if feature_flags.get("node_pct", False):
            components.append(node_pct)

    if feature_flags.get("meta", False):
        components.append(meta_features(state, x))

    return torch.cat(components, dim=-1)
