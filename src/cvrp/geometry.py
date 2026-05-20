"""Geometry and static feature helpers for CVRP.

Ported from LGSA_OLD/src/utils.py.
"""
from typing import Tuple
import torch


def calculate_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    """(B, N, 2) -> (B, N, N) Euclidean pairwise distances."""
    return torch.cdist(coords, coords, p=2)


def calculate_client_angles(coords: torch.Tensor) -> torch.Tensor:
    """(B, N+1, 2) -> (B, N+1, 1) normalized angle in [0, 1] (depot=0)."""
    depot = coords[:, :1]
    clients = coords[:, 1:]
    delta = clients - depot
    ang = torch.atan2(delta[..., 1], delta[..., 0])              # [-pi, pi]
    norm = ang.div(2 * torch.pi).add(0.5)                         # [0, 1]
    all_ang = torch.cat(
        [torch.zeros(coords.size(0), 1, device=coords.device), norm], dim=1
    )
    return all_ang.unsqueeze(-1)


def calculate_knn_isolation(dists: torch.Tensor, k: int = 5) -> torch.Tensor:
    """For each node, mean distance to its k nearest neighbours (excluding self).

    Caller masks ghost/padding nodes with +inf in `dists` if needed.
    """
    values, _ = torch.topk(dists, k=k + 1, dim=-1, largest=False)
    knn_dists = values[:, :, 1:]
    return knn_dists.mean(dim=-1, keepdim=True)


def calculate_detour_features(
    solution: torch.Tensor, distance_matrix: torch.Tensor
) -> torch.Tensor:
    """Per-position detour cost: d(prev, x) + d(x, next) - d(prev, next).

    solution: (B, L, 1). distance_matrix: (B, N, N). Returns (B, L, 1).
    """
    sol = solution.squeeze(-1).long()
    B, L = sol.shape
    prev_sol = torch.roll(sol, shifts=1, dims=1)
    next_sol = torch.roll(sol, shifts=-1, dims=1)
    batch_idx = torch.arange(B, device=distance_matrix.device).unsqueeze(1).expand(B, L)
    d_pi = distance_matrix[batch_idx, prev_sol, sol]
    d_in = distance_matrix[batch_idx, sol, next_sol]
    d_pn = distance_matrix[batch_idx, prev_sol, next_sol]
    return (d_pi + d_in - d_pn).unsqueeze(-1)


def is_feasible(
    solution: torch.Tensor, ordered_demands: torch.Tensor, capacity: torch.Tensor
) -> torch.Tensor:
    """Check that no route exceeds capacity. Returns (B,) bool.

    Also enforces that solutions start and end at the depot.
    """
    batch_size = solution.size(0)
    device = solution.device

    mask = solution.squeeze(-1) != 0
    segment_start = mask & ~torch.cat(
        [torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1
    )
    segment_ids = torch.cumsum(segment_start, 1) * mask
    # Static upper bound L+1 (no host-device sync).
    L = segment_ids.size(1)
    route_loads = torch.zeros(
        batch_size, L + 1, device=device, dtype=ordered_demands.dtype
    )
    route_loads.scatter_add_(1, segment_ids, ordered_demands)

    feasible = (route_loads <= capacity).all(dim=1)
    start_w_depot = solution[:, 0, 0] == 0
    end_w_depot = solution[:, -1, 0] == 0
    return feasible & start_w_depot & end_w_depot


def capacity_utilization(
    solution: torch.Tensor, ordered_demands: torch.Tensor, capacity: torch.Tensor
) -> torch.Tensor:
    """Mean (1 - utilization) per route, averaged across routes. Returns (B,)."""
    batch_size = solution.size(0)
    device = solution.device

    mask = solution.squeeze(-1) != 0
    route_starts = torch.cat(
        [
            torch.ones(batch_size, 1, dtype=torch.bool, device=device),
            (~mask[:, :-1]) & mask[:, 1:],
        ],
        dim=1,
    )
    routes_count = route_starts.sum(dim=1).float()
    route_ids = torch.cumsum(route_starts, dim=1) - 1
    route_ids[~mask] = -1
    # Static upper bound: route_ids in [-1, L-1], so L slots cover all valid ids.
    L = solution.size(1)
    route_demands = torch.zeros(
        batch_size, max(L, 1), dtype=torch.float, device=device
    )
    route_demands.scatter_add_(
        1, torch.clamp(route_ids, min=0), ordered_demands.float() * mask.float()
    )
    route_utilization = route_demands / capacity.float()
    avg = route_utilization.sum(dim=1) / routes_count.clamp(min=1)
    return 1 - avg


def extend(tensor: torch.Tensor, dims: int) -> torch.Tensor:
    """Append `dims` trailing singleton dimensions."""
    return tensor[(...,) + (None,) * dims]


def extend_to(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Extend tensor1 to have same number of dims as tensor2."""
    return extend(tensor1, len(tensor2.shape) - len(tensor1.shape))


def repeat_to(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Broadcast tensor1 to match tensor2's shape (keep tensor1's last dim)."""
    tensor1 = extend_to(tensor1, tensor2)
    ones = torch.ones(
        tensor2.shape[:-1] + (1,), device=tensor1.device, dtype=tensor1.dtype
    )
    return tensor1 * ones
