"""TensorDict-native CVRP state management.

The state TensorDict carries:
  - Static fields: coords, demands, capacity, distance_matrix, angles,
    dist_to_depot, demand_normalized, mean_dist_k.
  - Dynamic fields: solution, segment_ids, ordered_demands.
  - SA meta: temp, progress.

`init_state` is called once per epoch after instance generation;
`update_state` is called every time the solution is replaced in the SA loop.
"""
import torch
from tensordict import TensorDict

from .geometry import (
    calculate_client_angles,
    calculate_distance_matrix,
    calculate_knn_isolation,
    is_feasible,
)


def _enrich_static(static_td: TensorDict) -> TensorDict:
    """Add cached static geometry/feature fields. Idempotent."""
    if "distance_matrix" in static_td.keys():
        return static_td
    coords = static_td["coords"]
    demands = static_td["demands"]
    capacity = static_td["capacity"]

    distance_matrix = calculate_distance_matrix(coords)
    angles = calculate_client_angles(coords)

    real_counts = (demands > 0).sum(dim=1) + 1
    min_real = int(real_counts.min().item())
    is_ghost = demands == 0
    is_ghost[:, 0] = False
    masked = distance_matrix.masked_fill(is_ghost.unsqueeze(1), float("inf"))

    # Cap each k at min_real - 1 so topk(k+1) stays within bounds for small instances.
    max_k = max(1, min_real - 1)
    mean_d5 = calculate_knn_isolation(masked, k=min(5, max_k))
    mean_d10 = calculate_knn_isolation(masked, k=min(max(1, min_real // 10), max_k))
    mean_d33 = calculate_knn_isolation(masked, k=min(max(1, min_real // 3), max_k))
    mean_dist_k = torch.cat([mean_d5, mean_d10, mean_d33], dim=-1)

    dist_to_depot = distance_matrix[:, 0, :]
    dmin = dist_to_depot.min(dim=1, keepdim=True)[0]
    dmax = dist_to_depot.max(dim=1, keepdim=True)[0]
    divisor = (dmax - dmin).clamp(min=1e-10)
    dist_to_depot = ((dist_to_depot - dmin) / divisor).unsqueeze(-1)

    demand_normalized = (demands.float() / capacity.float()).unsqueeze(-1)

    static_td.update({
        "distance_matrix": distance_matrix,
        "angles": angles,
        "mean_dist_k": mean_dist_k,
        "dist_to_depot": dist_to_depot,
        "demand_normalized": demand_normalized,
    })
    return static_td


def init_state(
    static_td: TensorDict,
    solution: torch.Tensor,
    *,
    init_temp_normalized: float = 1.0,
    init_progress: float = 1.0,
) -> TensorDict:
    """Build the full state TD from a static TD + an initial solution."""
    static_td = _enrich_static(static_td)
    B = static_td.batch_size[0]
    device = static_td.device

    ordered_demands = torch.gather(static_td["demands"], 1, solution.squeeze(-1))
    if not is_feasible(solution, ordered_demands, static_td["capacity"]).all():
        raise ValueError("Initial solution is not feasible.")

    is_depot = ordered_demands == 0
    segment_ids = is_depot.long().cumsum(dim=1)

    state = static_td.clone()
    state.update({
        "solution": solution,
        "segment_ids": segment_ids,
        "ordered_demands": ordered_demands,
        "temp": torch.full((B, 1), init_temp_normalized, device=device),
        "progress": torch.full((B, 1), init_progress, device=device),
    })
    return state


def update_state(state: TensorDict, new_solution: torch.Tensor) -> TensorDict:
    """Replace solution and refresh dependent fields. Mutates and returns state."""
    ordered_demands = torch.gather(state["demands"], 1, new_solution.squeeze(-1))
    is_depot = ordered_demands == 0
    segment_ids = is_depot.long().cumsum(dim=1)
    state["solution"] = new_solution
    state["ordered_demands"] = ordered_demands
    state["segment_ids"] = segment_ids
    return state


def _cost_with_solution(state: TensorDict, solution: torch.Tensor) -> torch.Tensor:
    coords = torch.gather(
        state["coords"], 1, solution.expand(-1, -1, state["coords"].size(-1))
    )
    next_coords = torch.cat([coords[:, 1:, :], coords[:, :1, :]], dim=1)
    return (coords - next_coords).norm(p=2, dim=-1).sum(dim=-1)


def cost(state: TensorDict) -> torch.Tensor:
    """Total tour length of the current solution. Returns (B,)."""
    return _cost_with_solution(state, state["solution"])


def cost_with_solution(state: TensorDict, solution: torch.Tensor) -> torch.Tensor:
    """Public wrapper — cost of `solution` against this state's coords."""
    return _cost_with_solution(state, solution)
