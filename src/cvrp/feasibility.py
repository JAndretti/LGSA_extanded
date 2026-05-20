"""Feasibility checks and action masks for CVRP moves.

Action mask logic is dispatched by heuristic (insertion / swap / two_opt).
Ported from LGSA_OLD/src/problem.py::get_action_mask.
"""
from typing import Tuple
import torch
from tensordict import TensorDict


def get_route_loads(state: TensorDict) -> torch.Tensor:
    """Total load per route (segment). Returns (B, L+1).

    The trailing slots beyond the actual number of routes stay at 0 (scatter_add
    only writes at the segment_ids indices). Allocating with the static upper
    bound L+1 avoids a host-device sync on every SA step.
    """
    segment_ids = state["segment_ids"]
    ordered_demands = state["ordered_demands"]
    B, L = segment_ids.shape
    route_loads = torch.zeros(
        B, L + 1, device=state.device, dtype=ordered_demands.dtype
    )
    route_loads.scatter_add_(1, segment_ids, ordered_demands)
    return route_loads


def prefix_loads(state: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cumulative loads from route start to each node (head) and remaining to end (tail).

    Both shapes (B, L). Same math as LGSA_OLD/src/problem.py::_get_prefix_loads.
    """
    segment_ids = state["segment_ids"]
    ordered_demands = state["ordered_demands"]
    same_segment = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
    lower_tri = torch.tril(torch.ones_like(same_segment))
    mask = same_segment * lower_tri
    head = (ordered_demands.unsqueeze(1) * mask).sum(dim=2)
    totals_per_route = get_route_loads(state)
    total = torch.gather(totals_per_route, 1, segment_ids)
    tail = total - head
    return head, tail


def get_action_mask(
    state: TensorDict, node_pos: torch.Tensor, heuristic: str
) -> torch.Tensor:
    """Per-heuristic action mask for c2 given c1=node_pos. Returns (B, L) bool."""
    segment_ids = state["segment_ids"]
    ordered_demands = state["ordered_demands"]
    capacity = state["capacity"]
    B, L = segment_ids.shape
    node_pos_e = node_pos.unsqueeze(-1)

    target_route_ids = segment_ids
    source_route_id = torch.gather(target_route_ids, 1, node_pos_e)
    source_demand = torch.gather(ordered_demands, 1, node_pos_e)
    is_source_valid = source_demand > 0
    is_intra = target_route_ids == source_route_id

    mask = torch.zeros(B, L, device=state.device, dtype=torch.bool)

    if heuristic == "insertion":
        per_route_loads = get_route_loads(state)
        target_loads = torch.gather(per_route_loads, 1, target_route_ids)
        potential_loads = target_loads + source_demand
        is_cap_valid = potential_loads <= capacity
        mask = is_intra | is_cap_valid
        mask[:, -1] = False

    elif heuristic == "swap":
        per_route_loads = get_route_loads(state)
        target_loads = torch.gather(per_route_loads, 1, target_route_ids)
        source_load = torch.gather(per_route_loads, 1, source_route_id)
        new_source = source_load - source_demand + ordered_demands
        new_target = target_loads - ordered_demands + source_demand
        is_cap_valid = (new_source <= capacity) & (new_target <= capacity)
        standard = (is_intra | is_cap_valid) & (ordered_demands > 0)
        is_depot = ordered_demands == 0
        is_prev = torch.roll(is_depot, shifts=1, dims=1);   is_prev[:, 0] = False
        is_next = torch.roll(is_depot, shifts=-1, dims=1);  is_next[:, -1] = False
        new_route = is_prev & is_depot & is_next
        mask = standard | new_route

    elif heuristic == "two_opt":
        head, tail = prefix_loads(state)
        src_head = torch.gather(head, 1, node_pos_e)
        src_tail = torch.gather(tail, 1, node_pos_e)
        is_target_customer = ordered_demands > 0
        intra_customer = is_intra & is_target_customer
        new_a = src_head + tail
        new_b = head + src_tail
        is_cap_valid = (new_a <= capacity) & (new_b <= capacity)
        inter = (~is_intra) & is_cap_valid
        mask = intra_customer | inter
        prev_idx = (node_pos_e - 1).clamp(min=0)
        next_idx = (node_pos_e + 1).clamp(max=L - 1)
        mask.scatter_(1, prev_idx, False)
        mask.scatter_(1, next_idx, False)
        mask[:, -1] = False

    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")

    mask.scatter_(1, node_pos_e, False)
    has_moves = mask.any(dim=1, keepdim=True)
    force_no_op = (~is_source_valid) | (~has_moves)
    no_op = torch.zeros_like(mask)
    no_op.scatter_(1, node_pos_e, True)
    return torch.where(force_no_op, no_op, mask)
