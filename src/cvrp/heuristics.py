"""Local move operators and the apply_move dispatcher.

Ported verbatim (math) from LGSA_OLD/src/algo/local_heuristics.py and
LGSA_OLD/src/algo/heur_init.py::construct_cvrp_solution.
"""
import torch
from tensordict import TensorDict

from .geometry import is_feasible


_MULT = 0.6  # padding factor for construct_cvrp_solution


def swap(solution: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Swap two nodes in the solution."""
    sol = solution.clone()
    batch_idx = torch.arange(sol.size(0), device=sol.device)
    idx1, idx2 = indices[:, 0], indices[:, 1]
    temp = sol[batch_idx, idx1].clone()
    sol[batch_idx, idx1] = sol[batch_idx, idx2]
    sol[batch_idx, idx2] = temp
    return sol


def two_opt(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Reverse the segment between indices a[:, 0] and a[:, 1] (inclusive of left)."""
    left = torch.minimum(a[:, 0], a[:, 1])
    right = torch.maximum(a[:, 0], a[:, 1])
    batch_size, dim = x.size(0), x.size(1)
    ones = torch.ones((batch_size, 1), dtype=torch.long, device=x.device)
    fidx = torch.arange(dim, device=x.device) * ones
    offset = left + right - 1
    ridx = torch.arange(0, -dim, -1, device=x.device) + offset[:, None]
    flip = torch.ge(fidx, left[:, None]) * torch.lt(fidx, right[:, None])
    idx = (~flip) * fidx + flip * ridx
    return torch.gather(x, 1, idx.unsqueeze(-1))


def two_opt_star(
    x: torch.Tensor, a: torch.Tensor, segment_ids: torch.Tensor
) -> torch.Tensor:
    """Inter-route 2-opt*: swap tails (positions > selected idx) of two routes."""
    batch_size, seq_len = x.size(0), x.size(1)
    u_idx = a[:, 0]
    v_idx = a[:, 1]
    range_idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

    u_seg = segment_ids.gather(1, u_idx.unsqueeze(1))
    v_seg = segment_ids.gather(1, v_idx.unsqueeze(1))
    is_u_seg = segment_ids == u_seg
    is_v_seg = segment_ids == v_seg

    u_tail_mask = is_u_seg & (range_idx > u_idx.unsqueeze(1))
    v_tail_mask = is_v_seg & (range_idx > v_idx.unsqueeze(1))

    new_pos = range_idx.float()
    new_pos = torch.where(
        v_tail_mask,
        u_idx.unsqueeze(1).float() + 0.1 + (range_idx.float() * 1e-5),
        new_pos,
    )
    new_pos = torch.where(
        u_tail_mask,
        v_idx.unsqueeze(1).float() + 0.1 + (range_idx.float() * 1e-5),
        new_pos,
    )
    _, sort_indices = torch.sort(new_pos, dim=1)
    return torch.gather(x, 1, sort_indices.unsqueeze(-1))


def insertion(solution: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Move the node at action[:, 0] to land immediately after action[:, 1]."""
    src_idx = action[:, 0]
    tgt_idx = action[:, 1]
    batch_size, num_nodes = solution.shape[:2]
    device = solution.device

    scores = torch.arange(num_nodes, device=device, dtype=torch.float32)
    scores = scores.unsqueeze(0).expand(batch_size, -1).clone()
    target_scores = tgt_idx.float() + 0.5
    scores.scatter_(1, src_idx.unsqueeze(1), target_scores.unsqueeze(1))
    _, new_order = torch.sort(scores, dim=1)

    if solution.dim() == 3:
        idx_expanded = new_order.unsqueeze(-1).expand_as(solution)
        return torch.gather(solution, 1, idx_expanded)
    return torch.gather(solution, 1, new_order)


def construct_cvrp_solution(
    x: torch.Tensor, demands: torch.Tensor, capacity: torch.Tensor
) -> torch.Tensor:
    """Rebuild a CVRP solution from a TSP-style sequence by re-inserting depots.

    Used by `apply_move` when update_method='rm_depot'.
    """
    x = x.squeeze(-1)
    batch_size, num_nodes = demands.shape
    device = x.device

    max_route_length = num_nodes + int(num_nodes * _MULT)
    routes = torch.zeros(batch_size, max_route_length, dtype=torch.long, device=device)

    x = torch.cat(
        [torch.zeros(batch_size, 1, dtype=torch.long, device=device), x], dim=1
    )
    ordered_demands = torch.gather(demands, 1, x)

    capacity = capacity.squeeze()
    remaining_capacity = capacity.long()
    route_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
    batch_idx = torch.arange(batch_size, device=device)

    for i in range(num_nodes):
        current_client = x[:, i]
        can_serve = ordered_demands[:, i] <= remaining_capacity

        routes[batch_idx, route_pos] = torch.where(can_serve, current_client, 0)
        route_pos += 1

        remaining_capacity = torch.where(
            can_serve, remaining_capacity - ordered_demands[:, i], remaining_capacity
        )
        remaining_capacity = torch.where(
            can_serve, remaining_capacity, capacity - ordered_demands[:, i]
        )

        routes[batch_idx, route_pos] = torch.where(can_serve, 0, current_client)
        route_pos += (~can_serve).long()

    return routes.unsqueeze(-1)


def two_opt_split(
    solution: torch.Tensor,
    action: torch.Tensor,
    segment_ids: torch.Tensor,
    update_method: str,
) -> torch.Tensor:
    """Dispatches intra-route 2-opt (reversal) vs inter-route 2-opt* (tail swap)."""
    if update_method == "rm_depot":
        return two_opt(solution, action)

    u, v = action[:, 0], action[:, 1]
    u_seg = segment_ids.gather(1, u.unsqueeze(1)).squeeze(1)
    v_seg = segment_ids.gather(1, v.unsqueeze(1)).squeeze(1)
    is_intra = u_seg == v_seg

    sol_intra = two_opt(solution, action)
    sol_inter = two_opt_star(solution, action, segment_ids)
    mask = is_intra.view(-1, 1, 1)
    return torch.where(mask, sol_intra, sol_inter)


def _dispatch_heuristic(
    name: str,
    solution: torch.Tensor,
    action: torch.Tensor,
    segment_ids: torch.Tensor,
    update_method: str,
) -> torch.Tensor:
    if name == "insertion":
        return insertion(solution, action)
    if name == "swap":
        return swap(solution, action)
    if name == "two_opt":
        return two_opt_split(solution, action, segment_ids, update_method)
    raise ValueError(f"Unknown heuristic: {name}")


def apply_move(
    state: TensorDict,
    action: torch.Tensor,
    heuristic: str,
    update_method: str,
):
    """Apply a move to the solution. Returns (new_solution, valid_mask)."""
    solution = state["solution"]
    demands = state["demands"]
    capacity = state["capacity"]
    segment_ids = state["segment_ids"]

    if update_method == "rm_depot":
        mask_nodes = solution.squeeze(-1) != 0
        # `.view(B, -1, ...)` assumes every row has the same number of non-zero
        # entries. Holds for random instances of fixed `dim` (all customers
        # demand > 0); will fail loudly for variable-size or degenerate inputs.
        per_row_nonzero = mask_nodes.sum(dim=1)
        assert (per_row_nonzero == per_row_nonzero[0]).all(), (
            "rm_depot requires uniform non-zero count per row; got "
            f"{per_row_nonzero.tolist()}"
        )
        compact = solution[mask_nodes].view(solution.size(0), -1, solution.size(-1))
        compact_seg = torch.zeros(
            compact.size(0), compact.size(1), dtype=torch.long, device=compact.device
        )
        modified = _dispatch_heuristic(
            heuristic, compact, action, compact_seg, update_method
        ).long()
        sol = construct_cvrp_solution(modified, demands, capacity)
        pad = solution.size(1) - sol.size(1)
        if pad > 0:
            sol = torch.cat(
                [sol, torch.zeros(sol.size(0), pad, sol.size(2),
                                   dtype=sol.dtype, device=sol.device)],
                dim=1,
            )
        elif pad < 0:
            sol = sol[:, :solution.size(1)]
    else:
        sol = _dispatch_heuristic(
            heuristic, solution, action, segment_ids, update_method
        ).long()

    if update_method == "free":
        new_demands = torch.gather(demands, 1, sol.squeeze(-1))
        valid = is_feasible(sol, new_demands, capacity).unsqueeze(-1)
    else:
        valid = torch.ones(sol.size(0), 1, device=sol.device, dtype=torch.bool)
    return sol, valid
