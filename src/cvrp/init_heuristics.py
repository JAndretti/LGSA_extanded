"""Initial-solution heuristics for CVRP.

Eight heuristics ported from LGSA_OLD/src/algo/heur_init.py. Each takes a
static TensorDict (with coords, demands, capacity) and returns a (B, L, 1)
solution tensor. A small `_StaticView` adapter lets the ported function
bodies use the same attribute access pattern as the old CVRP class.
"""
from types import SimpleNamespace
from typing import Callable, List

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .geometry import calculate_distance_matrix
from .heuristics import construct_cvrp_solution


_MULT = 0.6


def _view(static_td: TensorDict) -> SimpleNamespace:
    """Adapter so ported heuristics can use cvrp_instance.attr access."""
    coords = static_td["coords"]
    return SimpleNamespace(
        n_problems=static_td.batch_size[0],
        dim=coords.shape[1] - 1,
        coords=coords,
        demands=static_td["demands"],
        capacity=static_td["capacity"],
        matrix=static_td.get("distance_matrix", calculate_distance_matrix(coords)),
    )


# ----------------------------------------------------------------------
# Heuristic implementations (ported from LGSA_OLD).
# ----------------------------------------------------------------------


def generate_isolate_solution(view) -> torch.Tensor:
    """One client per route: [0, 1, 0, 2, 0, ..., 0, N, 0]."""
    batch_size = view.n_problems
    dim = view.dim
    device = "cpu"
    route_length = 2 * dim + 1
    pattern = torch.zeros(route_length, dtype=torch.long, device=device)
    pattern[0] = 0
    pattern[-1] = 0
    pattern[1:-1:2] = torch.arange(1, dim + 1, device=device)
    pattern[2:-1:2] = 0
    routes = pattern.unsqueeze(0).repeat(batch_size, 1)
    return routes.unsqueeze(-1)


def generate_sweep_solution(view) -> torch.Tensor:
    """Sweep clients by polar angle around depot."""
    batch_size = view.n_problems
    dim = view.dim
    num_total_nodes = dim + 1
    device = "cpu"
    coords = view.coords.cpu()
    demands = view.demands.cpu()
    capacity = view.capacity.squeeze(-1).cpu()

    max_route_len = num_total_nodes + int(num_total_nodes * _MULT)
    routes = torch.zeros(batch_size, max_route_len, dtype=torch.long, device=device)

    depot_coords = coords[:, 0:1, :]
    client_coords = coords[:, 1:, :]
    delta_coords = client_coords - depot_coords
    angles = torch.atan2(delta_coords[:, :, 1], delta_coords[:, :, 0])

    client_indices = torch.arange(1, dim + 1, device=device).expand(batch_size, -1)
    sorted_indices, sorted_demands = [], []
    for b in range(batch_size):
        _, idx = torch.sort(angles[b])
        sorted_indices.append(client_indices[b, idx])
        sorted_demands.append(demands[b, client_indices[b, idx]])
    sorted_indices = torch.stack(sorted_indices)
    sorted_demands = torch.stack(sorted_demands)

    current_loads = torch.zeros(batch_size, device=device)
    route_pos = torch.ones(batch_size, dtype=torch.long, device=device)
    routes[:, 0] = 0

    for i in range(dim):
        client_ids = sorted_indices[:, i]
        client_demands_i = sorted_demands[:, i]
        capacity_check = current_loads + client_demands_i <= capacity
        route_pos = torch.where(capacity_check, route_pos, route_pos + 1)
        current_loads = torch.where(
            capacity_check, current_loads + client_demands_i, client_demands_i
        )
        for b in range(batch_size):
            if route_pos[b] < max_route_len:
                routes[b, route_pos[b]] = client_ids[b]
        route_pos += 1

    for b in range(batch_size):
        if route_pos[b] < max_route_len and routes[b, route_pos[b] - 1] != 0:
            routes[b, route_pos[b]] = 0

    return routes.unsqueeze(-1)


def random_init_batch(view) -> torch.Tensor:
    """Shuffle clients, then re-insert depots when capacity is exceeded."""
    demands = view.demands
    capacity = view.capacity.squeeze(-1)
    batch_size, num_nodes = demands.size()
    device = demands.device

    clients = torch.arange(1, num_nodes, device=device).repeat(batch_size, 1)
    rand_floats = torch.rand_like(clients, dtype=torch.float)
    indices = torch.argsort(rand_floats, dim=1)
    clients = torch.gather(clients, 1, indices)
    return construct_cvrp_solution(clients, demands, capacity)


def generate_nearest_neighbor(view) -> torch.Tensor:
    """Greedy nearest-neighbour from depot, returning to depot when full."""
    batch_size = view.n_problems
    dim = view.dim
    device = "cpu"
    coords = view.coords.cpu()
    demands = view.demands.cpu()
    capacity = view.capacity.squeeze(-1).cpu()

    dist_matrix = torch.cdist(coords, coords, p=2)
    max_route_len = dim + 1 + int((dim + 1) * _MULT)
    routes = torch.zeros(batch_size, max_route_len, dtype=torch.long, device=device)
    visited = torch.zeros(batch_size, dim + 1, dtype=torch.bool, device=device)
    visited[:, 0] = True

    current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
    current_load = torch.zeros(batch_size, device=device)
    route_pos = torch.ones(batch_size, dtype=torch.long, device=device)
    routes[:, 0] = 0

    while not visited[:, 1:].all():
        unvisited_mask = ~visited
        dist_from_current = dist_matrix[torch.arange(batch_size), current_node].clone()
        dist_from_current[~unvisited_mask] = float("inf")
        dist_from_current[:, 0] = float("inf")
        nearest_neighbor = torch.argmin(dist_from_current, dim=1)
        all_clients_visited = visited[:, 1:].all(dim=1)
        nearest_neighbor = torch.where(
            all_clients_visited, torch.zeros_like(nearest_neighbor), nearest_neighbor
        )
        neighbor_demands = demands[torch.arange(batch_size), nearest_neighbor]
        over_capacity = (current_load + neighbor_demands) > capacity
        routes[torch.arange(batch_size), route_pos] = torch.where(
            over_capacity | all_clients_visited,
            torch.zeros_like(nearest_neighbor),
            nearest_neighbor,
        )
        route_pos += 1
        current_node = torch.where(
            over_capacity | all_clients_visited,
            torch.zeros_like(current_node),
            nearest_neighbor,
        )
        current_load = torch.where(
            over_capacity | all_clients_visited,
            torch.zeros_like(current_load),
            current_load + neighbor_demands,
        )
        for b in range(batch_size):
            if not all_clients_visited[b] and not over_capacity[b]:
                visited[b, nearest_neighbor[b]] = True

    for b in range(batch_size):
        if route_pos[b] < max_route_len and routes[b, route_pos[b] - 1] != 0:
            routes[b, route_pos[b]] = 0
            route_pos[b] += 1

    return routes.unsqueeze(-1)


def _clark_wright_worker(args):
    batch_index, coords, demands, capacity, dim = args
    device = "cpu"
    coords_i = coords.unsqueeze(1)
    coords_j = coords.unsqueeze(0)
    dist = torch.norm(coords_i - coords_j, dim=2)

    d0i = dist[0, 1:]
    d0j = dist[0, 1:]
    dij = dist[1:, 1:]
    savings = d0i.unsqueeze(1) + d0j.unsqueeze(0) - dij
    savings = savings + torch.diag(torch.full((dim,), float("-inf"), device=device))

    savings_flat = savings.view(-1)
    _, sorted_idx = torch.sort(savings_flat, descending=True)
    i_idx = sorted_idx // dim + 1
    j_idx = sorted_idx % dim + 1

    routes = [[0, i + 1, 0] for i in range(dim)]
    route_demands = [demands[i + 1].item() for i in range(dim)]
    client_route = {i + 1: i for i in range(dim)}

    for k in range(dim * dim):
        i = i_idx[k].item()
        j = j_idx[k].item()
        if i == j:
            continue
        route_i = client_route.get(i, None)
        route_j = client_route.get(j, None)
        if route_i is None or route_j is None or route_i == route_j:
            continue
        route_i_seq = routes[route_i]
        route_j_seq = routes[route_j]
        if route_i_seq[-2] == i and route_j_seq[1] == j:
            total_demand = route_demands[route_i] + route_demands[route_j]
            if total_demand <= capacity.item():
                new_route = route_i_seq[:-1] + route_j_seq[1:]
                routes[route_i] = new_route
                route_demands[route_i] = total_demand
                routes[route_j] = []
                route_demands[route_j] = 0
                for node in route_j_seq[1:-1]:
                    client_route[node] = route_i
                continue
        if route_j_seq[-2] == j and route_i_seq[1] == i:
            total_demand = route_demands[route_i] + route_demands[route_j]
            if total_demand <= capacity.item():
                new_route = route_j_seq[:-1] + route_i_seq[1:]
                routes[route_j] = new_route
                route_demands[route_j] = total_demand
                routes[route_i] = []
                route_demands[route_i] = 0
                for node in route_i_seq[1:-1]:
                    client_route[node] = route_j
                continue

    flat = []
    for r in routes:
        if r:
            if flat and flat[-1] != 0:
                flat.append(0)
            flat += r[1:] if flat else r
    if not flat or flat[0] != 0:
        flat = [0] + flat
    return batch_index, flat


def generate_Clark_and_Wright(view) -> torch.Tensor:
    """Clarke-Wright savings algorithm. Sequential per-instance (no mp.Pool)."""
    batch_size = view.n_problems
    dim = view.dim
    num_total_nodes = dim + 1
    device = "cpu"
    coords = view.coords.cpu()
    demands = view.demands.cpu()
    capacity = view.capacity.squeeze(-1).cpu()

    args_list = [(b, coords[b], demands[b], capacity[b], dim) for b in range(batch_size)]
    results = [_clark_wright_worker(a) for a in args_list]

    max_route_len = num_total_nodes + int(num_total_nodes * _MULT)
    batch_routes = torch.zeros(batch_size, max_route_len, dtype=torch.long, device=device)
    for batch_idx, flat_route in results:
        flat_route = flat_route[:max_route_len] + [0] * (max_route_len - len(flat_route))
        batch_routes[batch_idx, : len(flat_route)] = torch.tensor(
            flat_route[:max_route_len], dtype=torch.long
        )
    return batch_routes.unsqueeze(-1)


def cheapest_insertion(view) -> torch.Tensor:
    """Cheapest-insertion: insert clients at their best feasible position."""
    batch_size = view.n_problems
    n_clients = view.dim
    n_total = n_clients + 1
    distance_matrix = view.matrix.cpu()
    demand = view.demands.cpu()
    vehicle_capacity = view.capacity.squeeze(-1).cpu()
    device = "cpu"

    routes = [
        torch.tensor([0, 0], dtype=torch.long, device=device) for _ in range(batch_size)
    ]
    remaining_clients = (
        torch.arange(1, n_clients + 1, device=device).unsqueeze(0).expand(batch_size, -1)
    )
    current_load = torch.zeros(batch_size, device=device)
    inserted = torch.zeros((batch_size, n_total), dtype=torch.bool, device=device)
    inserted[:, 0] = True

    while True:
        not_inserted = ~inserted
        not_inserted[:, 0] = False
        if not_inserted.sum().item() == 0:
            break
        for b in range(batch_size):
            if not_inserted[b].sum().item() == 0:
                continue
            best_cost = float("inf")
            best_position = -1
            best_client = -1
            depot_positions = (routes[b] == 0).nonzero().flatten()
            for i in range(len(depot_positions) - 1):
                start_idx = depot_positions[i]
                end_idx = depot_positions[i + 1]
                subroute = routes[b][start_idx : end_idx + 1]
                route_demand = (
                    demand[b, subroute[1:-1]].sum() if len(subroute) > 2 else 0
                )
                remaining_capacity = vehicle_capacity[b] - route_demand
                eligible_clients = remaining_clients[b][
                    (demand[b, remaining_clients[b]] <= remaining_capacity)
                    & not_inserted[b, remaining_clients[b]]
                ]
                if len(eligible_clients) == 0:
                    continue
                for k in range(len(subroute) - 1):
                    u = subroute[k]
                    v = subroute[k + 1]
                    delta = (
                        distance_matrix[b, u, eligible_clients]
                        + distance_matrix[b, eligible_clients, v]
                        - distance_matrix[b, u, v]
                    )
                    min_delta, min_idx = delta.min(dim=0)
                    if min_delta < best_cost:
                        best_cost = min_delta.item()
                        best_client = eligible_clients[min_idx].item()
                        best_position = start_idx + k + 1
            if best_cost == float("inf"):
                eligible = remaining_clients[b][not_inserted[b, remaining_clients[b]]]
                if len(eligible) == 0:
                    continue
                depot_dist = distance_matrix[b, 0, eligible]
                _, closest_idx = depot_dist.min(dim=0)
                best_client = eligible[closest_idx].item()
                last_zero_pos = (routes[b] == 0).nonzero()[-1].item()
                routes[b] = torch.cat(
                    [
                        routes[b][: last_zero_pos + 1],
                        torch.tensor([best_client, 0], device=device),
                        routes[b][last_zero_pos + 1 :],
                    ]
                )
                inserted[b, best_client] = True
                current_load[b] = demand[b, best_client]
            else:
                routes[b] = torch.cat(
                    [
                        routes[b][:best_position],
                        torch.tensor([best_client], device=device),
                        routes[b][best_position:],
                    ]
                )
                inserted[b, best_client] = True
                current_load[b] += demand[b, best_client]

    max_len = max(len(r) for r in routes) + int(n_clients * _MULT)
    padded_routes = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for b in range(batch_size):
        padded_routes[b, : len(routes[b])] = routes[b]
    return padded_routes.unsqueeze(-1)


def path_cheapest_arc(view) -> torch.Tensor:
    """Cheapest-arc construction."""
    batch_size = view.n_problems
    n_clients = view.dim
    n_total = n_clients + 1
    distance_matrix = view.matrix.cpu()
    demand = view.demands.cpu()
    vehicle_capacity = view.capacity.squeeze(-1).cpu()
    device = "cpu"

    routes = [
        torch.tensor([0], dtype=torch.long, device=device) for _ in range(batch_size)
    ]
    remaining_clients = (
        torch.arange(1, n_clients + 1, device=device).unsqueeze(0).expand(batch_size, -1)
    )
    inserted = torch.zeros((batch_size, n_total), dtype=torch.bool, device=device)
    inserted[:, 0] = True

    while True:
        not_inserted = ~inserted
        not_inserted[:, 0] = False
        if not_inserted.sum() == 0:
            break
        for b in range(batch_size):
            if not_inserted[b].sum() == 0:
                continue
            best_cost = float("inf")
            best_client = -1
            best_pos = -1
            new_route_needed = True
            route = routes[b]
            depot_positions = (route == 0).nonzero().flatten()
            for i in range(len(depot_positions) - 1):
                start, end = depot_positions[i], depot_positions[i + 1]
                subroute = route[start : end + 1]
                current_demand = (
                    demand[b, subroute[1:-1]].sum() if len(subroute) > 2 else 0
                )
                remaining_cap = vehicle_capacity[b] - current_demand
                eligible = remaining_clients[b][
                    (demand[b, remaining_clients[b]] <= remaining_cap)
                    & not_inserted[b, remaining_clients[b]]
                ]
                if len(eligible) == 0:
                    continue
                for j in range(len(subroute) - 1):
                    u, v = subroute[j], subroute[j + 1]
                    delta = (
                        distance_matrix[b, u, eligible]
                        + distance_matrix[b, eligible, v]
                        - distance_matrix[b, u, v]
                    )
                    min_delta, min_idx = delta.min(dim=0)
                    if min_delta < best_cost:
                        best_cost = min_delta.item()
                        best_client = eligible[min_idx].item()
                        best_pos = start + j + 1
                        new_route_needed = False
            if new_route_needed:
                eligible = remaining_clients[b][not_inserted[b, remaining_clients[b]]]
                if len(eligible) == 0:
                    continue
                depot_dists = distance_matrix[b, 0, eligible]
                _, closest_idx = depot_dists.min(dim=0)
                best_client = eligible[closest_idx].item()
                if routes[b][-1] != 0:
                    routes[b] = torch.cat([routes[b], torch.tensor([0], device=device)])
                routes[b] = torch.cat(
                    [routes[b], torch.tensor([best_client, 0], device=device)]
                )
            else:
                routes[b] = torch.cat(
                    [
                        routes[b][:best_pos],
                        torch.tensor([best_client], device=device),
                        routes[b][best_pos:],
                    ]
                )
            inserted[b, best_client] = True

    max_len = max(len(r) for r in routes) + int(n_clients * _MULT)
    padded_routes = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for b in range(batch_size):
        padded_routes[b, : len(routes[b])] = routes[b]
    return padded_routes.unsqueeze(-1)


def farthest_insertion(view) -> torch.Tensor:
    """Farthest-insertion: pick the unvisited node farthest from any inserted one."""
    batch_size = view.n_problems
    n_clients = view.dim
    n_total = n_clients + 1
    distance_matrix = view.matrix.cpu()
    demand = view.demands.cpu()
    vehicle_capacity = view.capacity.squeeze(-1).cpu()
    device = "cpu"

    routes: List[List[int] | torch.Tensor] = [[0] for _ in range(batch_size)]
    inserted = torch.zeros((batch_size, n_total), dtype=torch.bool, device=device)
    inserted[:, 0] = True

    for b in range(batch_size):
        depot_distances = distance_matrix[b, 0, 1:]
        farthest_client = torch.argmax(depot_distances).item() + 1
        routes[b] = [0, farthest_client, 0]
        inserted[b, farthest_client] = True

        while not inserted[b, 1:].all():
            unvisited = torch.where(~inserted[b, 1:])[0] + 1
            if len(unvisited) == 0:
                break
            dist_to_visited = distance_matrix[b, unvisited][:, inserted[b]]
            min_distances = dist_to_visited.min(dim=1).values
            farthest_unvisited = unvisited[min_distances.argmax()].item()

            best_cost = float("inf")
            best_position = -1
            can_insert = False
            depot_positions = [i for i, x in enumerate(routes[b]) if x == 0]
            for i in range(len(depot_positions) - 1):
                start, end = depot_positions[i], depot_positions[i + 1]
                subroute = routes[b][start : end + 1]
                current_demand = sum(demand[b, c] for c in subroute[1:-1])
                if current_demand + demand[b, farthest_unvisited] <= vehicle_capacity[b]:
                    for j in range(len(subroute) - 1):
                        u, v = subroute[j], subroute[j + 1]
                        cost = (
                            distance_matrix[b, u, farthest_unvisited]
                            + distance_matrix[b, farthest_unvisited, v]
                            - distance_matrix[b, u, v]
                        )
                        if cost < best_cost:
                            best_cost = cost
                            best_position = start + j + 1
                            can_insert = True
            if can_insert:
                routes[b].insert(best_position, farthest_unvisited)
            else:
                routes[b].extend([farthest_unvisited, 0])
            inserted[b, farthest_unvisited] = True

        routes[b] = torch.tensor(routes[b], dtype=torch.long, device=device)

    max_len = max(len(r) for r in routes) + int(n_clients * _MULT)
    padded_routes = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for b in range(batch_size):
        padded_routes[b, : len(routes[b])] = routes[b]
    return padded_routes.unsqueeze(-1)


# ----------------------------------------------------------------------
# Registry + entrypoint
# ----------------------------------------------------------------------


_REGISTRY: dict[str, Callable] = {
    "random":             random_init_batch,
    "sweep":              generate_sweep_solution,
    "isolate":            generate_isolate_solution,
    "Clark_and_Wright":   generate_Clark_and_Wright,
    "nearest_neighbor":   generate_nearest_neighbor,
    "cheapest_insertion": cheapest_insertion,
    "path_cheapest_arc":  path_cheapest_arc,
    "farthest_insertion": farthest_insertion,
}


INIT_METHODS = list(_REGISTRY.keys())


def generate_init_solutions(
    static_td: TensorDict,
    methods: list[str],
    multi_init: bool,
    max_routes_estimate: int,
) -> torch.Tensor:
    """Build the initial solution batch. Returns (B, L, 1)."""
    view = _view(static_td)
    B = static_td.batch_size[0]
    if not multi_init:
        sol = _REGISTRY[methods[0]](view).to(static_td.device)
        return sol
    split = B // len(methods)
    parts = []
    for i, m in enumerate(methods):
        raw = _REGISTRY[m](view).to(static_td.device)
        start = i * split
        end = (i + 1) * split if i < len(methods) - 1 else B
        parts.append(raw[start:end])
    max_L = max(p.shape[1] for p in parts)
    parts = [F.pad(p, (0, 0, 0, max_L - p.shape[1])) for p in parts]
    return torch.cat(parts, dim=0)
