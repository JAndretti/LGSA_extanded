"""Sanity check: does the eval SA loop produce capacity-feasible solutions?

Runs the same actor + SA path used in eval.handler_random, then verifies
that the returned `best_solution` is feasible via src.cvrp.geometry.is_feasible
plus an "all customers visited exactly once" check (which is_feasible does
not enforce).

Usage:
  uv run python scripts/check_feasibility.py \
      --project lg-sa-cvrp --group lg-sa-baseline \
      --dim 100 --max_load 50 \
      --n_problems 64 --outer_steps 10000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cvrp.geometry import is_feasible
from src.cvrp.init_heuristics import generate_init_solutions
from src.cvrp.state import _cost_with_solution, init_state
from src.data.loader import load_eval_instances

from eval.eval_io import best_checkpoint_path, find_run_dirs, load_run_config
from eval.solver import (
    autocast_for,
    build_actor_from_config,
    run_sa,
    set_seed,
)


_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _all_customers_once(solution: torch.Tensor, n_customers: int) -> torch.Tensor:
    """For each row, true iff every customer id in 1..n_customers appears exactly once."""
    sol = solution.squeeze(-1)
    B = sol.size(0)
    # Count occurrences of each customer id. Bucket-size n_customers + 1 to
    # include the depot, which we'll just ignore.
    counts = torch.zeros(B, n_customers + 1, dtype=torch.long, device=sol.device)
    counts.scatter_add_(1, sol, torch.ones_like(sol))
    # Customers 1..n_customers must each appear exactly once.
    return (counts[:, 1:] == 1).all(dim=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--max_load", type=int, required=True)
    p.add_argument("--n_problems", type=int, default=64,
                   help="Small subset; we only want feasibility stats.")
    p.add_argument("--outer_steps", type=int, default=10000)
    p.add_argument("--INIT", type=str, default="random")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", type=str, default="float16",
                   choices=list(_DTYPE_MAP.keys()))
    p.add_argument("--device", type=str, default=_auto_device())
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = _DTYPE_MAP[args.dtype]

    run_dirs = find_run_dirs(args.project, args.group)
    if not run_dirs:
        raise SystemExit(f"No runs in wandb/{args.project}/{args.group}/")
    run_dir = run_dirs[0]
    cfg = load_run_config(run_dir)
    ckpt = best_checkpoint_path(run_dir)
    logger.info(f"Using run {run_dir.parent.name}, ckpt {ckpt.name}")
    logger.info(
        f"cfg.problem.update_method={cfg['problem']['update_method']}, "
        f"heuristic={cfg['problem']['heuristic']}, "
        f"trained max_load={cfg['problem']['max_load']}"
    )

    fake_cfg = {
        "eval": {"dim": args.dim, "n_problems": args.n_problems,
                 "instances_dir": "generated_problems"},
        "problem": {"max_load": args.max_load},
    }
    static_td = load_eval_instances(fake_cfg, device, n_problems=args.n_problems)
    if static_td is None:
        raise SystemExit(
            f"No generated_problems/gen_dim{args.dim}_load{args.max_load}.pt"
        )
    n_problems = static_td.batch_size[0]

    actor = build_actor_from_config(cfg, ckpt, device, dtype)
    actor_ctx = autocast_for(device, dtype)

    init_sol = generate_init_solutions(
        static_td, [args.INIT], False, cfg["problem"]["max_routes_estimate"]
    )
    state = init_state(static_td, init_sol)

    # Confirm the initial solution is feasible — sanity check on the setup.
    init_demands = torch.gather(static_td["demands"], 1, init_sol.squeeze(-1))
    init_feasible = is_feasible(init_sol, init_demands, static_td["capacity"])
    init_complete = _all_customers_once(init_sol, args.dim)
    logger.info(
        f"Initial solution: feasible={init_feasible.float().mean().item():.3f}, "
        f"complete={init_complete.float().mean().item():.3f}"
    )

    with torch.inference_mode():
        res = run_sa(
            actor, state, cfg,
            outer_steps=args.outer_steps,
            greedy=False, metropolis=True,
            autocast_ctx=actor_ctx,
        )
    best_sol = res["best_solution"]
    best_cost = res["best_cost"]

    # Independent re-cost on the returned solution. Should match best_cost.
    recost = _cost_with_solution(state, best_sol)
    cost_match = torch.allclose(recost.float(), best_cost.float(), atol=1e-3)

    # Feasibility: capacity (is_feasible) + completeness (all customers once).
    sol_demands = torch.gather(static_td["demands"], 1, best_sol.squeeze(-1))
    cap_ok = is_feasible(best_sol, sol_demands, static_td["capacity"])
    complete = _all_customers_once(best_sol, args.dim)
    fully_ok = cap_ok & complete

    # Per-row max route load — diagnostic for capacity violations.
    sol_flat = best_sol.squeeze(-1)
    is_depot = sol_flat == 0
    seg_start = is_depot & ~torch.cat(
        [torch.zeros_like(is_depot[:, :1]), is_depot[:, :-1]], dim=1
    )
    # Simpler: segment_ids treating non-depot runs.
    nondepot_mask = sol_flat != 0
    seg_start_nd = nondepot_mask & ~torch.cat(
        [torch.zeros_like(nondepot_mask[:, :1]), nondepot_mask[:, :-1]], dim=1
    )
    seg_ids = torch.cumsum(seg_start_nd, dim=1) * nondepot_mask
    L = sol_flat.size(1)
    route_loads = torch.zeros(n_problems, L + 1, device=device,
                              dtype=sol_demands.dtype)
    route_loads.scatter_add_(1, seg_ids, sol_demands)
    max_load_per_row = route_loads.max(dim=1).values
    cap = static_td["capacity"].squeeze(-1)
    overflow = (max_load_per_row.float() - cap.float()).clamp(min=0)

    # Number of routes per row (count of non-empty segments).
    n_routes = (route_loads > 0).sum(dim=1)

    logger.info("─" * 60)
    logger.info(f"best_cost (mean): {best_cost.mean().item():.4f}")
    logger.info(f"recomputed cost matches: {cost_match}")
    logger.info(f"capacity_ok rate:   {cap_ok.float().mean().item():.3f}")
    logger.info(f"complete rate:      {complete.float().mean().item():.3f}")
    logger.info(f"fully_feasible:     {fully_ok.float().mean().item():.3f}")
    logger.info(
        f"max_route_load:     mean={max_load_per_row.float().mean().item():.2f} "
        f"max={max_load_per_row.max().item()} (capacity={cap[0].item()})"
    )
    logger.info(
        f"capacity overflow:  mean={overflow.mean().item():.2f} "
        f"max={overflow.max().item()}"
    )
    logger.info(
        f"n_routes per sol:   mean={n_routes.float().mean().item():.2f} "
        f"min={n_routes.min().item()} max={n_routes.max().item()}"
    )

    if fully_ok.all():
        logger.success("All returned best_solutions are feasible CVRP tours.")
    else:
        n_bad = (~fully_ok).sum().item()
        logger.error(
            f"{n_bad}/{n_problems} returned best_solutions are NOT feasible. "
            f"The reported best_cost is meaningless for those rows."
        )


if __name__ == "__main__":
    main()
