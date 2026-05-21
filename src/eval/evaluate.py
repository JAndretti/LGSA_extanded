"""Periodic evaluation: greedy SA rollout on fixed test instances."""

import csv
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from tensordict import TensorDict

from src.cvrp.init_heuristics import generate_init_solutions
from src.cvrp.state import init_state
from src.data.loader import load_eval_instances
from src.sa.loop import sa_collect
from src.sa.temperature import make_schedule


def evaluate(
    actor,
    test_static: TensorDict,
    init_sol: torch.Tensor,
    cfg: dict,
    autocast_ctx=None,
) -> dict:
    """SA on the fixed test set. Returns aggregated metrics."""
    state = init_state(test_static, solution=init_sol)
    sched = make_schedule(
        cfg["eval"]["schedule"],
        T_max=cfg["sa"]["init_temp"],
        T_min=cfg["sa"]["stop_temp"],
        step_max=cfg["eval"]["outer_steps"],
    )
    ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    with torch.inference_mode(), ctx:
        res = sa_collect(
            actor,
            state,
            sched,
            cfg,
            total_steps=cfg["eval"]["outer_steps"],
            train=False,
            greedy=cfg["eval"]["greedy"],
        )
    return {
        "eval/min_cost": res["best_cost"].mean().item(),
        "eval/init_cost": res["init_cost"].mean().item(),
        "eval/gain": (res["init_cost"] - res["best_cost"]).mean().item(),
        "eval/acceptance": (res["n_accepted"] / cfg["eval"]["outer_steps"])
        .mean()
        .item(),
        "eval/valid_pct": res["n_valid_pct"],
        "eval/best_step": (res["best_step"] / cfg["eval"]["outer_steps"]).mean().item(),
    }


def run_final_sa(
    actor,
    cfg: dict,
    device: torch.device,
    save_dir: Path | str,
    autocast_ctx=None,
) -> Optional[Path]:
    """Run a long-horizon SA on pre-generated instances and dump results to CSV.

    Driven by cfg["final_eval"]: dim, max_load, n_problems, outer_steps, init,
    schedule, greedy. Loads from the same generated_problems directory used by
    periodic eval, slicing the first `n_problems` rows.

    The CSV is written to `save_dir/final_sa_dim{dim}_steps{outer_steps}.csv`
    with per-problem rows plus a single summary row at the bottom.

    Returns the CSV path, or None if no matching dataset exists / disabled.
    """
    fcfg = cfg.get("final_eval")
    if not fcfg or not fcfg.get("enabled", False):
        logger.info("Final SA evaluation disabled (cfg.final_eval.enabled=false).")
        return None

    # Build a config view that load_eval_instances + sa_collect will read.
    eval_overrides = {
        "dim": fcfg["dim"],
        "outer_steps": fcfg["outer_steps"],
        "schedule": fcfg["schedule"],
        "init": fcfg["init"],
        "greedy": fcfg["greedy"],
        "n_problems": fcfg["n_problems"],
        "instances_dir": cfg["eval"].get("instances_dir", "generated_problems"),
    }
    view_cfg = {
        **cfg,
        "eval": {**cfg["eval"], **eval_overrides},
        "problem": {**cfg["problem"], "max_load": fcfg["max_load"]},
    }

    test_static = load_eval_instances(view_cfg, device)
    if test_static is None:
        logger.warning(
            f"Final SA skipped: no pre-generated file for "
            f"dim={fcfg['dim']}, max_load={fcfg['max_load']}."
        )
        return None

    sol = generate_init_solutions(
        test_static, [fcfg["init"]], False, cfg["problem"]["max_routes_estimate"]
    )
    state = init_state(test_static, sol)
    sched = make_schedule(
        fcfg["schedule"],
        T_max=cfg["sa"]["init_temp"],
        T_min=cfg["sa"]["stop_temp"],
        step_max=fcfg["outer_steps"],
    )
    ctx = autocast_ctx if autocast_ctx is not None else nullcontext()

    n = test_static.batch_size[0]
    logger.info(
        f"Final SA: {n} problems × {fcfg['outer_steps']} steps "
        f"(dim={fcfg['dim']}, load={fcfg['max_load']}, greedy={fcfg['greedy']})..."
    )
    start = time.perf_counter()
    with torch.inference_mode(), ctx:
        res = sa_collect(
            actor,
            state,
            sched,
            view_cfg,
            total_steps=fcfg["outer_steps"],
            train=False,
            greedy=fcfg["greedy"],
        )
    elapsed = time.perf_counter() - start

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"final_sa_dim{fcfg['dim']}_steps{fcfg['outer_steps']}.csv"

    init_cost = res["init_cost"].detach().cpu()
    best_cost = res["best_cost"].detach().cpu()
    best_step = res["best_step"].detach().cpu()
    n_acc = res["n_accepted"].detach().cpu()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "problem_idx",
                "init_cost",
                "best_cost",
                "gain",
                "best_step",
                "n_accepted",
            ]
        )
        for i in range(n):
            w.writerow(
                [
                    i,
                    f"{init_cost[i].item():.6f}",
                    f"{best_cost[i].item():.6f}",
                    f"{(init_cost[i] - best_cost[i]).item():.6f}",
                    int(best_step[i].item()),
                    int(n_acc[i].item()),
                ]
            )
        # Aggregate footer row (marked with a sentinel id)
        w.writerow(
            [
                "AGGREGATE",
                f"{init_cost.mean().item():.6f}",
                f"{best_cost.mean().item():.6f}",
                f"{(init_cost - best_cost).mean().item():.6f}",
                f"{best_step.float().mean().item():.2f}",
                f"{n_acc.float().mean().item():.2f}",
            ]
        )
        w.writerow([])
        w.writerow(["# n_problems", n])
        w.writerow(["# outer_steps", fcfg["outer_steps"]])
        w.writerow(["# elapsed_seconds", f"{elapsed:.3f}"])
        w.writerow(["# valid_pct_mean", f"{res['n_valid_pct']:.4f}"])

    logger.info(
        f"Final SA done in {elapsed:.1f}s | "
        f"mean init={init_cost.mean().item():.4f} | "
        f"mean best={best_cost.mean().item():.4f} | "
        f"mean gain={(init_cost - best_cost).mean().item():.4f}"
    )
    logger.info(f"Saved per-problem results to {csv_path}")
    return csv_path
