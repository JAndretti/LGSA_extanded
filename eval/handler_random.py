"""Evaluate every model in a W&B group on a pre-generated CVRP test set."""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cvrp.init_heuristics import generate_init_solutions
from src.cvrp.state import cost, init_state
from src.data.loader import load_eval_instances

from eval.eval_io import (
    _differing_keys,
    _hp_row,
    best_checkpoint_path,
    find_run_dirs,
    load_run_config,
    save_results_csv,
)
from eval.solver import (
    UniformActor,
    augment_coords,
    autocast_for,
    build_actor_from_config,
    run_sa,
    set_seed,
    strip_enrichment,
    warmup_cuda,
)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dim", type=int, default=100, choices=[10, 20, 50, 100, 500, 1000]
    )
    parser.add_argument("--max_load", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument(
        "--no-baseline", dest="BASELINE", action="store_false", default=True
    )
    parser.add_argument("--greedy", dest="GREEDY", action="store_true", default=False)
    parser.add_argument(
        "--no-metro", dest="METROPOLIS", action="store_false", default=True
    )
    parser.add_argument(
        "--augment",
        type=int,
        default=1,
        choices=range(1, 9),
        metavar="K",
        help="Number of D4 augmentations (1=off, 8=full group).",
    )


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)

    # 1. Discover runs and detect swept HPs.
    run_dirs = find_run_dirs(args.project, args.group)
    if not run_dirs:
        raise SystemExit(
            f"No runs in wandb/{args.project}/{args.group}/wandb/. "
            f"Check --project / --group."
        )
    run_cfgs = [load_run_config(d) for d in run_dirs]
    diff_keys = _differing_keys(run_cfgs)
    logger.info(
        f"Found {len(run_dirs)} runs. Swept keys: {sorted(diff_keys) or 'none'}"
    )

    # 2. Load the shared test set from generated_problems/.
    fake_cfg = {
        "eval": {
            "dim": args.dim,
            "n_problems": args.batch_size,
            "instances_dir": "generated_problems",
        },
        "problem": {"max_load": args.max_load},
    }
    static_td = load_eval_instances(fake_cfg, device, n_problems=args.batch_size)
    if static_td is None:
        raise SystemExit(
            f"No file at generated_problems/gen_dim{args.dim}_load{args.max_load}.pt. "
            f"Run: uv run python -m scripts.generate_problems"
        )

    # The loader may have downsized n_problems if the file holds fewer rows
    # than requested. Use the actual count everywhere downstream.
    n_problems = static_td.batch_size[0]
    if n_problems != args.batch_size:
        logger.info(
            f"Effective n_problems = {n_problems} (requested {args.batch_size})."
        )

    warmup_cuda()
    rows: list[dict] = []

    # 3. Per-model loop. The whole per-run body is wrapped: any failure on
    # one run (bad config schema, missing key, weight shape mismatch, etc.)
    # skips that run rather than aborting the sweep.
    for run_dir, cfg in tqdm(list(zip(run_dirs, run_cfgs)), desc="Models", unit="run"):
        run_name = run_dir.parent.name
        try:
            ckpt_path = best_checkpoint_path(run_dir)
            actor = build_actor_from_config(cfg, ckpt_path, device, args.torch_dtype)

            autocast_ctx = autocast_for(device, args.torch_dtype)

            # Initial solution (random per --INIT) on the un-augmented coords.
            init_sol = generate_init_solutions(
                static_td,
                [args.INIT],
                False,
                cfg["problem"]["max_routes_estimate"],
            )
            init_state_td = init_state(static_td, init_sol)
            init_cost = cost(init_state_td).mean().item()

            # 3a. Baseline arm — uniform random sampler under the same SA loop.
            baseline_cost = float("nan")
            baseline_time = float("nan")
            if args.BASELINE:
                t0 = time.perf_counter()
                uniform = UniformActor(method=cfg["problem"]["update_method"])
                with torch.inference_mode():
                    bl_res = run_sa(
                        uniform,
                        init_state(static_td, init_sol.clone()),
                        cfg,
                        outer_steps=args.OUTER_STEPS,
                        greedy=False,
                        metropolis=args.METROPOLIS,
                    )
                baseline_time = time.perf_counter() - t0
                baseline_cost = bl_res["best_cost"].mean().item()

            # 3b. Learned arm with D4 ensemble.
            best_per_problem = torch.full((n_problems,), float("inf"), device=device)
            t0 = time.perf_counter()
            for k in range(args.augment):
                aug_static = strip_enrichment(static_td)
                aug_static["coords"] = augment_coords(static_td["coords"], k)
                init_sol_k = generate_init_solutions(
                    aug_static,
                    [args.INIT],
                    False,
                    cfg["problem"]["max_routes_estimate"],
                )
                state_k = init_state(aug_static, init_sol_k)
                with torch.inference_mode():
                    res_k = run_sa(
                        actor,
                        state_k,
                        cfg,
                        outer_steps=args.OUTER_STEPS,
                        greedy=args.GREEDY,
                        metropolis=args.METROPOLIS,
                        autocast_ctx=autocast_ctx,
                    )
                best_per_problem = torch.minimum(best_per_problem, res_k["best_cost"])
            elapsed = time.perf_counter() - t0
            final_cost = best_per_problem.mean().item()
        except (FileNotFoundError, KeyError, RuntimeError) as e:
            logger.warning(f"Skipping {run_name}: {type(e).__name__}: {e}")
            continue

        rows.append(
            {
                "run": run_name,
                "ckpt": ckpt_path.name,
                "dtype": args.dtype,
                "dim": args.dim,
                "max_load": args.max_load,
                "n_problems": n_problems,
                "outer_steps": args.OUTER_STEPS,
                "augment": args.augment,
                "initial_cost": init_cost,
                "final_cost": final_cost,
                "baseline_cost": baseline_cost,
                "exec_time": elapsed,
                "baseline_time": baseline_time,
                **_hp_row(cfg, diff_keys),
            }
        )

    # 4. Save.
    if not rows:
        raise SystemExit("No runs evaluated successfully.")
    fixed = [
        "run",
        "ckpt",
        "dtype",
        "dim",
        "max_load",
        "n_problems",
        "outer_steps",
        "augment",
        "initial_cost",
        "final_cost",
        "baseline_cost",
        "exec_time",
        "baseline_time",
    ]
    swept_cols = sorted(diff_keys)
    df = pd.DataFrame(rows, columns=fixed + swept_cols)
    out = save_results_csv(
        df,
        args.project,
        args.group,
        filename_stem=f"eval_dim{args.dim}",
    )
    logger.info(f"Saved {len(rows)} rows to {out}")
