"""LG-SA training entrypoint.

Coordinates per epoch:
  (a) regenerate instances; (b) build initial solutions (with multi-init schedule);
  (c) optional curriculum pretraining (greedy SA, no policy update);
  (d) SA collection (writes transitions); (e) GAE; (f) PPO update;
  (g) periodic eval + checkpointing.
"""

import contextlib
import os
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from loguru import logger
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

import wandb
from src.cvrp.init_heuristics import generate_init_solutions
from src.cvrp.instances import generate_random_instances
from src.cvrp.state import init_state
from src.data.loader import load_eval_instances
from src.eval.evaluate import evaluate, run_final_sa
from src.model import build_actor, build_critic
from src.ppo.adaptive import AdaptiveKLController
from src.ppo.update import LGSAPPOLoss, make_advantage_module, ppo_update
from src.sa.curriculum import active_init_methods, curriculum_pretrain_steps
from src.sa.loop import sa_collect
from src.sa.temperature import make_schedule
from src.utils import (
    TopKCheckpointManager,
    apply_overrides,
    autocast_dtype,
    get_device,
    get_script_arguments,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seed,
    setup_logger,
)


def _resolve_precision(request: str, device: torch.device) -> torch.dtype | None:
    """fp32 -> None (no autocast); fp16/bf16 -> dtype; auto -> autocast_dtype(device)."""
    if request == "fp32":
        return None
    if request == "fp16":
        return torch.float16
    if request == "bf16":
        return torch.bfloat16
    if request == "auto":
        return autocast_dtype(device)
    raise ValueError(f"Unknown precision: {request}")


def _guard_mps(device: torch.device, dtype, compile_on: bool) -> None:
    if device.type == "mps" and dtype is not None and compile_on:
        raise ValueError(
            "MPS does not support torch.compile + non-fp32 precision. "
            "Set training.compile=false or training.precision='fp32'."
        )


def main():
    setup_logger("logs")
    torch.set_float32_matmul_precision("high")

    cfg = load_config()
    overrides = get_script_arguments()
    if overrides:
        apply_overrides(cfg, overrides)
        logger.info(f"Applied CLI overrides: {overrides}")
    wcfg, tcfg, ccfg = cfg["wandb"], cfg["training"], cfg["checkpointing"]
    set_seed(tcfg["seed"])

    device = get_device(tcfg["device"])
    target_dtype = _resolve_precision(tcfg["precision"], device)
    _guard_mps(device, target_dtype, tcfg["compile"])
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=target_dtype)
        if target_dtype is not None
        else contextlib.nullcontext()
    )
    logger.info(
        f"Device {device} | precision {tcfg['precision']} | compile {tcfg['compile']}"
    )

    scaler = torch.amp.GradScaler() if tcfg["precision"] == "fp16" else None

    # Organise W&B runs as wandb/<project>/<group>/wandb/run-<id>/...
    # (the inner `wandb/` is a hardcoded W&B convention and can't be removed)
    wandb_dir = Path("wandb") / wcfg["project"] / wcfg["group"]
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        mode=wcfg["mode"],
        project=wcfg["project"],
        group=wcfg["group"],
        name=wcfg["name"],
        tags=wcfg.get("tags", []),
        notes=wcfg.get("notes", ""),
        config=cfg,
        dir=str(wandb_dir),
    )
    logger.info(f"W&B run dir: {run.dir}")

    # Prefer pre-generated eval instances matching (dim, max_load) if available;
    # fall back to fresh random generation otherwise.
    test_static = load_eval_instances(cfg, device)
    if test_static is None:
        test_static = generate_random_instances(
            cfg["eval"]["n_problems"],
            cfg["eval"]["dim"],
            seed=tcfg["seed"] + 10_000,
            device=device,
            max_load=cfg["problem"]["max_load"],
        )

    actor = build_actor(cfg, device)
    critic = build_critic(cfg, device)
    if device.type != "mps" and wcfg["mode"] != "disabled":
        wandb.watch([actor, critic], log="all", log_freq=100)
    if tcfg["compile"]:
        actor = cast(nn.Module, torch.compile(actor))
        critic = cast(nn.Module, torch.compile(critic))
    logger.info(
        f"Actor params: {sum(p.numel() for p in actor.parameters()):,} | "
        f"Critic params: {sum(p.numel() for p in critic.parameters()):,}"
    )

    actor_opt = torch.optim.Adam(
        actor.parameters(),
        lr=cfg["ppo"]["lr_actor"],
        weight_decay=cfg["ppo"]["weight_decay"],
        fused=(device.type == "cuda"),
    )
    critic_opt = torch.optim.Adam(
        critic.parameters(),
        lr=cfg["ppo"]["lr_critic"],
        weight_decay=cfg["ppo"]["weight_decay"],
        fused=(device.type == "cuda"),
    )
    critic_scheduler = ExponentialLR(critic_opt, gamma=0.985)

    ppo_loss = LGSAPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg["ppo"]["clip_range"],
        entropy_coeff=cfg["ppo"]["entropy_coef"],
        critic_coeff=cfg["ppo"]["value_coef"],
        functional=False,
    )
    advantage_module = make_advantage_module(critic, cfg)
    adaptive = AdaptiveKLController(
        target_kl=cfg["ppo"]["target_kl"],
        beta_kl=cfg["ppo"]["beta_kl"],
        beta_min=cfg["ppo"]["beta_min"],
        beta_max=cfg["ppo"]["beta_max"],
        lr_min=cfg["ppo"]["lr_min"],
        lr_max=cfg["ppo"]["lr_max"],
    )

    init_eval = evaluate(actor, test_static, cfg, autocast_ctx=autocast_ctx)
    a_min_cost = init_eval["eval/min_cost"]
    best_eval_loss = float("inf")
    early_stop = 0
    logger.info(f"Baseline eval/min_cost: {a_min_cost:.4f}")

    ckpt_manager = TopKCheckpointManager(top_k=ccfg["top_k"], mode=ccfg["mode"])
    ckpt_manager.setup()

    # Full training-state bundles used for both save and resume.
    _ckpt_models = {"actor": actor, "critic": critic}
    _ckpt_optimizers = {"actor_opt": actor_opt, "critic_opt": critic_opt}
    _ckpt_schedulers = {"critic_scheduler": critic_scheduler}
    _ckpt_extra = {"adaptive": adaptive}

    start_epoch = 0
    resume_path = cfg.get("resume", {}).get("checkpoint")
    if resume_path:
        start_epoch = load_checkpoint(
            resume_path,
            models=_ckpt_models, optimizers=_ckpt_optimizers,
            schedulers=_ckpt_schedulers, extra=_ckpt_extra,
            device=device,
        )
        # The adaptive controller's beta_kl is restored; propagate back to cfg
        # so the rest of the loop reads the resumed value.
        cfg["ppo"]["beta_kl"] = adaptive.beta_kl

    save_period = cfg["eval"]["save_period"]
    pbar = tqdm(range(start_epoch + 1, tcfg["num_epochs"] + 1), unit="epoch")

    for epoch in pbar:
        # A. Instances for this epoch
        train_static = generate_random_instances(
            cfg["problem"]["n_problems"],
            cfg["problem"]["dim"],
            seed=tcfg["seed"] + epoch,
            device=device,
            max_load=cfg["problem"]["max_load"],
        )

        # B. Initial solutions
        active = active_init_methods(epoch, cfg)
        sol0 = generate_init_solutions(
            train_static,
            active,
            cfg["sa"]["multi_init"],
            cfg["problem"]["max_routes_estimate"],
        )
        state = init_state(train_static, sol0)

        # C. Curriculum pretraining (greedy SA, no transition collection)
        pre_steps = 0
        if cfg["curriculum"]["enabled"]:
            t_init = curriculum_pretrain_steps(epoch, cfg)
            if t_init > 0:
                pre = sa_collect(
                    actor,
                    state,
                    make_schedule(
                        cfg["sa"]["schedule"],
                        cfg["sa"]["init_temp"],
                        cfg["sa"]["stop_temp"],
                        t_init,
                    ),
                    cfg,
                    total_steps=t_init,
                    train=False,
                    greedy=True,
                    autocast_ctx=autocast_ctx,
                )
                state = init_state(train_static, pre["best_solution"])
                pre_steps = t_init

        # D. Collection
        sched = make_schedule(
            cfg["sa"]["schedule"],
            cfg["sa"]["init_temp"],
            cfg["sa"]["stop_temp"],
            cfg["sa"]["outer_steps"],
        )
        sa_results = sa_collect(
            actor,
            state,
            sched,
            cfg,
            total_steps=cfg["sa"]["outer_steps"],
            train=True,
            epoch=epoch,
            autocast_ctx=autocast_ctx,
        )
        transitions = sa_results["transitions"]

        # E. GAE + advantage normalization
        with torch.no_grad():
            advantage_module(transitions)
        adv = transitions["advantage"]
        transitions["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        # F. PPO update
        train_metrics = ppo_update(
            actor,
            critic,
            ppo_loss,
            actor_opt,
            critic_opt,
            adaptive,
            transitions,
            cfg,
            autocast_ctx=autocast_ctx,
            scaler=scaler,
        )
        critic_scheduler.step()
        cfg["ppo"]["beta_kl"] = adaptive.beta_kl

        # G. Logging + periodic eval/checkpoint
        log_dict = {
            "epoch": epoch,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            "train/pre_step": pre_steps,
            "train/avg_reward": sa_results["avg_reward"].item(),
            "train/best_cost": sa_results["best_cost"].mean().item(),
        }
        if epoch % save_period == 0:
            eval_metrics = evaluate(actor, test_static, cfg, autocast_ctx=autocast_ctx)
            a_min_cost = min(a_min_cost, eval_metrics["eval/min_cost"])
            log_dict.update(eval_metrics)
            log_dict["eval/a_min_cost"] = a_min_cost
            log_dict["early_stopping_counter"] = early_stop

            ckpt_manager.update(
                epoch, eval_metrics["eval/min_cost"],
                models=_ckpt_models, optimizers=_ckpt_optimizers,
                schedulers=_ckpt_schedulers, extra=_ckpt_extra,
            )
            save_checkpoint(
                os.path.join(run.dir, "latest_checkpoint.pt"),
                models=_ckpt_models, optimizers=_ckpt_optimizers,
                schedulers=_ckpt_schedulers, extra=_ckpt_extra,
                epoch=epoch,
            )

            if eval_metrics["eval/min_cost"] < best_eval_loss:
                best_eval_loss = eval_metrics["eval/min_cost"]
                early_stop = 0
            else:
                early_stop += 1

        wandb.log(log_dict, step=epoch)

        if early_stop > tcfg["early_stop_patience"] and (
            not cfg["curriculum"]["enabled"]
            or epoch >= cfg["curriculum"]["max_prob_step"]
        ):
            logger.warning(f"Early stop at epoch {epoch}")
            break

        pbar.set_description(
            f"Epoch {epoch} | best_eval {best_eval_loss:.4f} | ES {early_stop}"
        )

    # Final long-horizon SA on pre-generated instances. Saves CSV alongside
    # checkpoints in the W&B run dir.
    run_final_sa(
        actor=actor,
        cfg=cfg,
        device=device,
        save_dir=run.dir,
        autocast_ctx=autocast_ctx,
    )

    wandb.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
