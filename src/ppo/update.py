"""PPO update wired around torchrl's ClipPPOLoss + GAE.

Verified against torchrl 0.12.0 / tensordict 0.12.3 (see Task 0.1).

LGSAPPOLoss overrides _get_cur_log_prob to call our custom
actor.evaluate(), which handles the two-stage factorized policy (c1, c2 | c1).
"""
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from src.ppo.adaptive import AdaptiveKLController


class _FakeDist:
    """Distribution-like stub so ClipPPOLoss._get_entropy(dist) → dist.entropy() works.

    Our two-stage policy doesn't have a single torch.distributions.Distribution
    object. evaluate() already computes the summed entropy, so we just hand it
    back via .entropy().
    """

    def __init__(self, entropy: torch.Tensor):
        self._entropy = entropy

    def entropy(self) -> torch.Tensor:
        return self._entropy


class LGSAPPOLoss(ClipPPOLoss):
    """ClipPPOLoss with the log-prob recompute overridden for our two-stage policy.

    Our actor stores its log-prob under "sample_log_prob" (the default torchrl
    name for non-probabilistic-module actors is "action_log_prob"). We pin our
    key here so the loss reads the right entry from the replay TD.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_keys(sample_log_prob="sample_log_prob")

    def _get_cur_log_prob(self, tensordict):
        log_prob, entropy = self.actor_network.evaluate(tensordict)
        return log_prob, _FakeDist(entropy), False


def make_advantage_module(critic, cfg: dict) -> GAE:
    """GAE module reading 'state_value' from critic and writing 'advantage' + 'value_target'.

    When cfg.ppo.gae_chunk_size > 0 the GAE is built without a value_network and
    expects 'state_value' + ('next','state_value') to be pre-filled by
    compute_advantages() (which calls the critic in chunks). Otherwise GAE owns
    the critic call and runs it on the full (T, B, ...) transitions in one shot.
    """
    chunk_size = int(cfg["ppo"].get("gae_chunk_size", 0))
    return GAE(
        gamma=cfg["ppo"]["gamma"],
        lmbda=cfg["ppo"]["gae_lambda"],
        value_network=None if chunk_size > 0 else critic,
        average_gae=False,
    )


def _prefill_state_value(critic, td, chunk_size: int) -> None:
    """Write td['state_value'] by calling critic on flattened chunks of td."""
    flat = td.reshape(-1)
    n = flat.batch_size[0]
    parts = []
    for i in range(0, n, chunk_size):
        chunk = flat[i:i + chunk_size].clone()
        critic(chunk)
        parts.append(chunk["state_value"])
    state_value = torch.cat(parts, 0).view(*td.batch_size, *parts[0].shape[1:])
    td.set("state_value", state_value)


def compute_advantages(advantage_module, critic, transitions, cfg: dict) -> None:
    """Run GAE on transitions, optionally chunking the critic pre-pass.

    cfg.ppo.gae_chunk_size <= 0: GAE calls the critic on the full (T, B, ...)
    block — fastest when VRAM allows.
    cfg.ppo.gae_chunk_size  > 0: pre-fill state_value for both transitions and
    transitions['next'] in chunks of that many flattened steps, then run GAE
    (which reads the pre-filled values since value_network is None).

    Caller owns the torch.no_grad() context.
    """
    chunk_size = int(cfg["ppo"].get("gae_chunk_size", 0))
    if chunk_size > 0:
        _prefill_state_value(critic, transitions, chunk_size)
        _prefill_state_value(critic, transitions["next"], chunk_size)
    advantage_module(transitions)


def critic_gradient_penalty(critic, obs: torch.Tensor) -> torch.Tensor:
    """(‖∇V/∇obs‖₂ − 1)² .mean(). Ported from LGSA_OLD/src/ppo/ppo.py."""
    obs = obs.detach().clone().requires_grad_(True)
    td = TensorDict({"observation": obs}, batch_size=obs.shape[:1])
    out = critic(td)
    grads = torch.autograd.grad(
        outputs=out["state_value"], inputs=obs,
        grad_outputs=torch.ones_like(out["state_value"]),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grads.norm(2.0, dim=tuple(range(1, grads.dim()))) - 1) ** 2).mean()


def ppo_update(
    actor,
    critic,
    ppo_loss: LGSAPPOLoss,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    adaptive: AdaptiveKLController,
    replay_data: TensorDict,
    cfg: dict,
    autocast_ctx=None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> dict:
    """Run cfg.ppo.ppo_epochs passes of mini-batch updates over `replay_data`.

    `replay_data` is the stacked (T, B, ...) TensorDict returned by sa_collect
    under `results["transitions"]` — not a torchrl ReplayBuffer. It is flattened
    to (T*B, ...) here and shuffled via `torch.randperm` for mini-batching.

    Assumes the GAE module has already been applied to replay_data so that
    "advantage" and "value_target" keys are present. Advantage should be
    normalized over the full batch before calling this.
    """
    flat = replay_data.reshape(-1)
    total = flat.batch_size[0]
    bs = cfg["ppo"]["batch_size"]
    ppo_epochs = cfg["ppo"]["ppo_epochs"]
    grad_clip = cfg["ppo"]["grad_clip"]
    gp_lambda = cfg["ppo"]["gp_lambda"]
    separate_backward = bool(cfg["ppo"].get("separate_backward", False))
    actor_ctx = autocast_ctx if autocast_ctx is not None else nullcontext()

    with torch.no_grad():
        y_pred = flat["state_value"].view(-1)
        y_true = flat["value_target"].view(-1)
        var_y = torch.var(y_true)
        explained_var = (1 - torch.var(y_true - y_pred) / (var_y + 1e-8)).item()

    losses_acc = {"loss_objective": [], "loss_critic": [], "loss_entropy": [], "kl": []}

    for _ in range(ppo_epochs):
        kls = []
        perm = torch.randperm(total, device=flat.device)
        for start in range(0, total, bs):
            batch = flat[perm[start:start + bs]]
            if batch.batch_size[0] <= 1:
                continue
            actor_opt.zero_grad(set_to_none=True)
            critic_opt.zero_grad(set_to_none=True)

            with actor_ctx:
                out = ppo_loss(batch)

            actor_loss = out["loss_objective"]
            if "loss_entropy" in out.keys():
                actor_loss = actor_loss + out["loss_entropy"]
            kl_term = out.get("kl_approx", None)
            if kl_term is not None:
                actor_loss = actor_loss + adaptive.beta_kl * kl_term
                kls.append(float(kl_term.detach()))

            critic_loss = out.get("loss_critic", torch.tensor(0.0, device=batch.device))
            if gp_lambda > 0:
                critic_loss = critic_loss + gp_lambda * critic_gradient_penalty(
                    critic, batch["observation"]
                )

            if separate_backward:
                # Old-style: independent backward+step per network. Actor and
                # critic are separate modules so their graphs don't overlap.
                if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                    continue
                if scaler is not None:
                    scaler.scale(actor_loss).backward()
                    scaler.unscale_(actor_opt)
                    nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
                    scaler.step(actor_opt)
                    scaler.scale(critic_loss).backward()
                    scaler.unscale_(critic_opt)
                    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                    scaler.step(critic_opt)
                    scaler.update()
                else:
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
                    actor_opt.step()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                    critic_opt.step()
            else:
                total_loss = actor_loss + critic_loss
                if torch.isnan(total_loss):
                    continue
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(actor_opt)
                    nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
                    scaler.unscale_(critic_opt)
                    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                    scaler.step(actor_opt); scaler.step(critic_opt); scaler.update()
                else:
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
                    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                    actor_opt.step(); critic_opt.step()

            losses_acc["loss_objective"].append(float(out["loss_objective"].detach()))
            if "loss_critic" in out.keys():
                losses_acc["loss_critic"].append(float(out["loss_critic"].detach()))
            if "loss_entropy" in out.keys():
                losses_acc["loss_entropy"].append(float(out["loss_entropy"].detach()))
            if kl_term is not None:
                losses_acc["kl"].append(float(kl_term.detach()))

        if kls:
            adaptive.update(mean_kl=sum(kls) / len(kls), actor_opt=actor_opt)

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    return {
        "actor_loss":         _mean(losses_acc["loss_objective"]),
        "critic_loss":        _mean(losses_acc["loss_critic"]),
        "entropy":            _mean(losses_acc["loss_entropy"]),
        "mean_kl":            _mean(losses_acc["kl"]),
        "explained_variance": explained_var,
        "beta_kl":            adaptive.beta_kl,
        "lr_actor":           actor_opt.param_groups[0]["lr"],
    }
