# PPO on CartPole

A clean PPO baseline on CartPole-v1 built on a PyTorch training template. Converges to maximum reward (500) by epoch ~20. Designed as a solid starting point to extend to harder environments and algorithms.

## Project structure

```
src/
├── train.py              # Main training loop (PPO)
├── sweep.py              # Multi-GPU HP sweep runner
├── utils.py              # Shared utilities
├── HP/
│   ├── HP.yaml           # Hyperparameters & experiment config
│   └── HP_sweep.yaml     # Sweep grid definition
├── algo/
│   └── rollout.py        # RolloutBuffer + compute_gae
├── env/
│   └── wrappers.py       # make_vec_env (SyncVectorEnv)
├── model/
│   └── model.py          # ActorCritic (shared trunk, policy + value heads)
└── eval/
```

## Setup

```bash
uv sync
```

Set your W&B API key in `.env`:

```
WANDB_API_KEY=your_key_here
```

## Usage

**Single training run:**

```bash
make train
```

**With CLI overrides** (dotted-path keys matching `HP.yaml`):

```bash
uv run python -m src.train --optimizer.lr 1e-3 --training.seed 1337
```

**HP sweep** (grid or random, parallelised across available GPUs):

```bash
make sweep
```

**Resume from a checkpoint:**

```yaml
# HP.yaml
resume:
  checkpoint: path/to/checkpoint.pt
```

## Algorithm

PPO with vectorised environments. Each epoch is one update cycle:

1. **Collect** `n_steps × n_envs` transitions via `SyncVectorEnv`
2. **Compute** GAE advantages and lambda-returns
3. **Update** for `ppo_epochs` passes over shuffled mini-batches using the clipped surrogate objective

The compiled step function (`torch.compile`) handles: clipped surrogate loss + MSE value loss + entropy bonus, gradient clipping, and optimizer step.

## Configuration

All hyperparameters live in `src/HP/HP.yaml`:

| Section | Key | Description |
|---|---|---|
| `wandb` | `mode` | `online` / `offline` / `disabled` |
| `training` | `num_epochs`, `eval_every`, `eval_episodes`, `seed`, `device` | Training loop |
| `model` | `hidden_dim` | Shared trunk width |
| `optimizer` | `lr`, `weight_decay`, `grad_clip` | Optimiser settings |
| `env` | `env_id`, `obs_dim`, `act_dim`, `n_envs`, `n_steps`, `gamma`, `gae_lambda`, `ppo_epochs`, `batch_size`, `clip_range`, `entropy_coef`, `value_coef` | Environment & PPO |
| `checkpointing` | `top_k`, `monitor`, `mode` | Checkpoint retention |

`device` accepts `"auto"` (cuda → mps → cpu), `"cuda"`, `"mps"`, or `"cpu"`.

## HP sweeps

Define the sweep grid in `src/HP/HP_sweep.yaml` using dotted paths:

```yaml
optimizer.lr: [1.0e-4, 3.0e-4, 1.0e-3]
model.hidden_dim: [64, 128]
training.seed: [42, 1337]
```

Multiple YAML documents (`---`) define distinct sweep groups. Set `SWEEP_MODE = "grid"` or `"random"` at the top of `sweep.py`. Runs are distributed one-per-GPU automatically.

## Extending to a new environment

1. Update `HP.yaml`: `env_id`, `obs_dim`, `act_dim`
2. For continuous actions: swap `Categorical` → `Normal` in `ActorCritic.act` / `evaluate`
3. For expensive envs: switch `SyncVectorEnv` → `AsyncVectorEnv` in `src/env/wrappers.py`
4. For observation normalization: add `gymnasium.wrappers.NormalizeObservation` inside `make_vec_env`

## Extending to a new algorithm

- Rollout collection (`collect_rollout`) and the model are reusable
- Replace `src/algo/rollout.py` and `ppo_update` with your algorithm's buffer and update logic

## Checkpointing

`TopKCheckpointManager` retains the K best checkpoints by a monitored metric:

```yaml
checkpointing:
  top_k: 3
  monitor: "eval/mean_reward"
  mode: "max"   # or "min" for loss
```

Full training state is saved (model, optimiser, epoch, all RNG states) so runs can be resumed exactly.

## Dependencies

- Python 3.13+
- PyTorch 2.11+
- gymnasium[classic-control]
- W&B, loguru, numpy, tqdm, pyyaml
