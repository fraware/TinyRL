# Training

TinyRL trains **PPO** and **A2C** agents with YAML configs under `configs/train/` and optional [Weights & Biases](https://wandb.ai) logging.

## CLI (`train.py` at repo root)

```bash
python train.py --config configs/train/ppo_cartpole.yaml
```

Common flags:

| Flag | Purpose |
|------|---------|
| `--config` | Path to YAML under `configs/train/` |
| `--timesteps` | Override `training.total_timesteps` |
| `--seed` | Random seed |
| `--no-wandb` | Disable W&B (recommended for CI and offline runs) |
| `--output-dir` | Override output directory |
| `--eval-only` | Evaluate an existing checkpoint (`--model-path` required) |

The CLI loads the YAML with OmegaConf and applies these overrides. For programmatic use, import `Trainer` from `tinyrl.train`.

## Python API

See the [Training API reference](../api/training.md) for `Trainer` and related types.

## Configuration files

Only training YAMLs ship in-repo today (`configs/train/ppo_cartpole.yaml`, `configs/train/a2c_lunarlander.yaml`). Edit those files or copy them for new experiments.
