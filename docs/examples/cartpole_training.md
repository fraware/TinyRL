# CartPole training

From the repository root, with `pip install -e .`:

```bash
python train.py --config configs/train/ppo_cartpole.yaml --no-wandb
```

Outputs default to paths configured in the YAML (commonly under `outputs/`). Override timesteps or seed with `--timesteps` and `--seed`.

More detail: [Training user guide](../user_guide/training.md).
