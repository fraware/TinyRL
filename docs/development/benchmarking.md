# Benchmarking

## CI

The Python workflow runs **short** `train.py` invocations (reduced timesteps, `--no-wandb`) so configs and imports stay healthy. This is a smoke test, not a performance regression gate.

## Local / research

- Use full `train.py` runs with your YAML and optional W&B logging.
- Use [`benchmark_harness.py`](https://github.com/fraware/TinyRL/blob/main/benchmark_harness.py) to compare checkpoints.
- For reproducibility, set `PYTHONHASHSEED` and `TORCH_DETERMINISTIC` as in CI (see workflow env).

## Future work

Formal regression tracking (stored baselines, `pytest-benchmark`, or statistical thresholds) can be added when you want stricter performance gates.
