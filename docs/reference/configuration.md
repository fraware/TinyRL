# Configuration

## Training

YAML files in **`configs/train/`** drive the root **`train.py`** script. The file is loaded with OmegaConf; the CLI applies explicit flags (`--timesteps`, `--seed`, `--no-wandb`, `--output-dir`, etc.).

There is no separate `configs/quantization.yaml` in the repository today—quantization, pruning, and codegen are configured via their Python APIs and CLI arguments (`python quantize.py --help`, and so on).

## In-code configs

Dataclasses such as `QuantizationConfig`, `CodegenConfig`, and `DispatcherConfig` live in their respective modules under `tinyrl/`. Use them when calling library APIs directly.

## Environment extras

Optional dependency groups in `pyproject.toml`:

| Extra | Use |
|-------|-----|
| `dev` | pytest, Ruff, Mypy, Bandit, pre-commit, etc. |
| `docs` | MkDocs, Material, mkdocstrings |
| `atari` | Atari ROM tooling |
| `mujoco` | MuJoCo |
| `embedded` | pyserial, pyocd |
| `verify` | z3-solver |

Install with `pip install -e ".[extra]"` or combine: `pip install -e ".[dev,docs]"`.
