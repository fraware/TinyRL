# Quick Start Guide

Train, compress, and prepare MCU-oriented artifacts with TinyRL. Run commands from the **repository root** unless noted.

## Installation

### Prerequisites

- **Python 3.10+** (see `requires-python` in [`pyproject.toml`](https://github.com/fraware/TinyRL/blob/main/pyproject.toml))
- **Git**
- **Node.js 20+** (for the `ui/` app only)
- **ARM GCC** (`arm-none-eabi-gcc`) only when you compile generated C/C++ for a real MCU (optional for Python-only workflows)

### Install TinyRL

```bash
git clone https://github.com/fraware/TinyRL.git
cd TinyRL

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# Runtime (training stack)
pip install -e .

# Developers: lint, tests, docs, security tooling
pip install -e ".[dev,docs]"
```

Flat requirements files are optional mirrors:

- `requirements.txt` → editable install of the package (`-e .`)
- `requirements-dev.txt` → `pip install -e ".[dev,docs]"`

Optional extras: `atari`, `mujoco`, `embedded`, `verify` (see `pyproject.toml`).

## Your First Agent (CartPole)

### Step 1: Train

```bash
python train.py --config configs/train/ppo_cartpole.yaml --no-wandb
```

Artifacts go under the output directory from your YAML (often `outputs/...`). Use `--timesteps`, `--seed`, and `--no-wandb` as needed.

### Step 2: Distillation (optional)

```bash
python distill.py outputs/ppo_cartpole/final_model.zip CartPole-v1
```

### Step 3: Quantization

```bash
python quantize.py outputs/ppo_cartpole/final_model.zip CartPole-v1
```

### Step 4: Pruning

```bash
python prune.py outputs/ppo_cartpole/final_model.zip CartPole-v1
```

### Step 5: Code generation

```bash
python codegen.py outputs/ppo_cartpole/final_model.zip CartPole-v1
```

This emits CMake/Make sketches, Rust/Arduino scaffolding, and related metadata. You still need a vendor toolchain to produce a flashable binary.

### Step 6: Verification (optional)

```bash
python verify_cli.py --epsilon 0.05
```

For Z3-oriented helpers: `pip install -e ".[verify]"`.

### Step 7: Benchmarks (optional)

```bash
python benchmark_harness.py --model-paths outputs/ppo_cartpole/final_model.zip
```

## Configuration

Training hyperparameters live in **`configs/train/*.yaml`**. The root `train.py` loads that YAML with [OmegaConf](https://omegaconf.readthedocs.io/) and applies **explicit CLI flags** (for example `--timesteps`, `--seed`, `--no-wandb`, `--output-dir`), not Hydra-style `key=value` suffixes on the same command line.

Quantization, pruning, and codegen settings are primarily controlled through their Python APIs and CLIs; use `python quantize.py --help` (and similar) for options.

## Web UI

```bash
cd ui
npm ci
npm run dev
```

## Building the docs site

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Troubleshooting

- **Import errors** — `pip install -e .` from the repo root.
- **Wrong paths** — Run `train.py` and other root scripts from the repository root so `configs/` resolves.
- **UI build** — Node 20+, `npm ci` in `ui/`.
- **CI expectations** — See [CI/CD](development/cicd.md) and [Testing](development/testing.md).

## Next steps

- [User guide: Training](user_guide/training.md)
- [API reference](api/models.md)
