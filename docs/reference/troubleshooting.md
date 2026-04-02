# Troubleshooting

**Import errors for `torch`, `onnx`, SB3** — From the repo root run `pip install -e .` so `pyproject.toml` dependencies resolve.

**`train.py` cannot find config** — Run scripts from the repository root; paths like `configs/train/ppo_cartpole.yaml` are relative to it.

**Weights & Biases login / network in CI** — Use `python train.py ... --no-wandb` for offline or automated runs.

**UI build failures** — Use **Node.js 20+**, run `npm ci` inside `ui/`, then `npm run build`.

**Ruff or Mypy version skew** — Match the versions implied by `pip install -e ".[dev]"` / `pyproject.toml`; for hooks, run `pre-commit autoupdate` cautiously and re-run `pre-commit run --all-files`.

**MkDocs / API pages fail** — Install docs extras: `pip install -e ".[docs]"` and ensure `tinyrl` is importable (editable install).

**ARM / codegen** — Generated Make/CMake targets expect `arm-none-eabi-gcc` when you actually compile for Cortex-M. Python codegen and tests can run without it (some tests skip if the toolchain is missing).
