# Architecture overview

TinyRL spans training (PyTorch / Stable-Baselines3), optimization (distillation, quantization, pruning), codegen (C / Rust / Arduino scaffolding), optional verification helpers, and a **Next.js** web UI under `ui/`.

- **Packaging**: `pyproject.toml` defines the `tinyrl` distribution and optional extras.
- **Containers**: a root [`Dockerfile`](https://github.com/fraware/TinyRL/blob/main/Dockerfile) targets the Python training CLI stack.

See the diagram on the [home page](../index.md) and [data flow](data_flow.md).
