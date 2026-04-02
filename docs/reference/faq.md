# FAQ

**Does TinyRL ship a prebuilt MCU binary?**  
No. Codegen produces sources and build files; you compile with your vendor toolchain (for example `arm-none-eabi-gcc`).

**How do I install for development?**  
`pip install -e ".[dev,docs]"` from the repo root, then `pre-commit install`. See [CONTRIBUTING.md](https://github.com/fraware/TinyRL/blob/main/CONTRIBUTING.md).

**Is Lean required?**  
No. Optional Z3-related tooling is available via `pip install -e ".[verify]"`. Lean-oriented content in the repo is illustrative unless you add a full Lake project.

**Does the root `train.py` accept Hydra-style `foo=bar` on the command line?**  
No. It loads YAML with OmegaConf and uses explicit flags such as `--timesteps` and `--seed`. See [Training](../user_guide/training.md).

**Can I use the UI without the Python backend?**  
The UI is a separate app; local workflows often run `train.py` and APIs independently. Align any custom deployment with the contracts your UI expects.

**Where is CI defined?**  
[`.github/workflows/ci-python.yml`](https://github.com/fraware/TinyRL/blob/main/.github/workflows/ci-python.yml) and [`ci-ui.yml`](https://github.com/fraware/TinyRL/blob/main/.github/workflows/ci-ui.yml).
