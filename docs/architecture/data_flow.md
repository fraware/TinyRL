# Data flow

For a longer narrative, see the [data flow specification](../data_flow_spec.md).

High level:

1. Train policies with Gymnasium + Stable-Baselines3 (`train.py` / `tinyrl.train`).
2. Optional distillation; then quantization and pruning (`distill.py`, `quantize.py`, `prune.py`).
3. Codegen emits MCU-facing projects; compile and verify on hardware with your toolchain.
