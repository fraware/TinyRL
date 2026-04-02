# Quantization

Differentiable int8 quantization reduces model size for MCU deployment while preserving policy behavior.

Prerequisites: `pip install -e .` from the repo root (includes ONNX stack per `pyproject.toml`).

## Workflow

1. Train a full-precision policy (`train.py`).
2. Run the quantization pipeline (`quantize.py` or Python API).
3. Validate reward delta against thresholds in `QuantizationConfig`.

See the [Quantization API](../api/quantization.md) for `DifferentiableQuantizer`, ONNX helpers, and reports.
