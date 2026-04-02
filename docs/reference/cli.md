# CLI reference

Top-level scripts (run from repository root):

| Script | Purpose |
|--------|---------|
| `train.py` | Train PPO/A2C from `configs/train/*.yaml` |
| `quantize.py` | Quantization pipeline |
| `distill.py` | Distillation |
| `prune.py` | Pruning / LUT-oriented compression |
| `codegen.py` | MCU-oriented codegen (CMake, Rust, Arduino, TVM placeholder) |
| `verify_cli.py` | Verification CLI |
| `monitor_cli.py` | Monitoring utilities |
| `dispatcher_cli.py` | Dispatcher utilities |
| `benchmark_harness.py` | Compare model checkpoints |

```bash
python <script>.py --help
```

The **`ui/`** app is a separate Node project (`npm run dev`, `npm run build`, …).
