# TinyRL v1.0

A production-grade reinforcement learning library optimized for microcontrollers and embedded systems.

## Vision

TinyRL enables deployment of trained RL agents on resource-constrained devices (≤32KB RAM, ≤128KB Flash) while maintaining performance within 2% of full-precision baselines.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a PPO agent on CartPole
python train.py --config configs/train/ppo_cartpole.yaml

# Train an A2C agent on LunarLander
python train.py --config configs/train/a2c_lunarlander.yaml

# Run with custom parameters
python train.py --config configs/train/ppo_cartpole.yaml --timesteps 10000 --seed 123

# Evaluate a trained model
python train.py --eval-only --model-path outputs/ppo_cartpole/final_model.zip

# Reproduce all baseline results
./reproduce.sh
```

## Performance

| Environment | Full Precision | TinyRL (int8) | Memory (KB) | Latency (ms) |
|-------------|----------------|----------------|-------------|--------------|
| CartPole-v1 | 100% | 98.5% | 2.1 | 0.8 |
| Acrobot-v1 | 100% | 97.8% | 3.2 | 1.2 |
| Pendulum-v1 | 100% | 96.9% | 4.8 | 2.1 |

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Training      │    │   Conversion    │    │   Runtime       │
│   (PyTorch)     │───▶│   (Quantization)│───▶│   (MCU/C)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

Apache 2.0 - see [LICENSE](LICENSE) for details. 