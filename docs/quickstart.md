# Quick Start Guide

Get up and running with TinyRL in minutes. This guide will walk you through training, quantizing, and deploying a reinforcement learning agent on a microcontroller.

## Installation

### Prerequisites

- Python 3.9+
- Git
- ARM GCC toolchain (for MCU deployment)

### Install TinyRL

```bash
# Clone the repository
git clone https://github.com/fraware/TinyRL.git
cd TinyRL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Your First TinyRL Agent

### Step 1: Train a PPO Agent

Train a PPO agent on the CartPole environment:

```bash
# Train PPO agent on CartPole
python train.py --config configs/train/ppo_cartpole.yaml

# This will create:
# - outputs/ppo_cartpole/final_model.zip (trained model)
# - outputs/ppo_cartpole/logs/ (training logs)
# - outputs/ppo_cartpole/checkpoints/ (model checkpoints)
```

### Step 2: Knowledge Distillation

Distill the trained model to create a more compact representation:

```bash
# Run knowledge distillation
python distill.py outputs/ppo_cartpole/final_model.zip CartPole-v1

# This creates a distilled model with:
# - Reduced model size
# - Preserved performance
# - Better generalization
```

### Step 3: Quantization

Convert the model to int8 for MCU deployment:

```bash
# Quantize the model
python quantize.py outputs/ppo_cartpole/final_model.zip CartPole-v1

# This produces:
# - Int8 weights and scales
# - Hardware cost analysis
# - Performance validation
```

### Step 4: Critic Pruning

Eliminate the runtime critic using LUT folding:

```bash
# Prune critic and generate LUT
python prune.py outputs/ppo_cartpole/final_model.zip CartPole-v1

# This creates:
# - Pruned actor network
# - Lookup table for critic values
# - Memory-optimized representation
```

### Step 5: Code Generation

Generate MCU-ready code:

```bash
# Generate code for multiple platforms
python codegen.py outputs/ppo_cartpole/final_model.zip CartPole-v1

# This produces:
# - C/C++ code with CMSIS-NN
# - Rust crate (no_std)
# - Arduino library
# - TVM-Micro integration
```

### Step 6: Formal Verification

Verify the quantized model preserves reward ordering:

```bash
# Run formal verification
python verify_cli.py --epsilon 0.05

# This validates:
# - Reward preservation within ε
# - Memory bounds compliance
# - Latency guarantees
```

### Step 7: Benchmark

Measure performance and power consumption:

```bash
# Run comprehensive benchmarks
python benchmark_harness.py --model-paths outputs/ppo_cartpole/final_model.zip

# This measures:
# - Inference latency
# - Power consumption
# - Reward performance
# - Memory usage
```

## Expected Results

After completing the pipeline, you should see:

| Metric | Target | Typical Result |
|--------|--------|----------------|
| **Reward Preservation** | ≥98% | 98.5% |
| **Memory Usage** | ≤32KB | 2.1KB |
| **Inference Latency** | ≤5ms | 0.8ms |
| **Power Consumption** | ≤50mW | 15mW |

## Configuration

### Training Configuration

Edit `configs/train/ppo_cartpole.yaml` to customize training:

```yaml
# Training parameters
training:
  total_timesteps: 50000
  eval_freq: 1000
  save_freq: 5000

# Model architecture
model:
  actor:
    hidden_sizes: [64, 64]
    activation: "tanh"
  critic:
    hidden_sizes: [64, 64]
    activation: "tanh"

# Algorithm parameters
algorithm:
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
```

### Quantization Configuration

Customize quantization in `configs/quantization.yaml`:

```yaml
# Quantization settings
bits: 8
scheme: "symmetric"
per_channel: true

# Hardware constraints
max_flash_size: 32768  # 32KB
max_ram_size: 4096     # 4KB
target_latency_ms: 5.0

# Loss weights
policy_fidelity_weight: 0.7
hardware_cost_weight: 0.3
```

## Advanced Usage

### Multi-Environment Training

Train on multiple environments:

```bash
# Train on CartPole
python train.py --config configs/train/ppo_cartpole.yaml

# Train on LunarLander
python train.py --config configs/train/a2c_lunarlander.yaml

# Compare results
python benchmark_harness.py \
  --model-paths outputs/ppo_cartpole/final_model.zip \
  --model-paths outputs/a2c_lunarlander/final_model.zip
```

### Custom Model Architectures

Define custom model architectures:

```python
from tinyrl.models import PPOActor, PPOCritic

# Custom actor network
actor = PPOActor(
    obs_dim=4,
    action_dim=2,
    hidden_sizes=[128, 128],  # Larger network
    activation="relu",         # Different activation
    std=0.0                   # Learnable std
)

# Custom critic network
critic = PPOCritic(
    obs_dim=4,
    hidden_sizes=[128, 128],
    activation="relu"
)
```

### Hardware-Specific Optimization

Optimize for specific MCU targets:

```bash
# Optimize for Cortex-M55
python codegen.py model.zip env \
  --target-mcu cortex-m55 \
  --target-arch armv8-m.main \
  --max-stack-size 4096 \
  --max-heap-size 28672

# Optimize for Arduino Nano 33 BLE
python codegen.py model.zip env \
  --arduino-board nano33ble \
  --arduino-library-name TinyRL
```

## Troubleshooting

### Common Issues

**Training doesn't converge:**
- Increase `total_timesteps`
- Adjust learning rate
- Check environment configuration

**Quantization degrades performance:**
- Increase `policy_fidelity_weight`
- Use asymmetric quantization
- Adjust temperature in distillation

**Code generation fails:**
- Verify ARM toolchain installation
- Check memory constraints
- Ensure model compatibility

**Verification fails:**
- Increase epsilon tolerance
- Check model bounds
- Verify SMT solver installation

### Getting Help

- **Documentation**: Check the [User Guide](user_guide/) for detailed tutorials
- **Examples**: See [Examples](examples/) for real-world use cases
- **Issues**: Report bugs on [GitHub Issues](https://github.com/fraware/TinyRL/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/fraware/TinyRL/discussions)

## Next Steps

Congratulations! You've successfully trained and deployed a TinyRL agent. Here's what you can explore next:

1. **Try Different Environments**: Experiment with LunarLander, Acrobot, or custom environments
2. **Optimize for Your Hardware**: Customize quantization and code generation for your specific MCU
3. **Explore Advanced Features**: Dive into formal verification, power profiling, and custom architectures
4. **Contribute**: Help improve TinyRL by contributing code, documentation, or examples

---

<div align="center">

**Ready for more advanced features?**  
[User Guide →](user_guide/training.md)

</div> 