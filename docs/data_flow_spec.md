# TinyRL Data Flow Specification

## Overview

This document specifies the data flow, tensor shapes, memory budgets, and worst-case latencies for the TinyRL pipeline.

## Training Layer Data Flow

### Input Tensors
- **Observation**: `[batch_size, obs_dim]` (float32)
- **Action**: `[batch_size, action_dim]` (int64 for discrete, float32 for continuous)
- **Reward**: `[batch_size]` (float32)
- **Done**: `[batch_size]` (bool)

### Model Architecture
```
Actor Network:
  Input: [batch_size, obs_dim]
  Hidden: [batch_size, 64] (ReLU)
  Output: [batch_size, action_dim] (logits for discrete, mean+std for continuous)

Critic Network:
  Input: [batch_size, obs_dim]
  Hidden: [batch_size, 64] (ReLU)
  Output: [batch_size, 1] (value estimate)
```

### Memory Budget (Training)
- **Model Parameters**: ~50KB (float32)
- **Gradients**: ~50KB (float32)
- **Optimizer State**: ~100KB (AdamW)
- **Buffer**: ~10MB (experience replay)
- **Total**: ~10.2MB

## Conversion Layer Data Flow

### Knowledge Distillation
- **Teacher Output**: `[batch_size, action_dim]` (float32)
- **Student Output**: `[batch_size, action_dim]` (float32)
- **KL Divergence Loss**: scalar (float32)
- **Temperature**: 2.0 (hyperparameter)

### Quantization Process
```
Input: float32 weights [out_features, in_features]
↓
Quantize to int8: [-128, 127]
↓
Calculate scales: float32 [out_features]
↓
Output: int8 weights + float32 scales
```

### Memory Budget (Conversion)
- **Quantized Weights**: ~12.5KB (int8, 8x compression)
- **Scales**: ~2KB (float32)
- **LUT Tables**: ~1KB (int8)
- **Total**: ~15.5KB

## Runtime Layer Data Flow

### Inference Pipeline
```
Input: float32 observation [obs_dim]
↓
Quantize to int8: [obs_dim]
↓
Matrix multiply: int8 [hidden_dim] = int8 [obs_dim] × int8 [obs_dim, hidden_dim]
↓
Dequantize: float32 [hidden_dim]
↓
ReLU activation: float32 [hidden_dim]
↓
Output layer: float32 [action_dim]
↓
Softmax (discrete) or sample (continuous): float32 [action_dim]
```

### Memory Budget (Runtime)
- **Stack**: ≤4KB
- **Heap**: ≤28KB
- **Flash (Policy)**: ≤64KB
- **Flash (LUT)**: ≤32KB
- **Flash (HAL)**: ≤32KB
- **Total RAM**: ≤32KB
- **Total Flash**: ≤128KB

## Latency Constraints

### Worst-Case Analysis
| Operation | Cycles (Cortex-M55) | Time (80MHz) |
|-----------|---------------------|--------------|
| Input Quantization | 100 | 1.25µs |
| Matrix Multiply (64×64) | 8,192 | 102.4µs |
| ReLU Activation | 64 | 0.8µs |
| Output Dequantization | 64 | 0.8µs |
| Softmax/Sampling | 128 | 1.6µs |
| **Total Inference** | **8,548** | **106.85µs** |

### Real-Time Constraints
- **Maximum Inference Time**: ≤5ms (target: ≤1ms)
- **Interrupt Latency**: ≤50µs
- **Memory Access**: ≤100ns per word
- **Flash Read**: ≤200ns per word

## Tensor Shape Specifications

### Standard Environments

#### CartPole-v1
- **Observation**: `[4]` (cart position, cart velocity, pole angle, pole angular velocity)
- **Action**: `[2]` (left, right)
- **Hidden**: `[64]`
- **Policy Output**: `[2]` (logits)

#### Acrobot-v1
- **Observation**: `[6]` (cos/sin of two joint angles, angular velocities)
- **Action**: `[3]` (torque: -1, 0, +1)
- **Hidden**: `[64]`
- **Policy Output**: `[3]` (logits)

#### Pendulum-v1
- **Observation**: `[3]` (cos/sin of angle, angular velocity)
- **Action**: `[1]` (continuous torque)
- **Hidden**: `[64]`
- **Policy Output**: `[2]` (mean, log_std)

### Memory Layout

#### Flash Memory Layout
```
0x0000 - 0x3FFF: Policy Weights (16KB)
0x4000 - 0x7FFF: Policy Weights (16KB)
0x8000 - 0x9FFF: Policy Weights (8KB)
0xA000 - 0xBFFF: Scales (8KB)
0xC000 - 0xDFFF: LUT Tables (8KB)
0xE000 - 0xFFFF: HAL Interface (8KB)
```

#### RAM Layout
```
0x20000000 - 0x20000FFF: Stack (4KB)
0x20001000 - 0x20007FFF: Heap (28KB)
```

## Interface Contracts

### TrainingArtifact
```json
{
  "model_path": "string",
  "config": {
    "obs_dim": "integer",
    "action_dim": "integer",
    "hidden_dim": "integer",
    "learning_rate": "float",
    "batch_size": "integer"
  },
  "metrics": {
    "final_reward": "float",
    "episode_count": "integer",
    "training_time": "float"
  },
  "reproducibility_hash": "string"
}
```

### QuantizedPolicyBin
```json
{
  "weights_int8": "base64_encoded_bytes",
  "scales": "base64_encoded_float32_array",
  "lut_tables": "base64_encoded_int8_array",
  "memory_layout": {
    "policy_offset": "integer",
    "scales_offset": "integer",
    "lut_offset": "integer"
  }
}
```

### FormalProofBundle
```json
{
  "lean_theorems": ["string"],
  "smt_queries": ["string"],
  "verification_results": {
    "reward_preservation": "boolean",
    "memory_bounds": "boolean",
    "latency_bounds": "boolean"
  }
}
```

## Error Handling

### Memory Overflow Protection
- **Stack Guard**: Check stack pointer before function calls
- **Heap Guard**: Validate allocation requests
- **Flash Guard**: Verify read addresses

### Numerical Stability
- **Overflow Check**: Monitor int8 arithmetic
- **Underflow Check**: Handle near-zero values
- **NaN Detection**: Validate float32 outputs

### Real-Time Guarantees
- **Interrupt Masking**: Critical sections ≤10µs
- **Priority Inversion**: Use priority inheritance
- **Deadlock Prevention**: Resource ordering 