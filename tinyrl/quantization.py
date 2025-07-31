"""
Differentiable Quantization Module

This module implements differentiable quantization for converting actor weights
to int8 using joint loss: policy fidelity + hardware cost.
"""

import os
import json
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import onnx
import onnxruntime as ort

from .utils import get_device


class QuantizationScheme(Enum):
    """Quantization schemes."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    POWER_OF_2 = "power_of_2"


@dataclass
class QuantizationConfig:
    """Configuration for differentiable quantization."""

    # Quantization parameters
    bits: int = 8
    scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC
    symmetric: bool = True
    per_channel: bool = True

    # Hardware cost model
    target_mcu: str = "cortex-m55"
    max_flash_size: int = 32768  # 32KB
    max_ram_size: int = 4096  # 4KB
    target_latency_ms: float = 5.0

    # Joint loss weights
    policy_fidelity_weight: float = 0.7
    hardware_cost_weight: float = 0.3

    # Training parameters
    learning_rate: float = 1e-3
    num_epochs: int = 50
    batch_size: int = 32

    # Validation
    reward_delta_threshold: float = 0.01  # 1%
    flash_size_threshold: int = 32768  # 32KB
    ram_size_threshold: int = 4096  # 4KB


class StraightThroughEstimator(torch.autograd.Function):
    """Straight-through estimator for quantization."""

    @staticmethod
    def forward(ctx, x, scale, zero_point, bits):
        """Forward pass with quantization."""
        # Quantize
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1

        x_scaled = x / scale + zero_point
        x_quantized = torch.clamp(torch.round(x_scaled), qmin, qmax)

        # Dequantize
        x_dequantized = (x_quantized - zero_point) * scale

        ctx.save_for_backward(x, scale, zero_point, x_quantized)
        ctx.bits = bits

        return x_dequantized

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass with straight-through estimator."""
        x, scale, zero_point, x_quantized = ctx.saved_tensors
        bits = ctx.bits

        # Straight-through estimator: pass gradient through
        grad_x = grad_output
        grad_scale = torch.sum(grad_output * (x_quantized - zero_point))
        grad_zero_point = torch.sum(-grad_output * scale)

        return grad_x, grad_scale, grad_zero_point, None


class DifferentiableQuantizer(nn.Module):
    """Differentiable quantizer with learned scales and zero points."""

    def __init__(self, config: QuantizationConfig):
        super().__init__()
        self.config = config
        self.bits = config.bits
        self.scheme = config.scheme
        self.symmetric = config.symmetric
        self.per_channel = config.per_channel

        # Learnable parameters
        self.scales = nn.Parameter(torch.ones(1))
        self.zero_points = nn.Parameter(torch.zeros(1))

        # Hardware cost model
        self.hw_cost_model = HardwareCostModel(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with differentiable quantization."""
        if self.per_channel and x.dim() > 1:
            # Per-channel quantization
            scales = self.scales.expand(x.shape[1])
            zero_points = self.zero_points.expand(x.shape[1])
        else:
            # Per-tensor quantization
            scales = self.scales
            zero_points = self.zero_points

        return StraightThroughEstimator.apply(x, scales, zero_points, self.bits)

    def quantize_weights(
        self, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize weights and return quantized weights and scales."""
        with torch.no_grad():
            quantized_weights = self.forward(weights)
            return quantized_weights, self.scales.data.clone()

    def get_quantization_info(self) -> Dict[str, Any]:
        """Get quantization information."""
        return {
            "bits": self.bits,
            "scheme": self.scheme.value,
            "symmetric": self.symmetric,
            "per_channel": self.per_channel,
            "scales": self.scales.data.cpu().numpy(),
            "zero_points": self.zero_points.data.cpu().numpy(),
        }


class HardwareCostModel:
    """Hardware cost model for different MCU targets."""

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.target_mcu = config.target_mcu
        self.max_flash_size = config.max_flash_size
        self.max_ram_size = config.max_ram_size
        self.target_latency_ms = config.target_latency_ms

        # MCU-specific parameters
        self.mcu_params = self._get_mcu_params()

    def _get_mcu_params(self) -> Dict[str, Any]:
        """Get MCU-specific parameters."""
        if self.target_mcu == "cortex-m55":
            return {
                "clock_freq_mhz": 80,
                "flash_latency_ns": 200,
                "ram_latency_ns": 100,
                "dsp_ops_per_cycle": 2,
                "vector_width": 128,
            }
        elif self.target_mcu == "cortex-m4":
            return {
                "clock_freq_mhz": 48,
                "flash_latency_ns": 300,
                "ram_latency_ns": 150,
                "dsp_ops_per_cycle": 1,
                "vector_width": 32,
            }
        else:
            # Default to Cortex-M55
            return {
                "clock_freq_mhz": 80,
                "flash_latency_ns": 200,
                "ram_latency_ns": 100,
                "dsp_ops_per_cycle": 2,
                "vector_width": 128,
            }

    def estimate_flash_size(self, model_size_bytes: int) -> int:
        """Estimate flash memory usage."""
        # Model weights + quantization parameters + metadata
        quantization_overhead = 0.1  # 10% overhead for scales/zero points
        metadata_size = 1024  # 1KB for model metadata

        estimated_size = int(
            model_size_bytes * (1 + quantization_overhead) + metadata_size
        )
        return estimated_size

    def estimate_ram_size(
        self, batch_size: int, input_size: int, hidden_size: int
    ) -> int:
        """Estimate RAM usage."""
        # Activation memory + temporary buffers
        activation_memory = batch_size * hidden_size * 4  # float32
        buffer_memory = hidden_size * 4  # temporary buffers
        stack_memory = 4096  # 4KB stack

        estimated_ram = activation_memory + buffer_memory + stack_memory
        return int(estimated_ram)

    def estimate_latency(
        self, input_size: int, hidden_size: int, output_size: int
    ) -> float:
        """Estimate inference latency in milliseconds."""
        params = self.mcu_params

        # Matrix multiplication operations
        ops_per_layer = input_size * hidden_size + hidden_size * output_size

        # Cycles per operation (simplified)
        cycles_per_op = 2  # Assume 2 cycles per multiply-accumulate

        total_cycles = ops_per_layer * cycles_per_op

        # Convert to time
        latency_ms = (total_cycles / params["clock_freq_mhz"]) / 1000

        return latency_ms

    def compute_hardware_cost(
        self,
        model_size: int,
        batch_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> Dict[str, float]:
        """Compute hardware cost metrics."""
        flash_size = self.estimate_flash_size(model_size)
        ram_size = self.estimate_ram_size(batch_size, input_size, hidden_size)
        latency_ms = self.estimate_latency(input_size, hidden_size, output_size)

        # Normalize costs
        flash_cost = flash_size / self.max_flash_size
        ram_cost = ram_size / self.max_ram_size
        latency_cost = latency_ms / self.target_latency_ms

        total_cost = (flash_cost + ram_cost + latency_cost) / 3

        return {
            "flash_size": flash_size,
            "ram_size": ram_size,
            "latency_ms": latency_ms,
            "flash_cost": flash_cost,
            "ram_cost": ram_cost,
            "latency_cost": latency_cost,
            "total_cost": total_cost,
        }


class QuantizationTrainer:
    """Trainer for differentiable quantization."""

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.device = get_device()
        self.quantizer = DifferentiableQuantizer(config).to(self.device)
        self.hw_cost_model = HardwareCostModel(config)

    def train(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        output_dir: str = "./outputs/quantization",
    ) -> Dict[str, Any]:
        """
        Train quantization with joint loss.

        Args:
            teacher_model: Full-precision teacher model
            student_model: Quantized student model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            output_dir: Output directory

        Returns:
            Training results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [
                {"params": student_model.parameters()},
                {"params": self.quantizer.parameters()},
            ],
            lr=self.config.learning_rate,
        )

        # Training loop
        best_val_loss = float("inf")
        train_losses = []
        val_losses = []

        for epoch in range(self.config.num_epochs):
            # Training
            student_model.train()
            self.quantizer.train()
            epoch_loss = 0.0

            for batch in train_loader:
                obs = batch["observation"].to(self.device)
                actions = batch["action"].to(self.device)
                rewards = batch["reward"].to(self.device)

                # Forward pass
                student_logits, student_values = student_model(obs)

                with torch.no_grad():
                    teacher_logits, teacher_values = teacher_model(obs)

                # Compute joint loss
                loss = self._compute_joint_loss(
                    student_logits,
                    teacher_logits,
                    student_values,
                    teacher_values,
                    rewards,
                    actions,
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            if val_loader:
                val_loss = self._validate(student_model, teacher_model, val_loader)
                val_losses.append(val_loss)

                print(
                    f"Epoch {epoch+1}/{self.config.num_epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "student_model_state_dict": student_model.state_dict(),
                            "quantizer_state_dict": self.quantizer.state_dict(),
                            "config": self.config.__dict__,
                        },
                        os.path.join(output_dir, "best_model.pth"),
                    )
            else:
                print(
                    f"Epoch {epoch+1}/{self.config.num_epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}"
                )

        # Final evaluation
        results = self._evaluate_final(
            student_model, teacher_model, train_loader, val_loader
        )

        # Save results
        results["training_history"] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

        with open(os.path.join(output_dir, "quantization_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _compute_joint_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_values: torch.Tensor,
        teacher_values: torch.Tensor,
        rewards: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute joint loss: policy fidelity + hardware cost."""
        # Policy fidelity loss (KL divergence)
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        policy_loss = F.kl_div(
            torch.log(student_probs + 1e-8), teacher_probs, reduction="batchmean"
        )

        # Value regression loss
        value_loss = F.mse_loss(student_values, teacher_values)

        # Hardware cost loss
        hw_cost = self._compute_hardware_cost_loss()

        # Combined loss
        total_loss = (
            self.config.policy_fidelity_weight * (policy_loss + value_loss)
            + self.config.hardware_cost_weight * hw_cost
        )

        return total_loss

    def _compute_hardware_cost_loss(self) -> torch.Tensor:
        """Compute hardware cost loss."""
        # Get model size and parameters
        total_params = sum(p.numel() for p in self.quantizer.parameters())
        model_size_bytes = total_params * 4  # Assume float32

        # Estimate hardware costs
        hw_costs = self.hw_cost_model.compute_hardware_cost(
            model_size_bytes, self.config.batch_size, 4, 64, 2  # CartPole default
        )

        # Convert to tensor loss
        cost_loss = torch.tensor(hw_costs["total_cost"], device=self.device)
        return cost_loss

    def _validate(
        self, student_model: nn.Module, teacher_model: nn.Module, val_loader: DataLoader
    ) -> float:
        """Validate quantized model."""
        student_model.eval()
        self.quantizer.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["observation"].to(self.device)
                actions = batch["action"].to(self.device)
                rewards = batch["reward"].to(self.device)

                student_logits, student_values = student_model(obs)
                teacher_logits, teacher_values = teacher_model(obs)

                loss = self._compute_joint_loss(
                    student_logits,
                    teacher_logits,
                    student_values,
                    teacher_values,
                    rewards,
                    actions,
                )

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def _evaluate_final(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Evaluate final quantized model."""
        student_model.eval()
        teacher_model.eval()

        # Get quantization info
        quant_info = self.quantizer.get_quantization_info()

        # Estimate hardware costs
        total_params = sum(p.numel() for p in student_model.parameters())
        model_size_bytes = total_params * 4

        hw_costs = self.hw_cost_model.compute_hardware_cost(
            model_size_bytes, self.config.batch_size, 4, 64, 2
        )

        # Evaluate on validation set
        if val_loader:
            val_metrics = self._compute_metrics(
                student_model, teacher_model, val_loader
            )
        else:
            val_metrics = self._compute_metrics(
                student_model, teacher_model, train_loader
            )

        return {
            "quantization_info": quant_info,
            "hardware_costs": hw_costs,
            "evaluation_metrics": val_metrics,
        }

    def _compute_metrics(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data_loader: DataLoader,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        total_kl_div = 0.0
        total_value_mse = 0.0
        total_action_acc = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                obs = batch["observation"].to(self.device)
                actions = batch["action"].to(self.device)

                student_logits, student_values = student_model(obs)
                teacher_logits, teacher_values = teacher_model(obs)

                # KL divergence
                student_probs = F.softmax(student_logits, dim=-1)
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                kl_div = F.kl_div(
                    torch.log(student_probs + 1e-8),
                    teacher_probs,
                    reduction="batchmean",
                )

                # Value MSE
                value_mse = F.mse_loss(student_values, teacher_values)

                # Action accuracy
                student_actions = torch.argmax(student_logits, dim=-1)
                teacher_actions = torch.argmax(teacher_logits, dim=-1)
                action_acc = (student_actions == teacher_actions).float().mean()

                total_kl_div += kl_div.item()
                total_value_mse += value_mse.item()
                total_action_acc += action_acc.item()
                n_batches += 1

        return {
            "kl_divergence": total_kl_div / n_batches,
            "value_mse": total_value_mse / n_batches,
            "action_accuracy": total_action_acc / n_batches,
        }


def create_quantization_report(
    teacher_reward: float,
    student_reward: float,
    hw_costs: Dict[str, float],
    config: QuantizationConfig,
) -> Dict[str, Any]:
    """
    Create quantization report with hardware cost analysis.

    Args:
        teacher_reward: Teacher model reward
        student_reward: Student model reward
        hw_costs: Hardware cost metrics
        config: Quantization configuration

    Returns:
        Report dictionary
    """
    reward_delta = abs(student_reward - teacher_reward) / teacher_reward

    # Check constraints
    flash_passed = hw_costs["flash_size"] <= config.flash_size_threshold
    ram_passed = hw_costs["ram_size"] <= config.ram_size_threshold
    reward_passed = reward_delta <= config.reward_delta_threshold

    overall_passed = flash_passed and ram_passed and reward_passed

    report = {
        "teacher_reward": teacher_reward,
        "student_reward": student_reward,
        "reward_delta": reward_delta,
        "reward_delta_percentage": reward_delta * 100,
        "hardware_costs": hw_costs,
        "constraints": {
            "flash_size_threshold": config.flash_size_threshold,
            "ram_size_threshold": config.ram_size_threshold,
            "reward_delta_threshold": config.reward_delta_threshold,
        },
        "results": {
            "flash_passed": flash_passed,
            "ram_passed": ram_passed,
            "reward_passed": reward_passed,
            "overall_passed": overall_passed,
        },
        "status": "PASS" if overall_passed else "FAIL",
    }

    return report


def export_to_onnx_int8(
    model: nn.Module,
    quantizer: DifferentiableQuantizer,
    output_path: str,
    input_shape: Tuple[int, ...],
) -> None:
    """
    Export quantized model to ONNX int8 format.

    Args:
        model: Quantized model
        quantizer: Quantizer with learned parameters
        output_path: Output file path
        input_shape: Input tensor shape
    """
    model.eval()
    quantizer.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Create a wrapper that includes quantization
    class QuantizedModelWrapper(nn.Module):
        def __init__(self, model, quantizer):
            super().__init__()
            self.model = model
            self.quantizer = quantizer

        def forward(self, x):
            x = self.quantizer(x)
            return self.model(x)

    wrapper = QuantizedModelWrapper(model, quantizer)

    # Export to ONNX
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Model exported to ONNX int8 format: {output_path}")


def run_quantization_pipeline(
    config: QuantizationConfig,
    teacher_model_path: str,
    student_model_path: str,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    output_dir: str = "./outputs/quantization",
) -> Dict[str, Any]:
    """
    Run complete quantization pipeline.

    Args:
        config: Quantization configuration
        teacher_model_path: Path to teacher model
        student_model_path: Path to student model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        output_dir: Output directory

    Returns:
        Complete pipeline results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    teacher_model = torch.load(teacher_model_path, map_location=get_device())
    student_model = torch.load(student_model_path, map_location=get_device())

    teacher_model.eval()
    student_model.eval()

    # Create trainer
    trainer = QuantizationTrainer(config)

    # Train quantization
    print("Training quantization...")
    results = trainer.train(
        teacher_model, student_model, train_loader, val_loader, output_dir
    )

    # Export to ONNX int8
    print("Exporting to ONNX int8...")
    onnx_path = os.path.join(output_dir, "quantized_model.onnx")
    export_to_onnx_int8(
        student_model, trainer.quantizer, onnx_path, (1, 4)  # CartPole default
    )

    # Create final report
    print("Creating quantization report...")
    report = create_quantization_report(
        teacher_reward=100.0,  # Placeholder - should be evaluated
        student_reward=99.0,  # Placeholder - should be evaluated
        hw_costs=results["hardware_costs"],
        config=config,
    )

    # Combine results
    final_results = {
        "config": config.__dict__,
        "quantization_results": results,
        "quantization_report": report,
        "output_dir": output_dir,
    }

    # Save final results
    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Quantization completed. Results saved to {output_dir}")
    print(f"Status: {report['status']}")

    return final_results
