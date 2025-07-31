#!/usr/bin/env python3
"""
Quantization CLI Script

Run differentiable quantization pipeline with command-line interface.
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

from tinyrl.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    run_quantization_pipeline,
)
from tinyrl.distillation import TrajectoryDataset


def create_quantization_config(args) -> QuantizationConfig:
    """Create quantization configuration from command line arguments."""
    return QuantizationConfig(
        bits=args.bits,
        scheme=QuantizationScheme(args.scheme),
        symmetric=args.symmetric,
        per_channel=args.per_channel,
        target_mcu=args.target_mcu,
        max_flash_size=args.max_flash_size,
        max_ram_size=args.max_ram_size,
        target_latency_ms=args.target_latency_ms,
        policy_fidelity_weight=args.policy_fidelity_weight,
        hardware_cost_weight=args.hardware_cost_weight,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        reward_delta_threshold=args.reward_delta_threshold,
        flash_size_threshold=args.flash_size_threshold,
        ram_size_threshold=args.ram_size_threshold,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run differentiable quantization pipeline"
    )

    # Required arguments
    parser.add_argument("teacher_model_path", help="Path to teacher model")
    parser.add_argument("student_model_path", help="Path to student model")
    parser.add_argument("dataset_path", help="Path to trajectory dataset")

    # Quantization parameters
    parser.add_argument(
        "--bits", type=int, default=8, help="Quantization bits (default: 8)"
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="symmetric",
        choices=["symmetric", "asymmetric", "power_of_2"],
        help="Quantization scheme (default: symmetric)",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        default=True,
        help="Use symmetric quantization (default: True)",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        default=True,
        help="Use per-channel quantization (default: True)",
    )

    # Hardware cost model
    parser.add_argument(
        "--target-mcu",
        type=str,
        default="cortex-m55",
        choices=["cortex-m55", "cortex-m4"],
        help="Target MCU (default: cortex-m55)",
    )
    parser.add_argument(
        "--max-flash-size",
        type=int,
        default=32768,
        help="Maximum flash size in bytes (default: 32768 = 32KB)",
    )
    parser.add_argument(
        "--max-ram-size",
        type=int,
        default=4096,
        help="Maximum RAM size in bytes (default: 4096 = 4KB)",
    )
    parser.add_argument(
        "--target-latency-ms",
        type=float,
        default=5.0,
        help="Target latency in milliseconds (default: 5.0)",
    )

    # Joint loss weights
    parser.add_argument(
        "--policy-fidelity-weight",
        type=float,
        default=0.7,
        help="Policy fidelity loss weight (default: 0.7)",
    )
    parser.add_argument(
        "--hardware-cost-weight",
        type=float,
        default=0.3,
        help="Hardware cost loss weight (default: 0.3)",
    )

    # Training parameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size (default: 32)"
    )

    # Validation parameters
    parser.add_argument(
        "--reward-delta-threshold",
        type=float,
        default=0.01,
        help="Reward delta threshold (default: 0.01 = 1%)",
    )
    parser.add_argument(
        "--flash-size-threshold",
        type=int,
        default=32768,
        help="Flash size threshold in bytes (default: 32768 = 32KB)",
    )
    parser.add_argument(
        "--ram-size-threshold",
        type=int,
        default=4096,
        help="RAM size threshold in bytes (default: 4096 = 4KB)",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/quantization",
        help="Output directory (default: ./outputs/quantization)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # Other options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print configuration without running"
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.teacher_model_path):
        print(f"Error: Teacher model path does not exist: {args.teacher_model_path}")
        sys.exit(1)

    if not os.path.exists(args.student_model_path):
        print(f"Error: Student model path does not exist: {args.student_model_path}")
        sys.exit(1)

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)

    if args.policy_fidelity_weight + args.hardware_cost_weight != 1.0:
        print("Error: policy_fidelity_weight + hardware_cost_weight must equal 1.0")
        sys.exit(1)

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create configuration
    config = create_quantization_config(args)

    if args.verbose:
        print("Quantization Configuration:")
        print(json.dumps(config.__dict__, indent=2))

    if args.dry_run:
        print("Dry run mode - configuration validated")
        return

    # Load dataset
    try:
        print("Loading dataset...")
        from tinyrl.distillation import OfflineDatasetAggregator

        aggregator = OfflineDatasetAggregator(config)
        trajectories = aggregator.load_dataset(args.dataset_path)
        dataset = TrajectoryDataset(trajectories)

        # Create data loaders
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
        )

        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )

        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Run quantization pipeline
    try:
        results = run_quantization_pipeline(
            config=config,
            teacher_model_path=args.teacher_model_path,
            student_model_path=args.student_model_path,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=args.output_dir,
        )

        # Print summary
        print("\n" + "=" * 50)
        print("QUANTIZATION COMPLETED")
        print("=" * 50)

        report = results["quantization_report"]
        hw_costs = results["quantization_results"]["hardware_costs"]

        print(f"Flash Size: {hw_costs['flash_size']} bytes")
        print(f"RAM Size: {hw_costs['ram_size']} bytes")
        print(f"Latency: {hw_costs['latency_ms']:.2f} ms")
        print(f"Reward Delta: {report['reward_delta_percentage']:.2f}%")
        print(f"Status: {report['status']}")

        print(f"\nResults saved to: {args.output_dir}")

        if report["status"] == "PASS":
            print("✅ Quantization PASSED - all constraints met")
        else:
            print("❌ Quantization FAILED - constraints violated")
            sys.exit(1)

    except Exception as e:
        print(f"Error during quantization: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
