#!/usr/bin/env python3
"""
Critic Pruning CLI Script

Run critic pruning and LUT folding pipeline with command-line interface.
"""

import argparse
import json
import os
import sys

import torch

from tinyrl.pruning import PruningConfig, run_pruning_pipeline


def create_pruning_config(args) -> PruningConfig:
    """Create pruning configuration from command line arguments."""
    return PruningConfig(
        target_sparsity=args.target_sparsity,
        magnitude_threshold=args.magnitude_threshold,
        global_pruning=args.global_pruning,
        lut_bits=args.lut_bits,
        lut_size=args.lut_size,
        monotonic_guarantee=args.monotonic_guarantee,
        reward_delta_threshold=args.reward_delta_threshold,
        lut_size_threshold=args.lut_size_threshold,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run critic pruning and LUT folding pipeline"
    )

    # Required arguments
    parser.add_argument("teacher_model_path", help="Path to trained teacher model")
    parser.add_argument(
        "env_name", help="Environment name (e.g., CartPole-v1, LunarLander-v2)"
    )

    # Pruning parameters
    parser.add_argument(
        "--target-sparsity",
        type=float,
        default=0.9,
        help="Target sparsity for critic pruning (default: 0.9)",
    )
    parser.add_argument(
        "--magnitude-threshold",
        type=float,
        default=0.01,
        help="Minimum weight magnitude threshold (default: 0.01)",
    )
    parser.add_argument(
        "--global-pruning",
        action="store_true",
        default=True,
        help="Use global magnitude pruning (default: True)",
    )

    # LUT parameters
    parser.add_argument(
        "--lut-bits",
        type=int,
        default=8,
        help="LUT bit precision (default: 8)",
    )
    parser.add_argument(
        "--lut-size",
        type=int,
        default=256,
        help="LUT size in entries (default: 256)",
    )
    parser.add_argument(
        "--monotonic-guarantee",
        action="store_true",
        default=True,
        help="Ensure LUT monotonicity (default: True)",
    )

    # Validation parameters
    parser.add_argument(
        "--reward-delta-threshold",
        type=float,
        default=0.02,
        help="Reward delta threshold vs baseline (default: 0.02 = 2%)",
    )
    parser.add_argument(
        "--lut-size-threshold",
        type=int,
        default=1024,
        help="LUT size threshold in bytes (default: 1024 = 1KB)",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/pruning",
        help="Output directory (default: ./outputs/pruning)",
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

    if args.target_sparsity < 0.0 or args.target_sparsity > 1.0:
        print("Error: target_sparsity must be between 0.0 and 1.0")
        sys.exit(1)

    if args.lut_size <= 0 or args.lut_size > 65536:
        print("Error: lut_size must be between 1 and 65536")
        sys.exit(1)

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create configuration
    config = create_pruning_config(args)

    if args.verbose:
        print("Pruning Configuration:")
        print(json.dumps(config.__dict__, indent=2))

    if args.dry_run:
        print("Dry run mode - configuration validated")
        return

    # Create dummy data loaders (in practice, these would be real)
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np

    # Create dummy training data
    dummy_obs = torch.randn(1000, 4)  # CartPole observation dimension
    dummy_dataset = TensorDataset(dummy_obs)
    train_loader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dummy_dataset, batch_size=32, shuffle=False)

    # Run pruning pipeline
    try:
        results = run_pruning_pipeline(
            config=config,
            teacher_model_path=args.teacher_model_path,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=args.output_dir,
        )

        # Print summary
        print("\n" + "=" * 50)
        print("PRUNING COMPLETED")
        print("=" * 50)

        report = results["report"]
        pruning_stats = results["pruning_stats"]
        evaluation = results["evaluation"]

        print(f"Status: {report['status']}")
        print(f"Sparsity Achieved: {pruning_stats['sparsity']:.2%}")
        print(f"LUT Size: {evaluation['lut_size_bytes']} bytes")
        print(f"Reward Delta: {evaluation['reward_delta']:.2%}")
        print(f"Monotonic Preserved: {evaluation['monotonic_preserved']}")
        print(f"LUT Hash: {results['lut_hash']}")

        print(f"\nResults saved to: {args.output_dir}")

        if report["passed"]:
            print("✅ Pruning PASSED - all requirements met")
        else:
            print("❌ Pruning FAILED - some requirements not met")
            failed_reqs = [
                req for req, passed in report["requirements"].items() if not passed
            ]
            print(f"Failed requirements: {failed_reqs}")
            sys.exit(1)

    except Exception as e:
        print(f"Error during pruning: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
