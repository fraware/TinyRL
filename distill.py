#!/usr/bin/env python3
"""
Distillation CLI Script

Run stateless actor distillation pipeline with command-line interface.
"""

import argparse
import json
import os
import sys

import torch

from tinyrl.distillation import DistillationConfig, run_distillation_pipeline


def create_distillation_config(args) -> DistillationConfig:
    """Create distillation configuration from command line arguments."""
    return DistillationConfig(
        n_trajectories=args.n_trajectories,
        max_episode_steps=args.max_episode_steps,
        task_diversity_factor=args.task_diversity_factor,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        validation_split=args.validation_split,
        reward_delta_threshold=args.reward_delta_threshold,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run stateless actor distillation pipeline"
    )

    # Required arguments
    parser.add_argument("teacher_model_path", help="Path to trained teacher model")
    parser.add_argument(
        "env_name", help="Environment name (e.g., CartPole-v1, LunarLander-v2)"
    )

    # Dataset generation
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=1000,
        help="Number of trajectories to generate (default: 1000)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode (default: 1000)",
    )
    parser.add_argument(
        "--task-diversity-factor",
        type=float,
        default=10.0,
        help="Task diversity factor (default: 10.0)",
    )

    # Distillation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Distillation temperature (default: 2.0)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="KL divergence weight (default: 0.5)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Reward ranking hinge weight (default: 0.3)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.2, help="Policy fidelity weight (default: 0.2)"
    )

    # Training parameters
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size (default: 64)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )

    # Validation parameters
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--reward-delta-threshold",
        type=float,
        default=0.005,
        help="Reward delta threshold (default: 0.005 = 0.5%)",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/distillation",
        help="Output directory (default: ./outputs/distillation)",
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

    if args.alpha + args.beta + args.gamma > 1.0:
        print("Error: alpha + beta + gamma must be <= 1.0")
        sys.exit(1)

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create configuration
    config = create_distillation_config(args)

    if args.verbose:
        print("Distillation Configuration:")
        print(json.dumps(config.__dict__, indent=2))

    if args.dry_run:
        print("Dry run mode - configuration validated")
        return

    # Run distillation pipeline
    try:
        results = run_distillation_pipeline(
            config=config,
            teacher_model_path=args.teacher_model_path,
            env_name=args.env_name,
            output_dir=args.output_dir,
        )

        # Print summary
        print("\n" + "=" * 50)
        print("DISTILLATION COMPLETED")
        print("=" * 50)

        report = results["evaluation_report"]
        print(f"Teacher Reward: {report['teacher_reward']:.2f}")
        print(f"Student Reward: {report['student_reward']:.2f}")
        print(f"Reward Delta: {report['reward_delta_percentage']:.2f}%")
        print(f"Threshold: {report['threshold_percentage']:.2f}%")
        print(f"Status: {report['status']}")

        print(f"\nResults saved to: {args.output_dir}")

        if report["passed"]:
            print("✅ Distillation PASSED - reward delta within threshold")
        else:
            print("❌ Distillation FAILED - reward delta exceeds threshold")
            sys.exit(1)

    except Exception as e:
        print(f"Error during distillation: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
