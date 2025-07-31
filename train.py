#!/usr/bin/env python3
"""
TinyRL Training CLI

Simple command-line interface for training TinyRL models.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from tinyrl.train import Trainer
    from tinyrl.utils import setup_logging
    from omegaconf import OmegaConf
except ImportError:
    print(
        "Error: Could not import TinyRL modules. Make sure you're in the project root."
    )
    sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Train TinyRL models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/ppo_cartpole.yaml",
        help="Path to training configuration file",
    )

    parser.add_argument("--env", type=str, help="Override environment name")

    parser.add_argument(
        "--algorithm", type=str, choices=["ppo", "a2c"], help="Override algorithm"
    )

    parser.add_argument("--timesteps", type=int, help="Override total timesteps")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--output-dir", type=str, help="Override output directory")

    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging"
    )

    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation (skip training)"
    )

    parser.add_argument("--model-path", type=str, help="Path to model for evaluation")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Available configs:")
        config_dir = Path("configs/train")
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                print(f"  - {config_file}")
        sys.exit(1)

    config = OmegaConf.load(args.config)

    # Override configuration with command line arguments
    if args.env:
        config.env.name = args.env

    if args.algorithm:
        config.algorithm.name = args.algorithm

    if args.timesteps:
        config.training.total_timesteps = args.timesteps

    if args.seed:
        config.seed = args.seed

    if args.output_dir:
        config.output.dir = args.output_dir

    if args.no_wandb:
        config.logging.wandb.enabled = False

    # Create output directory
    os.makedirs(config.output.dir, exist_ok=True)

    if args.eval_only:
        # Evaluation only mode
        if not args.model_path:
            print("Error: --model-path is required for evaluation-only mode")
            sys.exit(1)

        print(f"Evaluating model: {args.model_path}")

        # Import here to avoid circular imports
        from stable_baselines3 import PPO, A2C
        import gymnasium as gym

        # Load model
        if "ppo" in args.model_path.lower():
            model = PPO.load(args.model_path)
        else:
            model = A2C.load(args.model_path)

        # Create environment
        env = gym.make(config.env.name)

        # Evaluate
        rewards = []
        for episode in range(10):
            obs, _ = env.reset(seed=args.seed + episode)
            done = False
            episode_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                if truncated:
                    done = True

            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: {episode_reward:.2f}")

        mean_reward = sum(rewards) / len(rewards)
        print(f"\nMean reward: {mean_reward:.2f}")

    else:
        # Training mode
        print(f"Starting training with config: {args.config}")
        print(f"Environment: {config.env.name}")
        print(f"Algorithm: {config.algorithm.name}")
        print(f"Timesteps: {config.training.total_timesteps}")
        print(f"Seed: {config.seed}")
        print(f"Output: {config.output.dir}")

        # Create trainer and train
        trainer = Trainer(config)
        trainer.train()

        # Evaluate
        print("\nEvaluating trained model...")
        mean_reward, std_reward = trainer.evaluate(n_eval_episodes=10)
        print(f"Final evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")

        print(f"\nTraining completed! Model saved to: {config.output.dir}")


if __name__ == "__main__":
    main()
