"""Main training pipeline for TinyRL."""

import os
import time
import logging
from typing import Optional, Tuple, Any
from dataclasses import dataclass

import torch
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from .utils import set_deterministic_seed, get_device, validate_config
from .callbacks import (
    WandbCallback,
    EvalCallback,
    CheckpointCallback,
    KnowledgeDistillationCallback,
    PruningHintCallback,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    env_name: str
    algorithm: str
    total_timesteps: int
    seed: int = 42
    deterministic: bool = True
    device: Optional[str] = None
    output_dir: str = "./outputs"
    save_model: bool = True
    save_replay_buffer: bool = False
    save_vecnormalize: bool = True


class Trainer:
    """Main trainer class for TinyRL algorithms."""

    def __init__(self, config: DictConfig):
        """Initialize trainer with configuration.

        Args:
            config: Hydra configuration object
        """
        self.config = config
        self.device = get_device(config.get("device", None))

        # Set deterministic seed
        set_deterministic_seed(config.seed, config.get("deterministic", True))

        # Setup logging
        self._setup_logging()

        # Create environment
        self.env = self._create_env()

        # Create model
        self.model = self._create_model()

        # Setup callbacks
        self.callbacks = self._setup_callbacks()

        logger.info(f"Initialized {config.algorithm.upper()} trainer")
        logger.info(f"Device: {self.device}")
        logger.info(f"Environment: {config.env.name}")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})

        if log_config.get("wandb", {}).get("enabled", False):
            wandb_config = log_config["wandb"]
            wandb.init(
                project=wandb_config.get("project", "tinyrl"),
                entity=wandb_config.get("entity"),
                tags=wandb_config.get("tags", []),
                group=wandb_config.get("group"),
                config=OmegaConf.to_container(self.config, resolve=True),
            )
            logger.info("Initialized Weights & Biases logging")

    def _create_env(self) -> gym.Env:
        """Create training environment."""
        env_config = self.config.env

        # Create environment
        env = gym.make(
            env_config.name,
            max_episode_steps=env_config.get("max_episode_steps"),
            render_mode=env_config.get("render_mode"),
        )

        # Wrap in DummyVecEnv for compatibility
        env = DummyVecEnv([lambda: env])

        # Apply normalization if specified
        if self.config.get("normalize_observations", True):
            env = VecNormalize(
                env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0
            )

        logger.info(f"Created environment: {env_config.name}")
        return env

    def _create_model(self) -> Any:
        """Create the RL model."""
        algorithm = self.config.algorithm.name.lower()

        # Model parameters
        model_params = {
            "policy": "MlpPolicy",
            "env": self.env,
            "verbose": 1,
            "device": self.device,
            "tensorboard_log": self.config.get("output", {}).get("dir", "./logs"),
        }

        # Algorithm-specific parameters
        if algorithm == "ppo":
            model_params.update(
                {
                    "learning_rate": self.config.algorithm.learning_rate,
                    "n_steps": self.config.algorithm.n_steps,
                    "batch_size": self.config.algorithm.batch_size,
                    "n_epochs": self.config.algorithm.n_epochs,
                    "gamma": self.config.algorithm.gamma,
                    "gae_lambda": self.config.algorithm.gae_lambda,
                    "clip_range": self.config.algorithm.clip_range,
                    "clip_range_vf": self.config.algorithm.clip_range_vf,
                    "normalize_advantage": self.config.algorithm.normalize_advantage,
                    "ent_coef": self.config.algorithm.ent_coef,
                    "vf_coef": self.config.algorithm.vf_coef,
                    "max_grad_norm": self.config.algorithm.max_grad_norm,
                    "use_sde": self.config.algorithm.use_sde,
                    "sde_sample_freq": self.config.algorithm.sde_sample_freq,
                    "target_kl": self.config.algorithm.target_kl,
                }
            )
            model = PPO(**model_params)

        elif algorithm == "a2c":
            model_params.update(
                {
                    "learning_rate": self.config.algorithm.learning_rate,
                    "n_steps": self.config.algorithm.n_steps,
                    "gamma": self.config.algorithm.gamma,
                    "gae_lambda": self.config.algorithm.gae_lambda,
                    "ent_coef": self.config.algorithm.ent_coef,
                    "vf_coef": self.config.algorithm.vf_coef,
                    "max_grad_norm": self.config.algorithm.max_grad_norm,
                    "rms_prop_eps": self.config.algorithm.rms_prop_eps,
                    "use_rms_prop": self.config.algorithm.use_rms_prop,
                    "use_sde": self.config.algorithm.use_sde,
                    "sde_sample_freq": self.config.algorithm.sde_sample_freq,
                }
            )
            model = A2C(**model_params)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Apply PyTorch 2.3 compilation if enabled
        if self.config.get("compilation", {}).get("enabled", False):
            compilation_config = self.config.compilation
            model.policy = torch.compile(
                model.policy,
                mode=compilation_config.get("mode", "reduce-overhead"),
                fullgraph=compilation_config.get("fullgraph", True),
                dynamic=compilation_config.get("dynamic", False),
            )
            logger.info("Applied PyTorch 2.3 compilation")

        return model

    def _setup_callbacks(self) -> CallbackList:
        """Setup training callbacks."""
        callbacks = []

        # Get callback configurations
        callback_configs = self.config.get("callbacks", [])

        for callback_config in callback_configs:
            callback_type = callback_config.get("_target_", "")

            if "WandbCallback" in callback_type:
                callbacks.append(WandbCallback(**callback_config))

            elif "EvalCallback" in callback_type:
                # Set paths if not provided
                if callback_config.get("best_model_save_path") is None:
                    output_dir = self.config.get("output", {}).get("dir", "./outputs")
                    callback_config["best_model_save_path"] = os.path.join(
                        output_dir, "best_model"
                    )
                if callback_config.get("log_path") is None:
                    output_dir = self.config.get("output", {}).get("dir", "./outputs")
                    callback_config["log_path"] = os.path.join(output_dir, "eval_logs")

                callbacks.append(EvalCallback(**callback_config))

            elif "CheckpointCallback" in callback_type:
                # Set save path if not provided
                if callback_config.get("save_path") is None:
                    output_dir = self.config.get("output", {}).get("dir", "./outputs")
                    callback_config["save_path"] = os.path.join(
                        output_dir, "checkpoints"
                    )

                callbacks.append(CheckpointCallback(**callback_config))

            elif "KnowledgeDistillationCallback" in callback_type:
                callbacks.append(KnowledgeDistillationCallback(**callback_config))

            elif "PruningHintCallback" in callback_type:
                callbacks.append(PruningHintCallback(**callback_config))

        return CallbackList(callbacks)

    def train(self) -> None:
        """Train the model."""
        logger.info("Starting training...")
        start_time = time.time()

        try:
            self.model.learn(
                total_timesteps=self.config.training.total_timesteps,
                callback=self.callbacks,
                progress_bar=True,
            )

            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f}s")

            # Save final model
            if self.config.get("output", {}).get("save_model", True):
                self._save_model()

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _save_model(self) -> None:
        """Save the trained model."""
        output_config = self.config.get("output", {})
        output_dir = output_config.get("dir", "./outputs")

        os.makedirs(output_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(output_dir, "final_model")
        self.model.save(model_path)
        logger.info(f"Saved model to {model_path}")

        # Save VecNormalize if used
        if isinstance(self.env, VecNormalize):
            vec_normalize_path = os.path.join(output_dir, "vec_normalize.pkl")
            self.env.save(vec_normalize_path)
            logger.info(f"Saved VecNormalize to {vec_normalize_path}")

        # Save configuration
        config_path = os.path.join(output_dir, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(self.config, f)
        logger.info(f"Saved config to {config_path}")

    def evaluate(self, n_eval_episodes: int = 10) -> Tuple[float, float]:
        """Evaluate the trained model.

        Args:
            n_eval_episodes: Number of episodes to evaluate

        Returns:
            Tuple of (mean_reward, std_reward)
        """
        logger.info("Evaluating model...")

        # Create evaluation environment
        eval_env = gym.make(self.config.env.name)
        eval_env = DummyVecEnv([lambda: eval_env])

        # Apply normalization if used during training
        if isinstance(self.env, VecNormalize):
            eval_env = VecNormalize(
                eval_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
            )
            # Copy normalization stats
            eval_env.obs_rms = self.env.obs_rms
            eval_env.ret_rms = self.env.ret_rms

        # Evaluate
        mean_reward, std_reward = self.model.evaluate_policy(
            eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
        )

        logger.info(f"Evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Log to wandb
        if wandb.run is not None:
            wandb.log(
                {
                    "final_eval/mean_reward": mean_reward,
                    "final_eval/std_reward": std_reward,
                    "final_eval/episodes": n_eval_episodes,
                }
            )

        return mean_reward, std_reward


@hydra.main(
    version_base=None, config_path="../configs/train", config_name="ppo_cartpole"
)
def main(config: DictConfig) -> None:
    """Main training function."""
    # Validate configuration
    validate_config(OmegaConf.to_container(config, resolve=True))

    # Create trainer
    trainer = Trainer(config)

    # Train
    trainer.train()

    # Evaluate
    trainer.evaluate(n_eval_episodes=10)

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
