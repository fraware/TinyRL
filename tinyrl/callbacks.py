"""Callbacks for TinyRL training pipeline."""

import os
import time
import logging
from typing import Optional
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)


class WandbCallback(BaseCallback):
    """Callback for logging to Weights & Biases."""

    def __init__(self, verbose: int = 0, log_freq: int = 100, **kwargs):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.training_start = None

    def _on_training_start(self) -> None:
        """Called at the start of training."""
        self.training_start = time.time()
        if self.verbose > 0:
            logger.info("Starting training with WandbCallback")

    def _on_step(self) -> bool:
        """Called after each step."""
        if self.n_calls % self.log_freq == 0:
            # Log training metrics
            if hasattr(self.model, "logger") and self.model.logger is not None:
                for key, value in self.model.logger.name_to_value.items():
                    if isinstance(value, (int, float)):
                        wandb.log({f"train/{key}": value}, step=self.n_calls)

            # Log custom metrics
            if hasattr(self.model, "rollout_buffer"):
                buffer = self.model.rollout_buffer
                if hasattr(buffer, "observations"):
                    obs_mean = buffer.observations.mean()
                    obs_std = buffer.observations.std()
                    wandb.log(
                        {"train/obs_mean": obs_mean, "train/obs_std": obs_std},
                        step=self.n_calls,
                    )

        return True

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.training_start is not None:
            training_time = time.time() - self.training_start
            wandb.log({"train/total_time": training_time})
            if self.verbose > 0:
                logger.info(f"Training completed in {training_time:.2f}s")


class EvalCallback(BaseCallback):
    """Callback for evaluating the agent during training."""

    def __init__(
        self,
        eval_env: Optional[VecEnv] = None,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
        eval_freq: int = 1000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render
        self.best_mean_reward = -np.inf

        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_training_start(self) -> None:
        """Called at the start of training."""
        if self.eval_env is None:
            # Create evaluation environment
            from stable_baselines3.common.vec_env import DummyVecEnv

            env_name = self.training_env.get_attr("spec")[0].id
            self.eval_env = DummyVecEnv(
                [lambda: self.training_env.get_attr("envs")[0].unwrapped.__class__()]
            )

        if self.verbose > 0:
            logger.info("Starting evaluation callback")

    def _on_step(self) -> bool:
        """Called after each step."""
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the agent
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=self.render,
            )

            # Log results
            if self.verbose > 0:
                logger.info(f"Eval mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

            # Log to wandb if available
            if wandb.run is not None:
                wandb.log(
                    {
                        "eval/mean_reward": mean_reward,
                        "eval/std_reward": std_reward,
                        "eval/episodes": self.n_eval_episodes,
                    },
                    step=self.n_calls,
                )

            # Save best model
            if (
                mean_reward > self.best_mean_reward
                and self.best_model_save_path is not None
            ):
                self.best_mean_reward = mean_reward
                best_model_path = os.path.join(self.best_model_save_path, "best_model")
                self.model.save(best_model_path)
                if self.verbose > 0:
                    logger.info(f"New best model saved with reward: {mean_reward:.2f}")

        return True


class CheckpointCallback(BaseCallback):
    """Callback for saving model checkpoints."""

    def __init__(
        self,
        save_freq: int = 10000,
        save_path: Optional[str] = None,
        name_prefix: str = "model",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """Called after each step."""
        if self.n_calls % self.save_freq == 0:
            if self.save_path is not None:
                checkpoint_path = os.path.join(
                    self.save_path, f"{self.name_prefix}_{self.n_calls}_steps"
                )
                self.model.save(checkpoint_path)

                if self.verbose > 0:
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

                # Log checkpoint info to wandb
                if wandb.run is not None:
                    wandb.log(
                        {
                            "checkpoint/step": self.n_calls,
                            "checkpoint/path": checkpoint_path,
                        },
                        step=self.n_calls,
                    )

        return True


class KnowledgeDistillationCallback(BaseCallback):
    """Callback for knowledge distillation staging."""

    def __init__(
        self,
        teacher_model_path: str,
        distillation_start: int = 10000,
        distillation_freq: int = 1000,
        alpha: float = 0.5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.teacher_model_path = teacher_model_path
        self.distillation_start = distillation_start
        self.distillation_freq = distillation_freq
        self.alpha = alpha
        self.teacher_model = None
        self.distillation_active = False

    def _on_training_start(self) -> None:
        """Called at the start of training."""
        # Load teacher model
        if os.path.exists(self.teacher_model_path):
            self.teacher_model = self.model.__class__.load(self.teacher_model_path)
            if self.verbose > 0:
                logger.info(f"Loaded teacher model from {self.teacher_model_path}")
        else:
            logger.warning(f"Teacher model not found at {self.teacher_model_path}")

    def _on_step(self) -> bool:
        """Called after each step."""
        # Start distillation at specified step
        if self.n_calls >= self.distillation_start and not self.distillation_active:
            self.distillation_active = True
            if self.verbose > 0:
                logger.info("Starting knowledge distillation")

        # Log distillation metrics
        if self.distillation_active and self.n_calls % self.distillation_freq == 0:
            if wandb.run is not None:
                wandb.log(
                    {"distillation/active": 1, "distillation/alpha": self.alpha},
                    step=self.n_calls,
                )

        return True


class PruningHintCallback(BaseCallback):
    """Callback for providing pruning hints during training."""

    def __init__(
        self,
        pruning_start: int = 20000,
        pruning_freq: int = 5000,
        target_sparsity: float = 0.9,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.pruning_start = pruning_start
        self.pruning_freq = pruning_freq
        self.target_sparsity = target_sparsity
        self.pruning_active = False

    def _on_step(self) -> bool:
        """Called after each step."""
        # Start pruning hints at specified step
        if self.n_calls >= self.pruning_start and not self.pruning_active:
            self.pruning_active = True
            if self.verbose > 0:
                logger.info("Starting pruning hints")

        # Log pruning metrics
        if self.pruning_active and self.n_calls % self.pruning_freq == 0:
            # Calculate current model sparsity
            total_params = 0
            zero_params = 0

            for param in self.model.parameters():
                total_params += param.numel()
                zero_params += (param == 0).sum().item()

            current_sparsity = zero_params / total_params if total_params > 0 else 0

            if wandb.run is not None:
                wandb.log(
                    {
                        "pruning/active": 1,
                        "pruning/current_sparsity": current_sparsity,
                        "pruning/target_sparsity": self.target_sparsity,
                    },
                    step=self.n_calls,
                )

        return True
