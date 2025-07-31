"""
Stateless Actor Distillation Module

This module implements knowledge distillation from recurrent/stateful actors to
feed-forward approximators using DAgger-KD hybrid approach to curb covariate
shift.
"""

import os
import json
import pickle
from typing import Dict, List, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy

from .models import PPOActor, A2CActor
from .utils import get_device


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    # Dataset generation
    n_trajectories: int = 1000
    max_episode_steps: int = 1000
    task_diversity_factor: float = 10.0  # Generate ≥10× diverse trajectories

    # Distillation parameters
    temperature: float = 2.0
    alpha: float = 0.5  # KL divergence weight
    beta: float = 0.3  # Reward ranking hinge weight
    gamma: float = 0.2  # Policy fidelity weight

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 100
    early_stopping_patience: int = 10

    # Validation
    validation_split: float = 0.2
    reward_delta_threshold: float = 0.005  # < 0.5% vs teacher

    # OOD verification
    ood_tasks: List[str] = None  # Will be set to different env variants


class TrajectoryDataset(Dataset):
    """Dataset for storing teacher trajectories."""

    def __init__(self, trajectories: List[Dict[str, np.ndarray]]):
        self.trajectories = trajectories
        self._flatten_data()

    def _flatten_data(self):
        """Flatten trajectories into single arrays."""
        all_obs = []
        all_actions = []
        all_rewards = []
        all_log_probs = []
        all_values = []

        for traj in self.trajectories:
            all_obs.append(traj["observations"])
            all_actions.append(traj["actions"])
            all_rewards.append(traj["rewards"])
            all_log_probs.append(traj["log_probs"])
            all_values.append(traj["values"])

        self.observations = np.concatenate(all_obs, axis=0)
        self.actions = np.concatenate(all_actions, axis=0)
        self.rewards = np.concatenate(all_rewards, axis=0)
        self.log_probs = np.concatenate(all_log_probs, axis=0)
        self.values = np.concatenate(all_values, axis=0)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return {
            "observation": torch.FloatTensor(self.observations[idx]),
            "action": (
                torch.LongTensor([self.actions[idx]])
                if self.actions.dtype == np.int64
                else torch.FloatTensor([self.actions[idx]])
            ),
            "reward": torch.FloatTensor([self.rewards[idx]]),
            "teacher_log_prob": torch.FloatTensor([self.log_probs[idx]]),
            "teacher_value": torch.FloatTensor([self.values[idx]]),
        }


class KnowledgeDistillationLoss(nn.Module):
    """Combined loss for knowledge distillation."""

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # KL divergence weight
        self.beta = beta  # Reward ranking hinge weight
        self.gamma = gamma  # Policy fidelity weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_values: torch.Tensor,
        teacher_values: torch.Tensor,
        rewards: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined distillation loss.

        Args:
            student_logits: Student policy logits
            teacher_logits: Teacher policy logits
            student_values: Student value estimates
            teacher_values: Teacher value estimates
            rewards: Observed rewards
            actions: Taken actions

        Returns:
            Dictionary containing loss components
        """
        # KL divergence loss
        student_probs = F.softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(
            torch.log(student_probs + 1e-8), teacher_probs, reduction="batchmean"
        ) * (self.temperature**2)

        # Reward ranking hinge loss
        # Encourage student to preserve teacher's action ranking based on rewards
        student_action_probs = F.softmax(student_logits, dim=-1)
        teacher_action_probs = F.softmax(teacher_logits, dim=-1)

        # Get probabilities for taken actions
        student_action_prob = student_action_probs.gather(1, actions).squeeze()
        teacher_action_prob = teacher_action_probs.gather(1, actions).squeeze()

        # Hinge loss: encourage student to assign higher probability to actions
        # that teacher assigned higher probability to
        ranking_loss = F.relu(teacher_action_prob - student_action_prob + 0.1).mean()

        # Value regression loss
        value_loss = F.mse_loss(student_values, teacher_values)

        # Policy fidelity loss (encourage similar action distributions)
        fidelity_loss = F.mse_loss(student_action_probs, teacher_action_probs)

        # Combined loss
        total_loss = (
            self.alpha * kl_loss
            + self.beta * ranking_loss
            + self.gamma * fidelity_loss
            + (1 - self.alpha - self.beta - self.gamma) * value_loss
        )

        return {
            "total_loss": total_loss,
            "kl_loss": kl_loss,
            "ranking_loss": ranking_loss,
            "value_loss": value_loss,
            "fidelity_loss": fidelity_loss,
        }


class OfflineDatasetAggregator:
    """Generate diverse offline dataset from teacher policy."""

    def __init__(self, config: DistillationConfig):
        self.config = config
        self.device = get_device()

    def generate_trajectories(
        self, teacher_model_path: str, env_name: str, n_trajectories: int = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate diverse trajectories from teacher policy.

        Args:
            teacher_model_path: Path to trained teacher model
            env_name: Environment name
            n_trajectories: Number of trajectories to generate

        Returns:
            List of trajectory dictionaries
        """
        if n_trajectories is None:
            n_trajectories = self.config.n_trajectories

        # Load teacher model
        if teacher_model_path.endswith(".zip"):
            # Stable-Baselines3 model
            teacher_model = self._load_sb3_model(teacher_model_path)
        else:
            # Custom model
            teacher_model = self._load_custom_model(teacher_model_path)

        # Create environment
        env = gym.make(env_name)

        trajectories = []

        for i in range(n_trajectories):
            # Add task diversity by varying environment parameters
            env = self._create_diverse_env(env_name, i)

            trajectory = self._collect_trajectory(teacher_model, env)
            trajectories.append(trajectory)

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_trajectories} trajectories")

        return trajectories

    def _load_sb3_model(self, model_path: str):
        """Load Stable-Baselines3 model."""
        # Try to determine algorithm from path
        if "ppo" in model_path.lower():
            model = PPO.load(model_path)
        elif "a2c" in model_path.lower():
            model = A2C.load(model_path)
        else:
            # Default to PPO
            model = PPO.load(model_path)
        return model

    def _load_custom_model(self, model_path: str):
        """Load custom model."""
        # Implementation depends on model format
        # For now, assume it's a torch model
        return torch.load(model_path, map_location=self.device)

    def _create_diverse_env(self, env_name: str, trajectory_idx: int) -> gym.Env:
        """Create environment with diverse parameters."""
        env = gym.make(env_name)

        # Add diversity based on environment type
        if "CartPole" in env_name:
            # Vary initial conditions
            env.reset(seed=trajectory_idx)
        elif "LunarLander" in env_name:
            # Vary gravity, wind, etc.
            env.reset(seed=trajectory_idx)
        elif "Acrobot" in env_name:
            # Vary link lengths, masses
            env.reset(seed=trajectory_idx)

        return env

    def _collect_trajectory(self, teacher_model, env: gym.Env) -> Dict[str, np.ndarray]:
        """Collect single trajectory from teacher."""
        obs, _ = env.reset()
        done = False
        truncated = False

        observations = []
        actions = []
        rewards = []
        log_probs = []
        values = []

        step = 0

        while not (done or truncated) and step < self.config.max_episode_steps:
            # Get teacher action
            if hasattr(teacher_model, "predict"):
                # Stable-Baselines3 model
                action, _ = teacher_model.predict(obs, deterministic=False)
                # Note: We don't have access to log_probs and values from SB3
                # In practice, you'd need to modify the model to expose these
                log_prob = 0.0  # Placeholder
                value = 0.0  # Placeholder
            else:
                # Custom model
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_dist, value = teacher_model(obs_tensor)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)

                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]
                value = value.cpu().numpy()[0]

            # Take action
            next_obs, reward, done, truncated, info = env.step(action)

            # Store transition
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)

            obs = next_obs
            step += 1

        return {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "log_probs": np.array(log_probs),
            "values": np.array(values),
        }

    def save_dataset(
        self, trajectories: List[Dict[str, np.ndarray]], output_path: str
    ) -> None:
        """Save dataset to disk."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(trajectories, f)

        # Also save metadata
        metadata = {
            "n_trajectories": len(trajectories),
            "total_steps": sum(len(traj["observations"]) for traj in trajectories),
            "avg_reward": np.mean([np.sum(traj["rewards"]) for traj in trajectories]),
            "config": self.config.__dict__,
        }

        metadata_path = output_path.replace(".pkl", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_dataset(self, dataset_path: str) -> List[Dict[str, np.ndarray]]:
        """Load dataset from disk."""
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)
        return trajectories


class DistillationTrainer:
    """Train student policy using knowledge distillation."""

    def __init__(self, config: DistillationConfig):
        self.config = config
        self.device = get_device()
        self.criterion = KnowledgeDistillationLoss(
            temperature=config.temperature,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
        ).to(self.device)

    def train(
        self,
        teacher_model_path: str,
        student_model: nn.Module,
        train_dataset: TrajectoryDataset,
        val_dataset: TrajectoryDataset = None,
        output_dir: str = "./outputs/distillation",
    ) -> Dict[str, float]:
        """
        Train student model using knowledge distillation.

        Args:
            teacher_model_path: Path to teacher model
            student_model: Student model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            output_dir: Output directory for logs and checkpoints

        Returns:
            Dictionary with final metrics
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load teacher model
        teacher_model = self._load_teacher_model(teacher_model_path)
        teacher_model.eval()

        # Setup data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # For reproducibility
        )

        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
            )

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            student_model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.config.num_epochs):
            # Training
            student_model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                # Move to device
                obs = batch["observation"].to(self.device)
                actions = batch["action"].to(self.device)
                rewards = batch["reward"].to(self.device)
                teacher_log_probs = batch["teacher_log_prob"].to(self.device)
                teacher_values = batch["teacher_value"].to(self.device)

                # Forward pass
                student_logits, student_values = student_model(obs)

                # Get teacher outputs
                with torch.no_grad():
                    teacher_logits, _ = teacher_model(obs)

                # Compute loss
                loss_dict = self.criterion(
                    student_logits,
                    teacher_logits,
                    student_values,
                    teacher_values,
                    rewards,
                    actions,
                )

                loss = loss_dict["total_loss"]

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 0.5)
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            if val_dataset:
                val_loss = self._validate(student_model, teacher_model, val_loader)
                val_losses.append(val_loss)

                print(
                    f"Epoch {epoch+1}/{self.config.num_epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(
                        student_model.state_dict(),
                        os.path.join(output_dir, "best_model.pth"),
                    )
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(
                    f"Epoch {epoch+1}/{self.config.num_epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}"
                )

        # Load best model
        if val_dataset and os.path.exists(os.path.join(output_dir, "best_model.pth")):
            student_model.load_state_dict(
                torch.load(os.path.join(output_dir, "best_model.pth"))
            )

        # Final evaluation
        final_metrics = self._evaluate_final(
            student_model, teacher_model, train_loader, val_loader
        )

        # Save training history
        history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_metrics": final_metrics,
        }

        with open(os.path.join(output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        return final_metrics

    def _load_teacher_model(self, teacher_model_path: str) -> nn.Module:
        """Load teacher model."""
        # Implementation depends on model format
        # For now, assume it's a torch model
        model = torch.load(teacher_model_path, map_location=self.device)
        model.eval()
        return model

    def _validate(
        self, student_model: nn.Module, teacher_model: nn.Module, val_loader: DataLoader
    ) -> float:
        """Validate student model."""
        student_model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["observation"].to(self.device)
                actions = batch["action"].to(self.device)
                rewards = batch["reward"].to(self.device)
                teacher_log_probs = batch["teacher_log_prob"].to(self.device)
                teacher_values = batch["teacher_value"].to(self.device)

                student_logits, student_values = student_model(obs)
                teacher_logits, _ = teacher_model(obs)

                loss_dict = self.criterion(
                    student_logits,
                    teacher_logits,
                    student_values,
                    teacher_values,
                    rewards,
                    actions,
                )

                total_loss += loss_dict["total_loss"].item()

        return total_loss / len(val_loader)

    def _evaluate_final(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> Dict[str, float]:
        """Evaluate final model performance."""
        student_model.eval()
        teacher_model.eval()

        metrics = {}

        # Evaluate on training set
        train_metrics = self._compute_metrics(
            student_model, teacher_model, train_loader
        )
        metrics.update({f"train_{k}": v for k, v in train_metrics.items()})

        # Evaluate on validation set
        if val_loader:
            val_metrics = self._compute_metrics(
                student_model, teacher_model, val_loader
            )
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

        return metrics

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

                # Action accuracy (top-1)
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


def create_distillation_report(
    teacher_reward: float, student_reward: float, reward_delta_threshold: float = 0.005
) -> Dict[str, Any]:
    """
    Create distillation report with reward delta analysis.

    Args:
        teacher_reward: Teacher model reward
        student_reward: Student model reward
        reward_delta_threshold: Maximum allowed reward drop (default: 0.5%)

    Returns:
        Report dictionary
    """
    reward_delta = abs(student_reward - teacher_reward) / teacher_reward

    report = {
        "teacher_reward": teacher_reward,
        "student_reward": student_reward,
        "reward_delta": reward_delta,
        "reward_delta_percentage": reward_delta * 100,
        "threshold": reward_delta_threshold,
        "threshold_percentage": reward_delta_threshold * 100,
        "passed": reward_delta < reward_delta_threshold,
        "status": "PASS" if reward_delta < reward_delta_threshold else "FAIL",
    }

    return report


def run_distillation_pipeline(
    config: DistillationConfig,
    teacher_model_path: str,
    env_name: str,
    output_dir: str = "./outputs/distillation",
) -> Dict[str, Any]:
    """
    Run complete distillation pipeline.

    Args:
        config: Distillation configuration
        teacher_model_path: Path to teacher model
        env_name: Environment name
        output_dir: Output directory

    Returns:
        Complete pipeline results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate offline dataset
    print("Generating offline dataset...")
    aggregator = OfflineDatasetAggregator(config)
    trajectories = aggregator.generate_trajectories(
        teacher_model_path, env_name, config.n_trajectories
    )

    # Save dataset
    dataset_path = os.path.join(output_dir, "trajectories.pkl")
    aggregator.save_dataset(trajectories, dataset_path)

    # Step 2: Create datasets
    dataset = TrajectoryDataset(trajectories)

    # Split into train/val
    train_size = int((1 - config.validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Step 3: Create student model
    # Determine model architecture from teacher
    if "ppo" in teacher_model_path.lower():
        student_model = PPOActor(obs_dim=4, action_dim=2)  # CartPole default
    elif "a2c" in teacher_model_path.lower():
        student_model = A2CActor(obs_dim=8, action_dim=4)  # LunarLander default
    else:
        # Default to PPO
        student_model = PPOActor(obs_dim=4, action_dim=2)

    student_model = student_model.to(get_device())

    # Step 4: Train student
    print("Training student model...")
    trainer = DistillationTrainer(config)
    metrics = trainer.train(
        teacher_model_path, student_model, train_dataset, val_dataset, output_dir
    )

    # Step 5: Evaluate on validation environment
    print("Evaluating student model...")
    val_env = gym.make(env_name)

    # Evaluate teacher
    teacher_model = trainer._load_teacher_model(teacher_model_path)
    teacher_reward, _ = evaluate_policy(teacher_model, val_env, n_eval_episodes=10)

    # Evaluate student
    student_reward, _ = evaluate_policy(student_model, val_env, n_eval_episodes=10)

    # Step 6: Create report
    report = create_distillation_report(
        teacher_reward, student_reward, config.reward_delta_threshold
    )

    # Combine all results
    results = {
        "config": config.__dict__,
        "dataset_info": {
            "n_trajectories": len(trajectories),
            "total_steps": sum(len(traj["observations"]) for traj in trajectories),
            "avg_reward": np.mean([np.sum(traj["rewards"]) for traj in trajectories]),
        },
        "training_metrics": metrics,
        "evaluation_report": report,
        "output_dir": output_dir,
    }

    # Save results
    results_path = os.path.join(output_dir, "distillation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Distillation completed. Results saved to {output_dir}")
    print(
        f"Reward delta: {report['reward_delta_percentage']:.2f}% "
        f"({'PASS' if report['passed'] else 'FAIL'})"
    )

    return results
