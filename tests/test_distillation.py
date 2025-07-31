"""
Tests for stateless actor distillation module.

This module tests the distillation pipeline including dataset generation,
knowledge distillation loss, and reward delta reporting.
"""

import os
import tempfile
import json
import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from tinyrl.distillation import (
    DistillationConfig,
    TrajectoryDataset,
    KnowledgeDistillationLoss,
    OfflineDatasetAggregator,
    DistillationTrainer,
    create_distillation_report,
    run_distillation_pipeline,
)


class TestDistillationConfig:
    """Test distillation configuration."""

    def test_config_creation(self):
        """Test creating distillation configuration."""
        config = DistillationConfig(
            n_trajectories=500, temperature=3.0, alpha=0.6, beta=0.2, gamma=0.1
        )

        assert config.n_trajectories == 500
        assert config.temperature == 3.0
        assert config.alpha == 0.6
        assert config.beta == 0.2
        assert config.gamma == 0.1
        assert config.alpha + config.beta + config.gamma <= 1.0

    def test_config_defaults(self):
        """Test configuration defaults."""
        config = DistillationConfig()

        assert config.n_trajectories == 1000
        assert config.temperature == 2.0
        assert config.alpha == 0.5
        assert config.beta == 0.3
        assert config.gamma == 0.2
        assert config.reward_delta_threshold == 0.005


class TestTrajectoryDataset:
    """Test trajectory dataset."""

    def setup_method(self):
        """Set up test data."""
        self.trajectories = [
            {
                "observations": np.random.randn(10, 4),
                "actions": np.random.randint(0, 2, 10),
                "rewards": np.random.randn(10),
                "log_probs": np.random.randn(10),
                "values": np.random.randn(10),
            },
            {
                "observations": np.random.randn(15, 4),
                "actions": np.random.randint(0, 2, 15),
                "rewards": np.random.randn(15),
                "log_probs": np.random.randn(15),
                "values": np.random.randn(15),
            },
        ]

    def test_dataset_creation(self):
        """Test creating trajectory dataset."""
        dataset = TrajectoryDataset(self.trajectories)

        assert len(dataset) == 25  # 10 + 15
        assert dataset.observations.shape == (25, 4)
        assert dataset.actions.shape == (25,)
        assert dataset.rewards.shape == (25,)
        assert dataset.log_probs.shape == (25,)
        assert dataset.values.shape == (25,)

    def test_dataset_getitem(self):
        """Test dataset indexing."""
        dataset = TrajectoryDataset(self.trajectories)

        item = dataset[0]
        assert "observation" in item
        assert "action" in item
        assert "reward" in item
        assert "teacher_log_prob" in item
        assert "teacher_value" in item

        assert isinstance(item["observation"], torch.Tensor)
        assert isinstance(item["action"], torch.Tensor)
        assert isinstance(item["reward"], torch.Tensor)
        assert isinstance(item["teacher_log_prob"], torch.Tensor)
        assert isinstance(item["teacher_value"], torch.Tensor)


class TestKnowledgeDistillationLoss:
    """Test knowledge distillation loss."""

    def setup_method(self):
        """Set up test data."""
        self.batch_size = 32
        self.action_dim = 4

        self.student_logits = torch.randn(self.batch_size, self.action_dim)
        self.teacher_logits = torch.randn(self.batch_size, self.action_dim)
        self.student_values = torch.randn(self.batch_size, 1)
        self.teacher_values = torch.randn(self.batch_size, 1)
        self.rewards = torch.randn(self.batch_size, 1)
        self.actions = torch.randint(0, self.action_dim, (self.batch_size, 1))

        self.criterion = KnowledgeDistillationLoss(
            temperature=2.0, alpha=0.5, beta=0.3, gamma=0.2
        )

    def test_loss_creation(self):
        """Test creating loss function."""
        assert self.criterion.temperature == 2.0
        assert self.criterion.alpha == 0.5
        assert self.criterion.beta == 0.3
        assert self.criterion.gamma == 0.2

    def test_loss_forward(self):
        """Test loss forward pass."""
        loss_dict = self.criterion(
            self.student_logits,
            self.teacher_logits,
            self.student_values,
            self.teacher_values,
            self.rewards,
            self.actions,
        )

        assert "total_loss" in loss_dict
        assert "kl_loss" in loss_dict
        assert "ranking_loss" in loss_dict
        assert "value_loss" in loss_dict
        assert "fidelity_loss" in loss_dict

        assert isinstance(loss_dict["total_loss"], torch.Tensor)
        assert loss_dict["total_loss"].item() > 0

    def test_loss_components(self):
        """Test individual loss components."""
        loss_dict = self.criterion(
            self.student_logits,
            self.teacher_logits,
            self.student_values,
            self.teacher_values,
            self.rewards,
            self.actions,
        )

        # All components should be positive
        for key in ["kl_loss", "ranking_loss", "value_loss", "fidelity_loss"]:
            assert loss_dict[key].item() >= 0

        # Total loss should be weighted sum
        expected_total = (
            self.criterion.alpha * loss_dict["kl_loss"]
            + self.criterion.beta * loss_dict["ranking_loss"]
            + self.criterion.gamma * loss_dict["fidelity_loss"]
            + (1 - self.criterion.alpha - self.criterion.beta - self.criterion.gamma)
            * loss_dict["value_loss"]
        )

        assert torch.allclose(loss_dict["total_loss"], expected_total, atol=1e-6)


class TestOfflineDatasetAggregator:
    """Test offline dataset aggregator."""

    def setup_method(self):
        """Set up test data."""
        self.config = DistillationConfig(n_trajectories=10)
        self.aggregator = OfflineDatasetAggregator(self.config)

    @patch("tinyrl.distillation.gym.make")
    @patch("tinyrl.distillation.PPO.load")
    def test_generate_trajectories(self, mock_ppo_load, mock_gym_make):
        """Test trajectory generation."""
        # Mock environment
        mock_env = Mock()
        mock_env.reset.return_value = (np.array([0.1, 0.2, 0.3, 0.4]), {})
        mock_env.step.return_value = (
            np.array([0.1, 0.2, 0.3, 0.4]),
            1.0,
            False,
            False,
            {},
        )
        mock_gym_make.return_value = mock_env

        # Mock teacher model
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([0]), None)
        mock_ppo_load.return_value = mock_model

        trajectories = self.aggregator.generate_trajectories(
            "dummy_model.zip", "CartPole-v1", n_trajectories=2
        )

        assert len(trajectories) == 2
        for traj in trajectories:
            assert "observations" in traj
            assert "actions" in traj
            assert "rewards" in traj
            assert "log_probs" in traj
            assert "values" in traj

    def test_save_load_dataset(self):
        """Test saving and loading dataset."""
        trajectories = [
            {
                "observations": np.random.randn(5, 4),
                "actions": np.random.randint(0, 2, 5),
                "rewards": np.random.randn(5),
                "log_probs": np.random.randn(5),
                "values": np.random.randn(5),
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            # Save dataset
            self.aggregator.save_dataset(trajectories, temp_path)

            # Check that files were created
            assert os.path.exists(temp_path)
            metadata_path = temp_path.replace(".pkl", "_metadata.json")
            assert os.path.exists(metadata_path)

            # Load dataset
            loaded_trajectories = self.aggregator.load_dataset(temp_path)

            assert len(loaded_trajectories) == len(trajectories)
            assert len(loaded_trajectories[0]["observations"]) == 5

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            metadata_path = temp_path.replace(".pkl", "_metadata.json")
            if os.path.exists(metadata_path):
                os.unlink(metadata_path)


class TestDistillationTrainer:
    """Test distillation trainer."""

    def setup_method(self):
        """Set up test data."""
        self.config = DistillationConfig(
            batch_size=4, num_epochs=2, early_stopping_patience=5
        )
        self.trainer = DistillationTrainer(self.config)

    def test_trainer_creation(self):
        """Test creating trainer."""
        assert self.trainer.config == self.config
        assert isinstance(self.trainer.criterion, KnowledgeDistillationLoss)

    @patch("tinyrl.distillation.torch.load")
    def test_load_teacher_model(self, mock_torch_load):
        """Test loading teacher model."""
        mock_model = Mock()
        mock_torch_load.return_value = mock_model

        model = self.trainer._load_teacher_model("dummy_model.pth")

        assert model == mock_model
        mock_model.eval.assert_called_once()


class TestDistillationReport:
    """Test distillation report creation."""

    def test_report_creation(self):
        """Test creating distillation report."""
        teacher_reward = 100.0
        student_reward = 99.5
        threshold = 0.005  # 0.5%

        report = create_distillation_report(teacher_reward, student_reward, threshold)

        assert report["teacher_reward"] == 100.0
        assert report["student_reward"] == 99.5
        assert report["reward_delta"] == 0.005  # 0.5%
        assert report["reward_delta_percentage"] == 0.5
        assert report["threshold"] == 0.005
        assert report["threshold_percentage"] == 0.5
        assert report["passed"] == True
        assert report["status"] == "PASS"

    def test_report_failure(self):
        """Test report when threshold is exceeded."""
        teacher_reward = 100.0
        student_reward = 98.0  # 2% drop
        threshold = 0.005  # 0.5%

        report = create_distillation_report(teacher_reward, student_reward, threshold)

        assert report["reward_delta"] == 0.02  # 2%
        assert report["reward_delta_percentage"] == 2.0
        assert report["passed"] == False
        assert report["status"] == "FAIL"


class TestDistillationPipeline:
    """Test complete distillation pipeline."""

    @patch("tinyrl.distillation.run_distillation_pipeline")
    def test_pipeline_integration(self, mock_run_pipeline):
        """Test pipeline integration."""
        config = DistillationConfig(n_trajectories=10)

        # Mock pipeline results
        mock_results = {
            "config": config.__dict__,
            "dataset_info": {"n_trajectories": 10},
            "training_metrics": {"train_kl_divergence": 0.1},
            "evaluation_report": {
                "teacher_reward": 100.0,
                "student_reward": 99.5,
                "reward_delta_percentage": 0.5,
                "passed": True,
                "status": "PASS",
            },
        }
        mock_run_pipeline.return_value = mock_results

        # Test pipeline execution
        results = run_distillation_pipeline(
            config=config,
            teacher_model_path="dummy_model.pth",
            env_name="CartPole-v1",
            output_dir="./test_output",
        )

        assert results == mock_results
        mock_run_pipeline.assert_called_once()


class TestDeterministicBehavior:
    """Test deterministic behavior for reproducibility."""

    def test_deterministic_seed(self):
        """Test that same seed produces same results."""
        config = DistillationConfig(n_trajectories=5)

        # Set seeds
        torch.manual_seed(42)
        np.random.seed(42)

        # Create loss function
        criterion = KnowledgeDistillationLoss()

        # Create test data
        student_logits = torch.randn(10, 4)
        teacher_logits = torch.randn(10, 4)
        student_values = torch.randn(10, 1)
        teacher_values = torch.randn(10, 1)
        rewards = torch.randn(10, 1)
        actions = torch.randint(0, 4, (10, 1))

        # Compute loss
        loss1 = criterion(
            student_logits,
            teacher_logits,
            student_values,
            teacher_values,
            rewards,
            actions,
        )

        # Reset seeds and recompute
        torch.manual_seed(42)
        np.random.seed(42)

        loss2 = criterion(
            student_logits,
            teacher_logits,
            student_values,
            teacher_values,
            rewards,
            actions,
        )

        # Results should be identical
        assert torch.allclose(loss1["total_loss"], loss2["total_loss"])
        assert torch.allclose(loss1["kl_loss"], loss2["kl_loss"])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_trajectories(self):
        """Test handling empty trajectory list."""
        with pytest.raises(ValueError):
            TrajectoryDataset([])

    def test_invalid_loss_weights(self):
        """Test invalid loss weight combination."""
        with pytest.raises(ValueError):
            KnowledgeDistillationLoss(alpha=0.6, beta=0.5, gamma=0.1)

    def test_zero_temperature(self):
        """Test zero temperature handling."""
        criterion = KnowledgeDistillationLoss(temperature=0.0)

        student_logits = torch.randn(5, 4)
        teacher_logits = torch.randn(5, 4)
        student_values = torch.randn(5, 1)
        teacher_values = torch.randn(5, 1)
        rewards = torch.randn(5, 1)
        actions = torch.randint(0, 4, (5, 1))

        # Should not raise exception
        loss = criterion(
            student_logits,
            teacher_logits,
            student_values,
            teacher_values,
            rewards,
            actions,
        )

        assert isinstance(loss["total_loss"], torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
