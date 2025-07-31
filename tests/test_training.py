"""Tests for TinyRL training pipeline."""

import os
import pytest
import torch
import numpy as np
from omegaconf import OmegaConf

from tinyrl.train import Trainer
from tinyrl.models import PPOActor, PPOCritic, A2CActor, A2CCritic
from tinyrl.utils import set_deterministic_seed, get_model_size


class TestTrainingPipeline:
    """Test the training pipeline components."""

    def setup_method(self):
        """Setup for each test method."""
        set_deterministic_seed(42, deterministic=True)

    def test_model_creation(self):
        """Test that models can be created correctly."""
        # Test PPO models
        ppo_actor = PPOActor(obs_dim=4, action_dim=2)
        ppo_critic = PPOCritic(obs_dim=4)

        assert ppo_actor.obs_dim == 4
        assert ppo_actor.action_dim == 2
        assert ppo_critic.value_net.output_dim == 1

        # Test A2C models
        a2c_actor = A2CActor(obs_dim=8, action_dim=4)
        a2c_critic = A2CCritic(obs_dim=8)

        assert a2c_actor.obs_dim == 8
        assert a2c_actor.action_dim == 4
        assert a2c_critic.value_net.output_dim == 1

    def test_model_forward_pass(self):
        """Test model forward passes."""
        # Test PPO actor
        ppo_actor = PPOActor(obs_dim=4, action_dim=2)
        obs = torch.randn(1, 4)

        action_mean, action_log_std = ppo_actor(obs)
        assert action_mean.shape == (1, 2)
        assert action_log_std.shape == (1, 2)

        # Test PPO critic
        ppo_critic = PPOCritic(obs_dim=4)
        value = ppo_critic(obs)
        assert value.shape == (1, 1)

        # Test A2C actor
        a2c_actor = A2CActor(obs_dim=8, action_dim=4)
        obs = torch.randn(1, 8)

        action_mean, action_log_std = a2c_actor(obs)
        assert action_mean.shape == (1, 4)
        assert action_log_std.shape == (1, 4)

        # Test A2C critic
        a2c_critic = A2CCritic(obs_dim=8)
        value = a2c_critic(obs)
        assert value.shape == (1, 1)

    def test_model_size_calculation(self):
        """Test model size calculation."""
        ppo_actor = PPOActor(obs_dim=4, action_dim=2)
        ppo_critic = PPOCritic(obs_dim=4)

        actor_size = get_model_size(ppo_actor)
        critic_size = get_model_size(ppo_critic)

        assert "parameters" in actor_size
        assert "total_memory_bytes" in actor_size
        assert "parameters" in critic_size
        assert "total_memory_bytes" in critic_size

        # Check that sizes are reasonable
        assert actor_size["parameters"] > 0
        assert critic_size["parameters"] > 0
        assert actor_size["total_memory_bytes"] > 0
        assert critic_size["total_memory_bytes"] > 0

    def test_deterministic_seed(self):
        """Test that deterministic seeding works."""
        # Set seed
        set_deterministic_seed(42, deterministic=True)

        # Generate some random numbers
        torch_rand1 = torch.randn(10)
        np_rand1 = np.random.randn(10)

        # Reset seed and generate again
        set_deterministic_seed(42, deterministic=True)
        torch_rand2 = torch.randn(10)
        np_rand2 = np.random.randn(10)

        # Should be identical
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)

    def test_device_selection(self):
        """Test device selection logic."""
        from tinyrl.utils import get_device

        device = get_device()
        assert device in [
            torch.device("cpu"),
            torch.device("cuda"),
            torch.device("mps"),
        ]

        # Test explicit device selection
        cpu_device = get_device("cpu")
        assert cpu_device == torch.device("cpu")

    def test_config_validation(self):
        """Test configuration validation."""
        from tinyrl.utils import validate_config

        # Valid config
        valid_config = {
            "env": {"name": "CartPole-v1"},
            "model": {
                "actor": {"hidden_sizes": [64, 64]},
                "critic": {"hidden_sizes": [64, 64]},
            },
            "algorithm": {"name": "ppo"},
            "training": {"total_timesteps": 1000},
        }

        assert validate_config(valid_config) is True

        # Invalid config - missing env
        invalid_config = {
            "model": {"actor": {}, "critic": {}},
            "algorithm": {"name": "ppo"},
            "training": {"total_timesteps": 1000},
        }

        with pytest.raises(ValueError, match="Environment name not specified"):
            validate_config(invalid_config)

    @pytest.mark.slow
    def test_mini_training_run(self):
        """Test a minimal training run."""
        # Create a minimal config for testing
        config_dict = {
            "env": {"name": "CartPole-v1", "max_episode_steps": 100},
            "model": {
                "actor": {
                    "type": "mlp",
                    "hidden_sizes": [32, 32],
                    "activation": "tanh",
                },
                "critic": {
                    "type": "mlp",
                    "hidden_sizes": [32, 32],
                    "activation": "tanh",
                },
            },
            "algorithm": {
                "name": "ppo",
                "learning_rate": 3e-4,
                "n_steps": 64,
                "batch_size": 32,
                "n_epochs": 4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "clip_range_vf": None,
                "normalize_advantage": True,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "use_sde": False,
                "sde_sample_freq": -1,
                "target_kl": None,
            },
            "training": {
                "total_timesteps": 1000,
                "eval_freq": 500,
                "n_eval_episodes": 5,
                "save_freq": 500,
                "log_freq": 100,
            },
            "optimizer": {
                "type": "adamw",
                "weight_decay": 1e-4,
                "eps": 1e-7,
                "amsgrad": False,
            },
            "compilation": {"enabled": False},
            "logging": {"wandb": {"enabled": False}, "tensorboard": False, "csv": True},
            "seed": 42,
            "deterministic": True,
            "output": {
                "dir": "./test_output",
                "save_model": True,
                "save_replay_buffer": False,
                "save_vecnormalize": True,
            },
            "callbacks": [],
        }

        config = OmegaConf.create(config_dict)

        # Create trainer
        trainer = Trainer(config)

        # Run a very short training
        trainer.train()

        # Check that output files were created
        assert os.path.exists("./test_output/final_model.zip")
        assert os.path.exists("./test_output/config.yaml")

        # Clean up
        import shutil

        shutil.rmtree("./test_output", ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
