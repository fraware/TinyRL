#!/usr/bin/env python3
"""
Tests for critic pruning and LUT folding functionality.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from tinyrl.pruning import (
    LUTFolder,
    MagnitudePruner,
    PruningConfig,
    PruningTrainer,
    create_pruning_report,
    run_pruning_pipeline,
)


class TestPruningConfig:
    """Test pruning configuration."""

    def test_config_creation(self):
        """Test configuration creation with defaults."""
        config = PruningConfig()

        assert config.target_sparsity == 0.9
        assert config.magnitude_threshold == 0.01
        assert config.global_pruning is True
        assert config.lut_bits == 8
        assert config.lut_size == 256
        assert config.monotonic_guarantee is True
        assert config.reward_delta_threshold == 0.02
        assert config.lut_size_threshold == 1024

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = PruningConfig(
            target_sparsity=0.8, lut_size=512, reward_delta_threshold=0.01
        )

        assert config.target_sparsity == 0.8
        assert config.lut_size == 512
        assert config.reward_delta_threshold == 0.01


class TestMagnitudePruner:
    """Test magnitude-based pruning."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PruningConfig()
        self.pruner = MagnitudePruner(self.config)

        # Create a simple critic model
        self.critic_model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 1))

    def test_compute_global_threshold(self):
        """Test global threshold computation."""
        threshold = self.pruner.compute_global_threshold(
            self.critic_model, target_sparsity=0.5
        )

        assert isinstance(threshold, float)
        assert threshold >= 0.0

    def test_prune_model(self):
        """Test model pruning."""
        threshold = 0.1
        pruned_model, stats = self.pruner.prune_model(self.critic_model, threshold)

        assert isinstance(pruned_model, nn.Module)
        assert "total_params" in stats
        assert "pruned_params" in stats
        assert "sparsity" in stats
        assert "threshold" in stats
        assert stats["threshold"] == threshold
        assert 0.0 <= stats["sparsity"] <= 1.0


class TestLUTFolder:
    """Test LUT folding functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PruningConfig()
        self.lut_folder = LUTFolder(self.config)

        # Create a simple critic model
        self.critic_model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 1))

    def test_generate_lut(self):
        """Test LUT generation."""
        lut_data = self.lut_folder.generate_lut(
            self.critic_model, observation_range=(-5.0, 5.0), num_samples=1000
        )

        assert "lut_values" in lut_data
        assert "lut_scales" in lut_data
        assert "min_val" in lut_data
        assert "max_val" in lut_data
        assert "lut_size" in lut_data
        assert "monotonic" in lut_data

        assert len(lut_data["lut_values"]) == self.config.lut_size
        assert len(lut_data["lut_scales"]) == self.config.lut_size
        assert lut_data["lut_size"] == self.config.lut_size
        assert lut_data["monotonic"] == self.config.monotonic_guarantee

    def test_compute_lut_hash(self):
        """Test LUT hash computation."""
        lut_data = self.lut_folder.generate_lut(self.critic_model)
        hash_value = self.lut_folder.compute_lut_hash(lut_data)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hash length

    def test_validate_monotonicity(self):
        """Test monotonicity validation."""
        lut_data = self.lut_folder.generate_lut(self.critic_model)
        is_monotonic = self.lut_folder.validate_monotonicity(lut_data)

        assert isinstance(is_monotonic, bool)

        if self.config.monotonic_guarantee:
            assert is_monotonic


class TestPruningTrainer:
    """Test pruning trainer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PruningConfig()
        self.trainer = PruningTrainer(self.config)

        # Create a simple teacher model
        self.teacher_model = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)  # Actor output
        )

    def test_trainer_creation(self):
        """Test trainer creation."""
        assert self.trainer.config == self.config
        assert hasattr(self.trainer, "pruner")
        assert hasattr(self.trainer, "lut_folder")

    def test_extract_critic(self):
        """Test critic extraction."""
        critic = self.trainer._extract_critic(self.teacher_model)

        assert isinstance(critic, nn.Module)

    def test_evaluate_performance_no_loader(self):
        """Test performance evaluation without validation loader."""
        from torch.utils.data import DataLoader, TensorDataset

        # Create dummy data
        dummy_obs = torch.randn(100, 4)
        dummy_dataset = TensorDataset(dummy_obs)
        val_loader = DataLoader(dummy_dataset, batch_size=32)

        lut_data = self.trainer.lut_folder.generate_lut(self.teacher_model)

        results = self.trainer._evaluate_performance(
            self.teacher_model,
            self.teacher_model,  # Use same model for simplicity
            lut_data,
            val_loader,
        )

        assert "reward_delta" in results
        assert "lut_size_bytes" in results
        assert "sparsity_achieved" in results
        assert "monotonic_preserved" in results


class TestPruningReport:
    """Test pruning report generation."""

    def test_report_creation(self):
        """Test report creation."""
        pruning_stats = {"sparsity": 0.85, "total_params": 1000, "pruned_params": 850}

        evaluation_results = {
            "reward_delta": 0.015,
            "lut_size_bytes": 512,
            "monotonic_preserved": True,
        }

        lut_hash = "test_hash_123"
        config = PruningConfig()

        report = create_pruning_report(
            pruning_stats, evaluation_results, lut_hash, config
        )

        assert "status" in report
        assert "passed" in report
        assert "requirements" in report
        assert "metrics" in report
        assert "thresholds" in report

        # Check that requirements are properly evaluated
        assert "sparsity_achieved" in report["requirements"]
        assert "reward_delta_ok" in report["requirements"]
        assert "lut_size_ok" in report["requirements"]
        assert "monotonic_ok" in report["requirements"]

    def test_report_failure(self):
        """Test report with failed requirements."""
        pruning_stats = {
            "sparsity": 0.7,  # Below target
            "total_params": 1000,
            "pruned_params": 700,
        }

        evaluation_results = {
            "reward_delta": 0.05,  # Above threshold
            "lut_size_bytes": 2048,  # Above threshold
            "monotonic_preserved": False,
        }

        lut_hash = "test_hash_456"
        config = PruningConfig()

        report = create_pruning_report(
            pruning_stats, evaluation_results, lut_hash, config
        )

        assert report["status"] == "FAILED"
        assert report["passed"] is False


class TestPruningPipeline:
    """Test complete pruning pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PruningConfig()

        # Create a simple teacher model
        self.teacher_model = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def test_pipeline_integration(self):
        """Test complete pipeline integration."""
        from torch.utils.data import DataLoader, TensorDataset

        # Create dummy data
        dummy_obs = torch.randn(100, 4)
        dummy_dataset = TensorDataset(dummy_obs)
        train_loader = DataLoader(dummy_dataset, batch_size=32)
        val_loader = DataLoader(dummy_dataset, batch_size=32)

        # Save teacher model temporarily
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(self.teacher_model, f.name)
            model_path = f.name

        try:
            results = run_pruning_pipeline(
                config=self.config,
                teacher_model_path=model_path,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir="./test_outputs",
            )

            assert "pruning_stats" in results
            assert "lut_data" in results
            assert "lut_hash" in results
            assert "monotonic_valid" in results
            assert "evaluation" in results
            assert "config" in results
            assert "report" in results

        finally:
            # Clean up
            Path(model_path).unlink(missing_ok=True)
            import shutil

            shutil.rmtree("./test_outputs", ignore_errors=True)


class TestDeterministicBehavior:
    """Test deterministic behavior of pruning."""

    def test_deterministic_lut_generation(self):
        """Test that LUT generation is deterministic."""
        config = PruningConfig()
        lut_folder = LUTFolder(config)

        critic_model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 1))

        # Generate LUT twice with same seed
        torch.manual_seed(42)
        lut_data_1 = lut_folder.generate_lut(critic_model)
        hash_1 = lut_folder.compute_lut_hash(lut_data_1)

        torch.manual_seed(42)
        lut_data_2 = lut_folder.generate_lut(critic_model)
        hash_2 = lut_folder.compute_lut_hash(lut_data_2)

        # Hashes should be identical
        assert hash_1 == hash_2

    def test_deterministic_pruning(self):
        """Test that pruning is deterministic."""
        config = PruningConfig()
        pruner = MagnitudePruner(config)

        critic_model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 1))

        # Prune twice with same seed
        torch.manual_seed(42)
        pruned_1, stats_1 = pruner.prune_model(critic_model, threshold=0.1)

        torch.manual_seed(42)
        pruned_2, stats_2 = pruner.prune_model(critic_model, threshold=0.1)

        # Stats should be identical
        assert stats_1["sparsity"] == stats_2["sparsity"]
        assert stats_1["pruned_params"] == stats_2["pruned_params"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_model(self):
        """Test pruning with empty model."""
        config = PruningConfig()
        pruner = MagnitudePruner(config)

        # Create empty model
        empty_model = nn.Sequential()

        threshold = pruner.compute_global_threshold(empty_model, 0.5)
        assert threshold == 0.0

        pruned_model, stats = pruner.prune_model(empty_model, threshold)
        assert stats["total_params"] == 0
        assert stats["sparsity"] == 0.0

    def test_invalid_sparsity(self):
        """Test invalid sparsity values."""
        config = PruningConfig(target_sparsity=1.5)  # Invalid
        pruner = MagnitudePruner(config)

        critic_model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 1))

        # Should handle gracefully
        threshold = pruner.compute_global_threshold(critic_model, 1.5)
        assert threshold >= 0.0

    def test_lut_monotonicity_disabled(self):
        """Test LUT generation with monotonicity disabled."""
        config = PruningConfig(monotonic_guarantee=False)
        lut_folder = LUTFolder(config)

        critic_model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 1))

        lut_data = lut_folder.generate_lut(critic_model)
        is_monotonic = lut_folder.validate_monotonicity(lut_data)

        # Should return True when monotonicity is disabled
        assert is_monotonic is True
