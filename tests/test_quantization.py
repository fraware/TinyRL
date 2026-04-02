"""Tests for quantization utilities."""

import pytest
import torch
from tinyrl.quantization import (
    DifferentiableQuantizer,
    QuantizationConfig,
    QuantizationScheme,
    StraightThroughEstimator,
)


@pytest.mark.unit
class TestQuantizationConfig:
    def test_defaults(self) -> None:
        cfg = QuantizationConfig()
        assert cfg.bits == 8
        assert cfg.scheme == QuantizationScheme.SYMMETRIC


@pytest.mark.unit
class TestStraightThroughEstimator:
    def test_forward_backward_shape(self) -> None:
        x = torch.randn(4, 8, requires_grad=True)
        scale = torch.tensor(0.1, requires_grad=True)
        zp = torch.tensor(0.0, requires_grad=True)
        y = StraightThroughEstimator.apply(x, scale, zp, 8)
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None


@pytest.mark.unit
class TestDifferentiableQuantizer:
    def test_forward_1d(self) -> None:
        cfg = QuantizationConfig(per_channel=False)
        q = DifferentiableQuantizer(cfg)
        x = torch.randn(2, 16)
        y = q(x)
        assert y.shape == x.shape
