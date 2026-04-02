"""Tests for RAM-aware dispatcher."""

import numpy as np
import pytest
from tinyrl.dispatcher import CircularBuffer, DispatcherConfig


@pytest.mark.unit
class TestCircularBuffer:
    def test_write_read_roundtrip(self) -> None:
        cfg = DispatcherConfig()
        buf = CircularBuffer(8, cfg)
        data = np.array([1, 2, 3], dtype=np.uint8)
        n = buf.write(data)
        assert n == 3
        out = buf.read(3)
        assert np.array_equal(out, data)

    def test_overflow_increments_counter(self) -> None:
        cfg = DispatcherConfig()
        buf = CircularBuffer(2, cfg)
        buf.write(np.array([1, 2], dtype=np.uint8))
        buf.write(np.array([3], dtype=np.uint8))
        assert buf.overflow_count >= 1
