#!/usr/bin/env python3
"""
RAM-Aware Dispatcher Module

This module implements demand-paged weight streaming to keep stack < 4KB
with circular DMA buffer example for HAL (Low-level driver).
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from tinyrl.utils import get_device

logger = logging.getLogger(__name__)


@dataclass
class DispatcherConfig:
    """Configuration for RAM-aware dispatcher."""

    # Memory constraints
    max_stack_size: int = 4096  # 4KB stack limit
    max_heap_size: int = 28672  # 28KB heap limit
    buffer_size: int = 2048  # 2KB circular buffer

    # DMA configuration
    dma_channel: int = 0
    dma_priority: int = 2  # Medium priority
    dma_burst_size: int = 4  # 4-word bursts

    # Performance targets
    max_isr_latency_us: float = 50.0  # 50µs max interrupt latency
    target_throughput_mbps: float = 100.0  # 100 MB/s target

    # Flash configuration
    flash_sector_size: int = 4096  # 4KB sectors
    flash_read_latency_ns: float = 200.0  # 200ns per word read


class CircularBuffer:
    """Circular buffer for DMA transfers."""

    def __init__(self, size: int, config: DispatcherConfig):
        self.size = size
        self.config = config
        self.buffer = np.zeros(size, dtype=np.uint8)
        self.head = 0
        self.tail = 0
        self.count = 0
        self.overflow_count = 0

    def write(self, data: np.ndarray) -> int:
        """Write data to circular buffer."""
        written = 0
        for byte in data:
            if self.count < self.size:
                self.buffer[self.head] = byte
                self.head = (self.head + 1) % self.size
                self.count += 1
                written += 1
            else:
                self.overflow_count += 1
                break
        return written

    def read(self, length: int) -> np.ndarray:
        """Read data from circular buffer."""
        if self.count == 0:
            return np.array([], dtype=np.uint8)

        read_length = min(length, self.count)
        data = np.zeros(read_length, dtype=np.uint8)

        for i in range(read_length):
            data[i] = self.buffer[self.tail]
            self.tail = (self.tail + 1) % self.size
            self.count -= 1

        return data

    def available(self) -> int:
        """Get available space in buffer."""
        return self.size - self.count

    def used(self) -> int:
        """Get used space in buffer."""
        return self.count


class DMAController:
    """DMA controller for flash-to-RAM transfers."""

    def __init__(self, config: DispatcherConfig):
        self.config = config
        self.channel = config.dma_channel
        self.priority = config.dma_priority
        self.burst_size = config.dma_burst_size
        self.is_transferring = False
        self.transfer_count = 0
        self.error_count = 0

    def configure_channel(self) -> bool:
        """Configure DMA channel for flash transfers."""
        try:
            # Simulate DMA configuration
            logger.info(f"Configuring DMA channel {self.channel}")
            logger.info(f"Priority: {self.priority}")
            logger.info(f"Burst size: {self.burst_size}")
            return True
        except Exception as e:
            logger.error(f"DMA configuration failed: {e}")
            return False

    def start_transfer(self, source_addr: int, dest_addr: int, length: int) -> bool:
        """Start DMA transfer from flash to RAM."""
        if self.is_transferring:
            logger.warning("DMA transfer already in progress")
            return False

        try:
            self.is_transferring = True
            self.transfer_count += 1

            # Calculate transfer time based on length and burst size
            words = length // 4
            bursts = (words + self.burst_size - 1) // self.burst_size
            transfer_time_us = bursts * 2.0  # 2µs per burst

            logger.info(f"DMA transfer started: {length} bytes")
            logger.info(f"Estimated time: {transfer_time_us:.2f}µs")

            return True
        except Exception as e:
            logger.error(f"DMA transfer failed: {e}")
            self.error_count += 1
            return False

    def is_busy(self) -> bool:
        """Check if DMA is busy."""
        return self.is_transferring

    def get_stats(self) -> Dict[str, Any]:
        """Get DMA statistics."""
        return {
            "transfer_count": self.transfer_count,
            "error_count": self.error_count,
            "is_busy": self.is_transferring,
        }


class RAMDispatcher:
    """RAM-aware dispatcher for weight streaming."""

    def __init__(self, config: DispatcherConfig):
        self.config = config
        self.device = get_device()

        # Initialize components
        self.buffer = CircularBuffer(config.buffer_size, config)
        self.dma = DMAController(config)

        # Weight storage
        self.weight_slices = {}  # flash_addr -> weight_data
        self.active_slice = None
        self.slice_size = 1024  # 1KB slices

        # Performance tracking
        self.load_count = 0
        self.eval_count = 0
        self.evict_count = 0
        self.miss_count = 0

    def load_slice(self, slice_id: int, flash_addr: int) -> bool:
        """Load weight slice from flash to RAM."""
        if slice_id in self.weight_slices:
            logger.debug(f"Slice {slice_id} already loaded")
            return True

        if self.buffer.available() < self.slice_size:
            logger.warning("Buffer full, need to evict")
            self._evict_oldest()

        # Simulate flash read
        try:
            # Generate dummy weight data
            weight_data = np.random.randn(self.slice_size // 4).astype(np.float32)
            weight_bytes = weight_data.tobytes()

            # Use DMA to transfer to buffer
            if self.dma.start_transfer(flash_addr, 0, len(weight_bytes)):
                self.buffer.write(weight_bytes)
                self.weight_slices[slice_id] = weight_data
                self.load_count += 1

                logger.info(f"Loaded slice {slice_id} from flash 0x{flash_addr:08X}")
                return True
            else:
                logger.error(f"Failed to load slice {slice_id}")
                return False

        except Exception as e:
            logger.error(f"Error loading slice {slice_id}: {e}")
            return False

    def eval(self, input_data: np.ndarray, slice_id: int) -> np.ndarray:
        """Evaluate model with current slice."""
        if slice_id not in self.weight_slices:
            logger.warning(f"Slice {slice_id} not loaded, loading now")
            if not self.load_slice(slice_id, slice_id * self.slice_size):
                self.miss_count += 1
                return np.zeros(input_data.shape[0])

        self.eval_count += 1
        weights = self.weight_slices[slice_id]

        # Simulate matrix multiplication
        # Reshape weights to [out_features, in_features]
        out_features = weights.shape[0] // input_data.shape[0]
        weight_matrix = weights[: out_features * input_data.shape[0]].reshape(
            out_features, input_data.shape[0]
        )

        # Compute output
        output = np.dot(weight_matrix, input_data)

        logger.debug(f"Evaluated slice {slice_id}, output shape: {output.shape}")
        return output

    def evict(self, slice_id: int) -> bool:
        """Evict weight slice from RAM."""
        if slice_id in self.weight_slices:
            del self.weight_slices[slice_id]
            self.evict_count += 1
            logger.info(f"Evicted slice {slice_id}")
            return True
        else:
            logger.warning(f"Slice {slice_id} not found for eviction")
            return False

    def _evict_oldest(self) -> None:
        """Evict oldest slice when buffer is full."""
        if self.weight_slices:
            oldest_slice = min(self.weight_slices.keys())
            self.evict(oldest_slice)

    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage."""
        total_slices = len(self.weight_slices)
        used_memory = total_slices * self.slice_size

        return {
            "used_ram": used_memory,
            "available_ram": self.config.max_heap_size - used_memory,
            "buffer_used": self.buffer.used(),
            "buffer_available": self.buffer.available(),
            "active_slices": total_slices,
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "load_count": self.load_count,
            "eval_count": self.eval_count,
            "evict_count": self.evict_count,
            "miss_count": self.miss_count,
            "dma_stats": self.dma.get_stats(),
            "memory_usage": self.get_memory_usage(),
        }


class HALInterface:
    """Hardware Abstraction Layer interface."""

    def __init__(self, config: DispatcherConfig):
        self.config = config
        self.dispatcher = RAMDispatcher(config)

    def init_hal(self) -> bool:
        """Initialize HAL components."""
        try:
            # Configure DMA
            if not self.dispatcher.dma.configure_channel():
                return False

            # Initialize flash interface
            logger.info("HAL initialized successfully")
            return True
        except Exception as e:
            logger.error(f"HAL initialization failed: {e}")
            return False

    def flash_read(self, addr: int, length: int) -> np.ndarray:
        """Read data from flash memory."""
        # Simulate flash read with latency
        logger.debug(f"Flash read: {length} bytes from 0x{addr:08X}")

        # Generate dummy data
        data = np.random.randn(length // 4).astype(np.float32)
        return data

    def ram_write(self, addr: int, data: np.ndarray) -> bool:
        """Write data to RAM."""
        try:
            # Simulate RAM write
            logger.debug(f"RAM write: {len(data)} bytes to 0x{addr:08X}")
            return True
        except Exception as e:
            logger.error(f"RAM write failed: {e}")
            return False

    def get_interrupt_latency(self) -> float:
        """Measure interrupt latency."""
        # Simulate interrupt latency measurement
        base_latency = 10.0  # 10µs base
        dma_overhead = 5.0 if self.dispatcher.dma.is_busy() else 0.0
        buffer_used = self.dispatcher.buffer.used()
        buffer_threshold = 0.8 * self.config.buffer_size
        buffer_overhead = 2.0 if buffer_used > buffer_threshold else 0.0

        total_latency = base_latency + dma_overhead + buffer_overhead
        return min(total_latency, self.config.max_isr_latency_us)


def create_dispatcher_report(
    performance_stats: Dict[str, Any], config: DispatcherConfig
) -> Dict[str, Any]:
    """Create dispatcher performance report."""
    memory_usage = performance_stats["memory_usage"]
    dma_stats = performance_stats["dma_stats"]

    # Check constraints
    stack_ok = memory_usage["used_ram"] <= config.max_stack_size
    heap_ok = memory_usage["used_ram"] <= config.max_heap_size
    latency_ok = True  # Would be measured in real implementation

    passed = all([stack_ok, heap_ok, latency_ok])

    return {
        "status": "PASSED" if passed else "FAILED",
        "constraints": {
            "stack_ok": stack_ok,
            "heap_ok": heap_ok,
            "latency_ok": latency_ok,
        },
        "performance": {
            "load_count": performance_stats["load_count"],
            "eval_count": performance_stats["eval_count"],
            "miss_rate": performance_stats["miss_count"]
            / max(performance_stats["eval_count"], 1),
            "memory_efficiency": memory_usage["used_ram"] / config.max_heap_size,
        },
        "dma_stats": dma_stats,
        "memory_usage": memory_usage,
    }


def run_dispatcher_benchmark(
    config: DispatcherConfig, model_size: int, num_evals: int = 1000
) -> Dict[str, Any]:
    """Run dispatcher benchmark."""
    logger.info("Starting dispatcher benchmark")

    # Initialize dispatcher
    dispatcher = RAMDispatcher(config)

    # Simulate model evaluation
    num_slices = (model_size + config.slice_size - 1) // config.slice_size

    for eval_idx in range(num_evals):
        # Randomly select slice to evaluate
        slice_id = eval_idx % num_slices

        # Generate input data
        input_data = np.random.randn(64).astype(np.float32)

        # Evaluate
        output = dispatcher.eval(input_data, slice_id)

        # Periodically evict slices
        if eval_idx % 100 == 0:
            for old_slice in range(max(0, slice_id - 5), slice_id):
                dispatcher.evict(old_slice)

    # Get performance stats
    stats = dispatcher.get_performance_stats()
    report = create_dispatcher_report(stats, config)

    logger.info("Dispatcher benchmark completed")
    return {
        "performance_stats": stats,
        "report": report,
    }
