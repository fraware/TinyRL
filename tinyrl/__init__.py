"""
TinyRL: Production-grade reinforcement learning for microcontrollers.

A complete RL pipeline from training to deployment on resource-constrained
devices.
"""

__version__ = "0.1.0"
__author__ = "TinyRL Contributors"
__license__ = "Apache-2.0"

from .train import Trainer, TrainingConfig
from .models import PPOActor, PPOCritic, A2CActor, A2CCritic
from .utils import set_deterministic_seed, get_device

__all__ = [
    "Trainer",
    "TrainingConfig",
    "PPOActor",
    "PPOCritic",
    "A2CActor",
    "A2CCritic",
    "set_deterministic_seed",
    "get_device",
]
