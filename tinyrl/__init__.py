"""
TinyRL: Production-grade reinforcement learning for microcontrollers.

A complete RL pipeline from training to deployment on resource-constrained
devices.
"""

__version__ = "0.1.0"
__author__ = "TinyRL Contributors"
__license__ = "Apache-2.0"

from .models import A2CActor, A2CCritic, PPOActor, PPOCritic
from .train import Trainer, TrainingConfig
from .utils import get_device, set_deterministic_seed

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
