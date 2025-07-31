"""Utility functions for TinyRL training pipeline."""

import os
import random
import numpy as np
import torch
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def set_deterministic_seed(seed: int, deterministic: bool = True) -> None:
    """Set deterministic seed for reproducibility.

    Args:
        seed: Random seed to use
        deterministic: Whether to enable deterministic mode
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logger.info(f"Set deterministic seed: {seed}")


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device for training.

    Args:
        device: Device string (e.g., 'cuda', 'cpu', 'mps')

    Returns:
        torch.device: The device to use
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    return torch.device(device)


def get_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """Calculate model size in parameters and bytes.

    Args:
        model: PyTorch model

    Returns:
        Dict with parameter count and memory usage
    """
    param_count = sum(p.numel() for p in model.parameters())
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    return {
        "parameters": param_count,
        "param_memory_bytes": param_size,
        "buffer_memory_bytes": buffer_size,
        "total_memory_bytes": param_size + buffer_size,
    }


def create_env(env_name: str, **kwargs) -> Any:
    """Create a gymnasium environment with proper configuration.

    Args:
        env_name: Name of the environment
        **kwargs: Additional environment arguments

    Returns:
        The configured environment
    """
    import gymnasium as gym

    # Register custom environments if needed
    if env_name.startswith("tinyrl/"):
        from tinyrl.envs import register_envs

        register_envs()

    env = gym.make(env_name, **kwargs)
    return env


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate training configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ["env", "model", "algorithm", "training"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Validate environment
    if "name" not in config["env"]:
        raise ValueError("Environment name not specified")

    # Validate model architecture
    if "actor" not in config["model"] or "critic" not in config["model"]:
        raise ValueError("Both actor and critic must be specified in model config")

    # Validate algorithm parameters
    if "name" not in config["algorithm"]:
        raise ValueError("Algorithm name not specified")

    # Validate training parameters
    if "total_timesteps" not in config["training"]:
        raise ValueError("Total timesteps not specified")

    return True


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("tinyrl.log")],
    )


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dictionary
        path: Path to save config
    """
    import yaml

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    logger.info(f"Saved config to {path}")


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from file.

    Args:
        path: Path to config file

    Returns:
        Configuration dictionary
    """
    import yaml

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config
