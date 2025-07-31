"""Neural network architectures for TinyRL algorithms."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union
import numpy as np


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
        activation: str = "tanh",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm

        # Build layers
        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "elu":
            return nn.ELU()
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using orthogonal initialization."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class PPOActor(nn.Module):
    """Actor network for PPO algorithm."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [64, 64],
        activation: str = "tanh",
        std: Union[float, torch.Tensor] = 0.0,
        log_std_init: float = -0.5,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        # Policy network
        self.policy_net = MLP(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        # Standard deviation
        if isinstance(std, float):
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        else:
            self.log_std = nn.Parameter(torch.log(std))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and log std."""
        action_mean = self.policy_net(obs)
        action_log_std = self.log_std.expand_as(action_mean)
        return action_mean, action_log_std

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability."""
        action_mean, action_log_std = self.forward(obs)

        if deterministic:
            action = action_mean
        else:
            action_std = torch.exp(action_log_std)
            noise = torch.randn_like(action_mean)
            action = action_mean + noise * action_std

        log_prob = self._log_prob(action, action_mean, action_log_std)
        return action, log_prob

    def _log_prob(
        self, action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of action."""
        std = torch.exp(log_std)
        log_prob = -0.5 * ((action - mean) / std).pow(2) - log_std
        log_prob = log_prob.sum(-1, keepdim=True)
        return log_prob


class PPOCritic(nn.Module):
    """Critic network for PPO algorithm."""

    def __init__(
        self, obs_dim: int, hidden_sizes: List[int] = [64, 64], activation: str = "tanh"
    ):
        super().__init__()

        self.value_net = MLP(
            input_dim=obs_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning value estimate."""
        return self.value_net(obs)


class A2CActor(nn.Module):
    """Actor network for A2C algorithm."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [128, 128],
        activation: str = "relu",
        std: Union[float, torch.Tensor] = 0.0,
        log_std_init: float = -0.5,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        # Policy network
        self.policy_net = MLP(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        # Standard deviation
        if isinstance(std, float):
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        else:
            self.log_std = nn.Parameter(torch.log(std))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and log std."""
        action_mean = self.policy_net(obs)
        action_log_std = self.log_std.expand_as(action_mean)
        return action_mean, action_log_std

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability."""
        action_mean, action_log_std = self.forward(obs)

        if deterministic:
            action = action_mean
        else:
            action_std = torch.exp(action_log_std)
            noise = torch.randn_like(action_mean)
            action = action_mean + noise * action_std

        log_prob = self._log_prob(action, action_mean, action_log_std)
        return action, log_prob

    def _log_prob(
        self, action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of action."""
        std = torch.exp(log_std)
        log_prob = -0.5 * ((action - mean) / std).pow(2) - log_std
        log_prob = log_prob.sum(-1, keepdim=True)
        return log_prob


class A2CCritic(nn.Module):
    """Critic network for A2C algorithm."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: List[int] = [128, 128],
        activation: str = "relu",
    ):
        super().__init__()

        self.value_net = MLP(
            input_dim=obs_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning value estimate."""
        return self.value_net(obs)


def create_actor_critic(
    obs_dim: int, action_dim: int, algorithm: str, model_config: Dict
) -> Tuple[nn.Module, nn.Module]:
    """Create actor and critic networks based on algorithm and config.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        algorithm: Algorithm name ('ppo' or 'a2c')
        model_config: Model configuration

    Returns:
        Tuple of (actor, critic) networks
    """
    actor_config = model_config["actor"]
    critic_config = model_config["critic"]

    if algorithm.lower() == "ppo":
        actor = PPOActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=actor_config["hidden_sizes"],
            activation=actor_config["activation"],
            std=actor_config.get("std", 0.0),
        )
        critic = PPOCritic(
            obs_dim=obs_dim,
            hidden_sizes=critic_config["hidden_sizes"],
            activation=critic_config["activation"],
        )
    elif algorithm.lower() == "a2c":
        actor = A2CActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=actor_config["hidden_sizes"],
            activation=actor_config["activation"],
            std=actor_config.get("std", 0.0),
        )
        critic = A2CCritic(
            obs_dim=obs_dim,
            hidden_sizes=critic_config["hidden_sizes"],
            activation=critic_config["activation"],
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return actor, critic
