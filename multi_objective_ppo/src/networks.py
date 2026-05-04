from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Inicialización ortogonal usada comúnmente en PPO."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def build_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int, output_std: float = 1.0) -> nn.Sequential:
    """Construye una red MLP con activaciones Tanh."""
    layers = []
    last_dim = input_dim
    for h in hidden_sizes:
        layers.append(layer_init(nn.Linear(last_dim, h)))
        layers.append(nn.Tanh())
        last_dim = h
    layers.append(layer_init(nn.Linear(last_dim, output_dim), std=output_std))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Actor gaussiano y crítico vectorial para recompensas multi-objetivo."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        reward_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_sizes: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        self.actor_mean = build_mlp(obs_dim, hidden_sizes, action_dim, output_std=0.01)
        self.critic = build_mlp(obs_dim, hidden_sizes, reward_dim, output_std=1.0)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Devuelve el valor vectorial por objetivo."""
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Muestrea acción y devuelve log-probabilidad, entropía y valor vectorial."""
        mean_unit = torch.tanh(self.actor_mean(obs))
        mean = mean_unit * self.action_scale + self.action_bias
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        probs = Normal(mean, std)

        if action is None:
            action = probs.sample()

        logprob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        value = self.get_value(obs)
        clipped_action = torch.max(torch.min(action, self.action_bias + self.action_scale), self.action_bias - self.action_scale)
        return action, logprob, entropy, value, clipped_action

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Acción determinista usada en evaluación."""
        mean_unit = torch.tanh(self.actor_mean(obs))
        mean = mean_unit * self.action_scale + self.action_bias
        return torch.max(torch.min(mean, self.action_bias + self.action_scale), self.action_bias - self.action_scale)
