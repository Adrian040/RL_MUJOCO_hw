from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBatch:
    """Tensores aplanados para actualizar PPO."""
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages_scalar: torch.Tensor
    returns_vec: torch.Tensor
    values_vec: torch.Tensor


class RolloutBuffer:
    """Buffer on-policy con GAE vectorial."""

    def __init__(
        self,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        reward_dim: int,
        device: torch.device,
        gamma: float,
        gae_lambda: float,
        weight: np.ndarray,
    ) -> None:
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.weight = torch.as_tensor(weight, dtype=torch.float32, device=device)
        self.reset()

    def reset(self) -> None:
        """Reinicia el buffer."""
        self.obs = torch.zeros((self.num_steps, self.obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.num_steps, self.action_dim), dtype=torch.float32, device=self.device)
        self.logprobs = torch.zeros(self.num_steps, dtype=torch.float32, device=self.device)
        self.rewards_vec = torch.zeros((self.num_steps, self.reward_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.num_steps, dtype=torch.float32, device=self.device)
        self.values_vec = torch.zeros((self.num_steps, self.reward_dim), dtype=torch.float32, device=self.device)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward_vec: np.ndarray,
        done: bool,
        value_vec: torch.Tensor,
    ) -> None:
        """Agrega una transición al buffer."""
        if self.ptr >= self.num_steps:
            raise RuntimeError("El buffer está lleno.")
        self.obs[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = action.detach()
        self.logprobs[self.ptr] = logprob.detach()
        self.rewards_vec[self.ptr] = torch.as_tensor(reward_vec, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = float(done)
        self.values_vec[self.ptr] = value_vec.detach().squeeze(0)
        self.ptr += 1

    def compute_returns_and_advantages(self, next_value_vec: torch.Tensor, next_done: bool) -> RolloutBatch:
        """Calcula GAE por objetivo y escalariza las ventajas."""
        advantages_vec = torch.zeros_like(self.rewards_vec, device=self.device)
        lastgaelam = torch.zeros(self.reward_dim, dtype=torch.float32, device=self.device)
        next_nonterminal = 1.0 - float(next_done)
        next_values = next_value_vec.detach().reshape(-1)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nonterminal = next_nonterminal
                values_next = next_values
            else:
                nonterminal = 1.0 - self.dones[t + 1]
                values_next = self.values_vec[t + 1]
            delta = self.rewards_vec[t] + self.gamma * values_next * nonterminal - self.values_vec[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
            advantages_vec[t] = lastgaelam

        returns_vec = advantages_vec + self.values_vec
        advantages_scalar = advantages_vec @ self.weight

        return RolloutBatch(
            obs=self.obs,
            actions=self.actions,
            logprobs=self.logprobs,
            advantages_scalar=advantages_scalar,
            returns_vec=returns_vec,
            values_vec=self.values_vec,
        )
