from __future__ import annotations

import torch
from torch import nn


class PCNPolicy(nn.Module):
    def __init__(self, obs_dim: int, condition_dim: int, num_actions: int, embedding_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.state_embed = nn.Sequential(
            nn.Linear(obs_dim, embedding_dim),
            nn.Sigmoid(),
        )
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, embedding_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        state_z = self.state_embed(obs)
        cond_z = self.condition_embed(condition)
        return self.head(state_z * cond_z)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, condition: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
        logits = self.forward(obs, condition)
        if stochastic:
            probs = torch.softmax(logits, dim=-1)
            return torch.distributions.Categorical(probs=probs).sample()
        return torch.argmax(logits, dim=-1)
