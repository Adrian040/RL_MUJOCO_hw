from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class PolicySpec:
    obs_dim: int
    action_dim: int
    hidden_sizes: List[int]
    action_low: np.ndarray
    action_high: np.ndarray

    @property
    def layer_sizes(self) -> List[Tuple[int, int]]:
        sizes = [self.obs_dim] + list(self.hidden_sizes) + [self.action_dim]
        return [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    @property
    def n_params(self) -> int:
        total = 0
        for din, dout in self.layer_sizes:
            total += din * dout + dout
        return total


def build_policy_spec(env, hidden_sizes: Sequence[int]) -> PolicySpec:
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    return PolicySpec(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=list(hidden_sizes),
        action_low=np.asarray(env.action_space.low, dtype=np.float32).reshape(-1),
        action_high=np.asarray(env.action_space.high, dtype=np.float32).reshape(-1),
    )


def unpack_genome(genome: np.ndarray, spec: PolicySpec):
    params = []
    idx = 0
    for din, dout in spec.layer_sizes:
        w_size = din * dout
        W = genome[idx : idx + w_size].reshape(din, dout)
        idx += w_size
        b = genome[idx : idx + dout]
        idx += dout
        params.append((W, b))
    return params


def act(genome: np.ndarray, obs: np.ndarray, spec: PolicySpec) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    params = unpack_genome(genome, spec)
    for layer_id, (W, b) in enumerate(params):
        x = x @ W + b
        x = np.tanh(x)
    action_unit = x.reshape(-1)
    action_scale = (spec.action_high - spec.action_low) / 2.0
    action_bias = (spec.action_high + spec.action_low) / 2.0
    return np.clip(action_unit * action_scale + action_bias, spec.action_low, spec.action_high).astype(np.float32)


def random_genome(spec: PolicySpec, rng: np.random.Generator, init_scale: float, low: float, high: float) -> np.ndarray:
    genome = rng.normal(loc=0.0, scale=init_scale, size=spec.n_params).astype(np.float32)
    return np.clip(genome, low, high)
