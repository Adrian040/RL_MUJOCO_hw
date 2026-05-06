from __future__ import annotations

from typing import Tuple

import numpy as np

from .policy import PolicySpec, act
from .utils import flatten_obs, make_env


def evaluate_genome(
    genome: np.ndarray,
    spec: PolicySpec,
    env_id: str,
    episodes: int,
    seed: int,
    max_episode_steps: int | None,
) -> Tuple[np.ndarray, float]:
    returns = []
    lengths = []
    for ep in range(episodes):
        env = make_env(env_id, seed=seed + ep, max_episode_steps=max_episode_steps)
        obs, _ = env.reset(seed=seed + ep)
        obs = flatten_obs(obs)
        done = False
        ep_ret = None
        ep_len = 0
        while not done:
            action = act(genome, obs, spec)
            next_obs, reward_vec, terminated, truncated, _ = env.step(action)
            reward_vec = np.asarray(reward_vec, dtype=np.float64).reshape(-1)
            if ep_ret is None:
                ep_ret = np.zeros_like(reward_vec, dtype=np.float64)
            ep_ret += reward_vec
            obs = flatten_obs(next_obs)
            done = bool(terminated or truncated)
            ep_len += 1
        env.close()
        returns.append(ep_ret)
        lengths.append(ep_len)
    return np.mean(np.asarray(returns), axis=0), float(np.mean(lengths))
