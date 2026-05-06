from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .pareto import pareto_mask, pruning_scores


def discounted_remaining_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    remaining = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        remaining[t] = running
    return remaining


def make_episode(observations, action_indices, rewards, gamma: float) -> Dict:
    observations = np.asarray(observations, dtype=np.float32)
    action_indices = np.asarray(action_indices, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float32)
    remaining = discounted_remaining_returns(rewards, gamma)
    return {
        "observations": observations,
        "action_indices": action_indices,
        "rewards": rewards,
        "remaining_returns": remaining,
        "return_vec": remaining[0].astype(np.float32),
        "length": int(len(action_indices)),
    }


def returns_array(episodes: List[Dict]) -> np.ndarray:
    if not episodes:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([np.asarray(ep["return_vec"], dtype=np.float32) for ep in episodes], axis=0)


def lengths_array(episodes: List[Dict]) -> np.ndarray:
    if not episodes:
        return np.zeros(0, dtype=np.int64)
    return np.asarray([int(ep["length"]) for ep in episodes], dtype=np.int64)


def prune_episodes(episodes: List[Dict], max_trajectories: int, crowding_threshold: float, crowding_penalty: float) -> List[Dict]:
    if len(episodes) <= max_trajectories:
        return episodes
    points = returns_array(episodes)
    scores = pruning_scores(points, crowding_threshold, crowding_penalty)
    order = np.argsort(scores)[::-1]
    keep = set(order[:max_trajectories].tolist())
    return [ep for i, ep in enumerate(episodes) if i in keep]


def sample_batch(episodes: List[Dict], batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not episodes:
        raise ValueError("No hay trayectorias disponibles para entrenar.")
    obs_batch, cond_batch, actions = [], [], []
    for _ in range(batch_size):
        ep = episodes[int(rng.integers(0, len(episodes)))]
        t = int(rng.integers(0, ep["length"]))
        horizon = float(ep["length"] - t)
        cond = np.concatenate([[horizon], ep["remaining_returns"][t].astype(np.float32)])
        obs_batch.append(ep["observations"][t])
        cond_batch.append(cond.astype(np.float32))
        actions.append(int(ep["action_indices"][t]))
    return np.stack(obs_batch).astype(np.float32), np.stack(cond_batch).astype(np.float32), np.asarray(actions, dtype=np.int64)


def compute_normalizer(episodes: List[Dict], obs_dim: int, reward_dim: int) -> Dict[str, np.ndarray]:
    if not episodes:
        return {
            "obs_mean": np.zeros(obs_dim, dtype=np.float32),
            "obs_std": np.ones(obs_dim, dtype=np.float32),
            "cond_mean": np.zeros(reward_dim + 1, dtype=np.float32),
            "cond_std": np.ones(reward_dim + 1, dtype=np.float32),
        }
    obs, cond = [], []
    for ep in episodes:
        obs.append(ep["observations"])
        for t in range(ep["length"]):
            horizon = float(ep["length"] - t)
            cond.append(np.concatenate([[horizon], ep["remaining_returns"][t]]))
    obs = np.concatenate(obs, axis=0).astype(np.float32)
    cond = np.stack(cond, axis=0).astype(np.float32)
    return {
        "obs_mean": obs.mean(axis=0).astype(np.float32),
        "obs_std": np.maximum(obs.std(axis=0), 1e-6).astype(np.float32),
        "cond_mean": cond.mean(axis=0).astype(np.float32),
        "cond_std": np.maximum(cond.std(axis=0), 1e-6).astype(np.float32),
    }


def nondominated_episode_indices(episodes: List[Dict]) -> np.ndarray:
    points = returns_array(episodes)
    if len(points) == 0:
        return np.array([], dtype=np.int64)
    return np.where(pareto_mask(points))[0]
