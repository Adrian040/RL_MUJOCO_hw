from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA no está disponible. Se usará CPU.")
        return torch.device("cpu")
    return requested


def make_env(env_id: str, seed: int, max_episode_steps: int | None = None):
    import gymnasium as gym
    import mo_gymnasium as mo_gym

    env = mo_gym.make(env_id)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def flatten_obs(obs: Any) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def infer_reward_dim(env) -> int:
    if hasattr(env.unwrapped, "reward_space"):
        return int(np.asarray(env.unwrapped.reward_space.sample()).reshape(-1).shape[0])
    obs, _ = env.reset()
    action = env.action_space.sample()
    _, reward_vec, _, _, _ = env.step(action)
    env.reset()
    return int(np.asarray(reward_vec).reshape(-1).shape[0])
