from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Carga un archivo YAML de configuración."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Guarda un diccionario en formato JSON con indentación."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    """Fija semillas para mejorar reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device: str) -> torch.device:
    """Selecciona CUDA si está disponible y la configuración lo permite."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA no está disponible. Se usará CPU.")
        return torch.device("cpu")
    return requested


def parse_weights(weights: List[List[float]]) -> np.ndarray:
    """Valida y normaliza una lista de vectores de preferencias."""
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("weights debe ser una lista de listas, por ejemplo [[0.5, 0.5]].")
    if np.any(w < 0):
        raise ValueError("Los pesos deben ser no negativos.")
    sums = w.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Cada vector de pesos debe tener suma positiva.")
    return w / sums


def make_env(env_id: str, seed: int):
    """Crea un ambiente MO-Gymnasium y fija semilla."""
    import mo_gymnasium as mo_gym

    env = mo_gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def flatten_obs(obs: Any) -> np.ndarray:
    """Convierte observaciones a un vector float32."""
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def safe_weight_name(weight: np.ndarray) -> str:
    """Crea un nombre corto y seguro para un vector de pesos."""
    return "w_" + "_".join(f"{x:.2f}".replace(".", "p") for x in weight)
