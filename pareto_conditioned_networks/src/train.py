from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .action_bank import action_bank_size, action_from_index, build_action_bank
from .dataset import compute_normalizer, lengths_array, make_episode, nondominated_episode_indices, prune_episodes, returns_array, sample_batch
from .networks import PCNPolicy
from .pareto import hypervolume_2d, pareto_front
from .utils import flatten_obs, infer_reward_dim, load_config, make_env, save_json, select_device, set_seed


def normalize_obs(obs: np.ndarray, normalizer: Dict[str, np.ndarray]) -> np.ndarray:
    return (obs - normalizer["obs_mean"]) / normalizer["obs_std"]


def normalize_condition(condition: np.ndarray, normalizer: Dict[str, np.ndarray]) -> np.ndarray:
    return (condition - normalizer["cond_mean"]) / normalizer["cond_std"]


def collect_random_episode(env, action_bank: np.ndarray, gamma: float, max_episode_steps: int, rng: np.random.Generator) -> Dict:
    obs, _ = env.reset()
    observations, action_indices, rewards = [], [], []
    done = False
    steps = 0
    while not done and steps < max_episode_steps:
        obs_vec = flatten_obs(obs)
        action_idx = int(rng.integers(0, action_bank_size(action_bank)))
        next_obs, reward_vec, terminated, truncated, _ = env.step(action_from_index(action_bank, action_idx))
        observations.append(obs_vec)
        action_indices.append(action_idx)
        rewards.append(np.asarray(reward_vec, dtype=np.float32).reshape(-1))
        obs = next_obs
        done = bool(terminated or truncated)
        steps += 1
    return make_episode(observations, action_indices, rewards, gamma)


def select_training_target(episodes: List[Dict], reward_dim: int, exploration_scale: float, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    nd_idx = nondominated_episode_indices(episodes)
    idx = int(rng.choice(nd_idx)) if len(nd_idx) else int(rng.integers(0, len(episodes)))
    base_return = np.asarray(episodes[idx]["return_vec"], dtype=np.float32).copy()
    horizon = int(episodes[idx]["length"])
    front = returns_array([episodes[i] for i in nd_idx]) if len(nd_idx) else returns_array(episodes)
    sigma = np.maximum(front.std(axis=0), 1e-6) if len(front) > 1 else np.maximum(np.abs(base_return), 1.0) * 0.05
    obj = int(rng.integers(0, reward_dim))
    base_return[obj] += float(rng.uniform(0.0, sigma[obj] * exploration_scale))
    return base_return.astype(np.float32), horizon


def collect_pcn_episode(env, model, action_bank, normalizer, target_return, target_horizon, gamma, max_episode_steps, device, stochastic) -> Dict:
    obs, _ = env.reset()
    desired_return = np.asarray(target_return, dtype=np.float32).copy()
    horizon = int(max(target_horizon, 1))
    observations, action_indices, rewards = [], [], []
    done = False
    steps = 0
    while not done and steps < max_episode_steps:
        obs_vec = flatten_obs(obs)
        cond = np.concatenate([[float(horizon)], desired_return]).astype(np.float32)
        obs_tensor = torch.as_tensor(normalize_obs(obs_vec, normalizer), dtype=torch.float32, device=device).unsqueeze(0)
        cond_tensor = torch.as_tensor(normalize_condition(cond, normalizer), dtype=torch.float32, device=device).unsqueeze(0)
        action_idx = int(model.act(obs_tensor, cond_tensor, stochastic=stochastic).item())
        next_obs, reward_vec, terminated, truncated, _ = env.step(action_from_index(action_bank, action_idx))
        reward_vec = np.asarray(reward_vec, dtype=np.float32).reshape(-1)
        observations.append(obs_vec)
        action_indices.append(action_idx)
        rewards.append(reward_vec)
        desired_return = desired_return - reward_vec
        horizon = max(horizon - 1, 1)
        obs = next_obs
        done = bool(terminated or truncated)
        steps += 1
    return make_episode(observations, action_indices, rewards, gamma)


def train_network(model, optimizer, episodes, normalizer, updates, batch_size, device, rng) -> float:
    model.train()
    losses = []
    for _ in range(updates):
        obs, cond, actions = sample_batch(episodes, batch_size, rng)
        obs_t = torch.as_tensor(normalize_obs(obs, normalizer), dtype=torch.float32, device=device)
        cond_t = torch.as_tensor(normalize_condition(cond, normalizer), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
        loss = F.cross_entropy(model(obs_t, cond_t), actions_t)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def log_dataset_state(episodes: List[Dict], iteration: int, phase: str) -> Dict:
    points = returns_array(episodes)
    out = {
        "iteration": iteration,
        "phase": phase,
        "num_trajectories": len(episodes),
        "num_transitions": int(sum(ep["length"] for ep in episodes)),
        "mean_length": float(lengths_array(episodes).mean()) if episodes else 0.0,
    }
    if len(points) > 0:
        front = pareto_front(points)
        out["num_nondominated"] = int(len(front))
        for i in range(points.shape[1]):
            out[f"mean_obj_{i}"] = float(points[:, i].mean())
            out[f"best_obj_{i}"] = float(points[:, i].max())
        if points.shape[1] == 2:
            hv, ref = hypervolume_2d(points)
            out["hypervolume_2d"] = float(hv)
            out["reference_obj_0"] = float(ref[0])
            out["reference_obj_1"] = float(ref[1])
    return out


def plot_dataset(points: np.ndarray, out_path: Path) -> None:
    if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
        return
    front = pareto_front(points)
    plt.figure(figsize=(7, 5))
    plt.scatter(points[:, 0], points[:, 1], alpha=0.45, label="Trayectorias recolectadas")
    if len(front) > 0:
        front_sorted = front[np.argsort(front[:, 0])]
        plt.plot(front_sorted[:, 0], front_sorted[:, 1], marker="o", linewidth=2, label="No dominadas")
    plt.xlabel("Objetivo 0")
    plt.ylabel("Objetivo 1")
    plt.title("Cobertura del dataset de PCN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena Pareto Conditioned Networks en ambientes MO-Gymnasium.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 0))
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = select_device(config.get("device", "auto"))
    run_dir = Path(config["results_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config_used.json", config)

    env = make_env(config["env_id"], seed, int(config["max_episode_steps"]))
    obs, _ = env.reset(seed=seed)
    obs_dim = int(np.asarray(obs).reshape(-1).shape[0])
    reward_dim = infer_reward_dim(env)
    action_bank = build_action_bank(env.action_space, int(config["action_bank_size"]), seed)
    np.save(run_dir / "action_bank.npy", action_bank)

    model = PCNPolicy(obs_dim, reward_dim + 1, action_bank_size(action_bank), int(config.get("embedding_dim", 64)), int(config.get("hidden_dim", 64))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    episodes: List[Dict] = []
    logs: List[Dict] = []
    gamma = float(config["gamma"])
    max_episode_steps = int(config["max_episode_steps"])

    print(f"Ambiente: {config['env_id']}")
    print(f"Dispositivo: {device}")
    print(f"Acciones prototipo/clases: {action_bank_size(action_bank)}")
    print(f"Resultados: {run_dir}")

    for _ in tqdm(range(int(config["random_episodes"])), desc="Recolectando episodios iniciales"):
        episodes.append(collect_random_episode(env, action_bank, gamma, max_episode_steps, rng))
    episodes = prune_episodes(episodes, int(config["max_trajectories"]), float(config["crowding_threshold"]), float(config["crowding_penalty"]))
    normalizer = compute_normalizer(episodes, obs_dim, reward_dim)
    logs.append(log_dataset_state(episodes, 0, "initial"))

    for iteration in tqdm(range(1, int(config["iterations"]) + 1), desc="Entrenando PCN"):
        normalizer = compute_normalizer(episodes, obs_dim, reward_dim)
        loss = train_network(model, optimizer, episodes, normalizer, int(config["updates_per_iteration"]), int(config["batch_size"]), device, rng)
        for _ in range(int(config["episodes_per_iteration"])):
            target_return, target_horizon = select_training_target(episodes, reward_dim, float(config["exploration_scale"]), rng)
            episodes.append(collect_pcn_episode(env, model, action_bank, normalizer, target_return, target_horizon, gamma, max_episode_steps, device, bool(config.get("stochastic_train_policy", True))))
        episodes = prune_episodes(episodes, int(config["max_trajectories"]), float(config["crowding_threshold"]), float(config["crowding_penalty"]))
        row = log_dataset_state(episodes, iteration, "train")
        row["loss"] = loss
        logs.append(row)
        pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)

    normalizer = compute_normalizer(episodes, obs_dim, reward_dim)
    points = returns_array(episodes)
    pd.DataFrame(points, columns=[f"return_obj_{i}" for i in range(points.shape[1])]).to_csv(run_dir / "dataset_coverage.csv", index=False)
    plot_dataset(points, run_dir / "pcn_coverage.png")
    torch.save({"model_state_dict": model.state_dict(), "config": config, "obs_dim": obs_dim, "reward_dim": reward_dim, "action_bank": action_bank, "normalizer": normalizer, "episodes": episodes}, run_dir / "checkpoints" / "pcn_final.pt")
    env.close()
    print("Entrenamiento terminado.")
    print(f"Checkpoint: {run_dir / 'checkpoints' / 'pcn_final.pt'}")


if __name__ == "__main__":
    main()
