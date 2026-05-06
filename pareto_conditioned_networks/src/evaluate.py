from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .action_bank import action_from_index
from .dataset import lengths_array, returns_array
from .networks import PCNPolicy
from .pareto import hypervolume_2d, pareto_front, pareto_mask
from .train import normalize_condition, normalize_obs
from .utils import flatten_obs, make_env, save_json, select_device


def select_eval_targets(episodes: List[Dict], num_targets: int) -> tuple[np.ndarray, np.ndarray]:
    points = returns_array(episodes)
    lengths = lengths_array(episodes)
    if len(points) == 0:
        raise ValueError("El checkpoint no contiene trayectorias.")
    nd_indices = np.where(pareto_mask(points))[0]
    if len(nd_indices) == 0:
        nd_indices = np.arange(len(points))
    nd_indices = nd_indices[np.argsort(points[nd_indices, 0])]
    if len(nd_indices) > num_targets:
        selected = np.linspace(0, len(nd_indices) - 1, num_targets).round().astype(int)
        nd_indices = nd_indices[selected]
    return points[nd_indices].astype(np.float32), lengths[nd_indices].astype(np.int64)


def rollout_target(env, model, action_bank, normalizer, target_return, horizon, max_episode_steps, device) -> tuple[np.ndarray, int]:
    obs, _ = env.reset()
    desired_return = np.asarray(target_return, dtype=np.float32).copy()
    horizon = int(max(horizon, 1))
    done = False
    steps = 0
    returns = None
    while not done and steps < max_episode_steps:
        obs_vec = flatten_obs(obs)
        cond = np.concatenate([[float(horizon)], desired_return]).astype(np.float32)
        obs_tensor = torch.as_tensor(normalize_obs(obs_vec, normalizer), dtype=torch.float32, device=device).unsqueeze(0)
        cond_tensor = torch.as_tensor(normalize_condition(cond, normalizer), dtype=torch.float32, device=device).unsqueeze(0)
        action_idx = int(model.act(obs_tensor, cond_tensor, stochastic=False).item())
        obs, reward_vec, terminated, truncated, _ = env.step(action_from_index(action_bank, action_idx))
        reward_vec = np.asarray(reward_vec, dtype=np.float32).reshape(-1)
        if returns is None:
            returns = np.zeros_like(reward_vec, dtype=np.float64)
        returns += reward_vec
        desired_return = desired_return - reward_vec
        horizon = max(horizon - 1, 1)
        done = bool(terminated or truncated)
        steps += 1
    if returns is None:
        returns = np.zeros_like(target_return, dtype=np.float64)
    return returns.astype(np.float64), steps


def plot_evaluation(df: pd.DataFrame, dataset_points: np.ndarray, out_path: Path) -> None:
    if "mean_obj_0" not in df.columns or "mean_obj_1" not in df.columns:
        return
    eval_points = df[["mean_obj_0", "mean_obj_1"]].to_numpy(dtype=np.float64)
    front = pareto_front(eval_points)
    plt.figure(figsize=(7, 5))
    if dataset_points.ndim == 2 and dataset_points.shape[1] == 2:
        plt.scatter(dataset_points[:, 0], dataset_points[:, 1], alpha=0.25, label="Dataset")
    plt.scatter(eval_points[:, 0], eval_points[:, 1], s=70, label="Evaluación PCN")
    if len(front) > 0:
        front_sorted = front[np.argsort(front[:, 0])]
        plt.plot(front_sorted[:, 0], front_sorted[:, 1], marker="o", linewidth=2, label="No dominadas")
    plt.xlabel("Objetivo 0")
    plt.ylabel("Objetivo 1")
    plt.title("Frente aproximado por PCN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa un checkpoint de Pareto Conditioned Networks.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--num-eval-targets", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = select_device(args.device or "auto")
    ckpt = torch.load(run_dir / "checkpoints" / "pcn_final.pt", map_location=device, weights_only=False)
    config = ckpt["config"]
    device = select_device(args.device or config.get("device", "auto"))
    model = PCNPolicy(int(ckpt["obs_dim"]), int(ckpt["reward_dim"]) + 1, len(ckpt["action_bank"]), int(config.get("embedding_dim", 64)), int(config.get("hidden_dim", 64))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = make_env(config["env_id"], int(config.get("seed", 0)) + 9999, int(config["max_episode_steps"]))
    targets, horizons = select_eval_targets(ckpt["episodes"], int(args.num_eval_targets or config.get("num_eval_targets", 10)))
    eval_episodes = int(args.eval_episodes or config.get("eval_episodes", 5))
    max_episode_steps = int(config["max_episode_steps"])
    rows: List[Dict] = []
    for target_id, target in enumerate(targets):
        horizon = int(horizons[target_id])
        returns, lengths = [], []
        for _ in range(eval_episodes):
            ret, length = rollout_target(env, model, ckpt["action_bank"], ckpt["normalizer"], target, horizon, max_episode_steps, device)
            returns.append(ret)
            lengths.append(length)
        returns = np.asarray(returns, dtype=np.float64)
        mean_ret = returns.mean(axis=0)
        std_ret = returns.std(axis=0)
        row = {"target_id": target_id, "target_return": target.tolist(), "target_horizon": int(horizon), "eval_episodes": eval_episodes, "mean_episode_length": float(np.mean(lengths))}
        for i, v in enumerate(target):
            row[f"target_obj_{i}"] = float(v)
        for i, v in enumerate(mean_ret):
            row[f"mean_obj_{i}"] = float(v)
        for i, v in enumerate(std_ret):
            row[f"std_obj_{i}"] = float(v)
        rows.append(row)
    env.close()
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "evaluation_summary.csv", index=False)
    dataset_points = returns_array(ckpt["episodes"])
    plot_evaluation(df, dataset_points, run_dir / "pcn_evaluation_front.png")

    metrics = {}
    obj_cols = [c for c in df.columns if c.startswith("mean_obj_")]
    if len(obj_cols) == 2:
        points = df[obj_cols].to_numpy(dtype=np.float64)
        hv, ref = hypervolume_2d(points)
        metrics["hypervolume_2d_auto_reference"] = hv
        metrics["reference_point"] = ref.tolist()
        metrics["pareto_points"] = pareto_front(points).tolist()
        metrics["num_nondominated"] = int(len(pareto_front(points)))
    save_json(run_dir / "metrics.json", metrics)
    print("Evaluación terminada.")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
