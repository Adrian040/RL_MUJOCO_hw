from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .networks import ActorCritic
from .pareto import hypervolume_2d, pareto_front
from .utils import flatten_obs, make_env, select_device, safe_weight_name, save_json


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    """Carga un checkpoint de PyTorch."""
    return torch.load(path, map_location=device)


def evaluate_checkpoint(ckpt_path: Path, eval_episodes: int, device: torch.device, seed: int) -> Dict:
    """Evalúa una política sin exploración y devuelve retornos vectoriales."""
    ckpt = load_checkpoint(ckpt_path, device)
    env = make_env(ckpt["env_id"], seed)

    model = ActorCritic(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        reward_dim=int(ckpt["reward_dim"]),
        action_low=np.asarray(ckpt["action_low"], dtype=np.float32),
        action_high=np.asarray(ckpt["action_high"], dtype=np.float32),
        hidden_sizes=ckpt["hidden_sizes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    weight = np.asarray(ckpt["weight"], dtype=np.float64)
    returns = []
    lengths = []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs = flatten_obs(obs)
        done = False
        ep_ret = np.zeros(int(ckpt["reward_dim"]), dtype=np.float64)
        ep_len = 0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = model.act_deterministic(obs_tensor).squeeze(0).cpu().numpy()
            obs, reward_vec, terminated, truncated, _ = env.step(action)
            obs = flatten_obs(obs)
            done = bool(terminated or truncated)
            ep_ret += np.asarray(reward_vec, dtype=np.float64).reshape(-1)
            ep_len += 1
        returns.append(ep_ret)
        lengths.append(ep_len)

    env.close()
    returns = np.asarray(returns)
    mean_ret = returns.mean(axis=0)
    std_ret = returns.std(axis=0)
    return {
        "checkpoint": ckpt_path.name,
        "weight": weight.tolist(),
        "weight_name": safe_weight_name(weight),
        "eval_episodes": eval_episodes,
        "mean_scalarized_return": float(mean_ret @ weight),
        "std_scalarized_return": float((returns @ weight).std()),
        "mean_episode_length": float(np.mean(lengths)),
        **{f"mean_obj_{i}": float(v) for i, v in enumerate(mean_ret)},
        **{f"std_obj_{i}": float(v) for i, v in enumerate(std_ret)},
    }


def plot_front(df: pd.DataFrame, out_path: Path) -> None:
    """Grafica el frente aproximado para ambientes con dos objetivos."""
    if "mean_obj_0" not in df or "mean_obj_1" not in df:
        return
    points = df[["mean_obj_0", "mean_obj_1"]].to_numpy(dtype=np.float64)
    front = pareto_front(points)

    plt.figure(figsize=(7, 5))
    plt.scatter(points[:, 0], points[:, 1], label="Políticas evaluadas")
    if len(front) > 0:
        front_sorted = front[np.argsort(front[:, 0])]
        plt.plot(front_sorted[:, 0], front_sorted[:, 1], marker="o", label="No dominadas")
    for _, row in df.iterrows():
        label = str(row["weight"])
        plt.annotate(label, (row["mean_obj_0"], row["mean_obj_1"]), fontsize=8)
    plt.xlabel("Objetivo 0")
    plt.ylabel("Objetivo 1")
    plt.title("Frente de Pareto aproximado")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa checkpoints de Multi-objective PPO.")
    parser.add_argument("--run-dir", type=str, required=True, help="Directorio de resultados generado por train.py.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Sobrescribe el número de episodios de evaluación.")
    parser.add_argument("--device", type=str, default=None, help="auto, cuda o cpu.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "config_used.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    eval_episodes = int(args.eval_episodes or config.get("eval_episodes", 5))
    device = select_device(args.device or config.get("device", "auto"))
    seed = int(config.get("seed", 0)) + 9999

    ckpt_paths = sorted((run_dir / "checkpoints").glob("*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No se encontraron checkpoints en {run_dir / 'checkpoints'}")

    rows: List[Dict] = []
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"Evaluando {ckpt_path.name}...")
        rows.append(evaluate_checkpoint(ckpt_path, eval_episodes, device, seed + 100 * i))

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "evaluation_summary.csv", index=False)
    plot_front(df, run_dir / "pareto_front.png")

    metrics = {}
    obj_cols = [c for c in df.columns if c.startswith("mean_obj_")]
    if len(obj_cols) == 2:
        hv, ref = hypervolume_2d(df[obj_cols].to_numpy(dtype=np.float64))
        metrics["hypervolume_2d_auto_reference"] = hv
        metrics["reference_point"] = ref.tolist()
        metrics["pareto_points"] = pareto_front(df[obj_cols].to_numpy(dtype=np.float64)).tolist()
    save_json(run_dir / "metrics.json", metrics)

    print("Evaluación terminada.")
    print(df)
    print(f"Resumen: {run_dir / 'evaluation_summary.csv'}")
    print(f"Figura: {run_dir / 'pareto_front.png'}")
    print(f"Métricas: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
