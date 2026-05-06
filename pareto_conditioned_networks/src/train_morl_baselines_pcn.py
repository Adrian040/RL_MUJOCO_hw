from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pareto import hypervolume_2d, pareto_front
from .utils import infer_reward_dim, load_config, make_env, save_json, select_device, set_seed


def patch_morl_baselines_numpy2() -> None:
    import morl_baselines.multi_policy.pcn.pcn as pcn_module

    def crowding_distance_numpy2(points):
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            return np.array([], dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        points = (points - points.min(axis=0)) / (np.ptp(points, axis=0) + 1e-8)
        dim_sorted = np.argsort(points, axis=0)
        point_sorted = np.take_along_axis(points, dim_sorted, axis=0)
        distances = np.abs(point_sorted[:-2] - point_sorted[2:])
        distances = np.pad(distances, ((1,), (0,)), constant_values=1)
        crowding = np.zeros(points.shape, dtype=np.float64)
        crowding[dim_sorted, np.arange(points.shape[-1])] = distances
        return np.sum(crowding, axis=-1)

    pcn_module.crowding_distance = crowding_distance_numpy2


def auto_vector(value, reward_dim: int, kind: str) -> np.ndarray:
    if value != "auto":
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if len(arr) == reward_dim:
            return arr
        if kind == "scaling" and len(arr) == reward_dim + 1:
            return arr
    if kind == "scaling":
        return np.asarray([0.01] * (reward_dim + 1), dtype=np.float32)
    if kind == "ref":
        return np.asarray([-1000.0] * reward_dim, dtype=np.float32)
    if kind == "max_return":
        return np.asarray([10000.0] * reward_dim, dtype=np.float32)
    raise ValueError(kind)


def plot_points(points: np.ndarray, out_path: Path) -> None:
    if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
        return
    front = pareto_front(points)
    plt.figure(figsize=(7, 5))
    plt.scatter(points[:, 0], points[:, 1], s=70, label="MORL-Baselines PCN")
    if len(front) > 0 and front.shape[1] >= 2:
        front_sorted = front[np.argsort(front[:, 0])]
        plt.plot(front_sorted[:, 0], front_sorted[:, 1], marker="o", linewidth=2, label="No dominadas")
    plt.xlabel("Objetivo 0")
    plt.ylabel("Objetivo 1")
    plt.title("Baseline oficial: PCN de MORL-Baselines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el PCN oficial de MORL-Baselines.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    try:
        patch_morl_baselines_numpy2()
        from morl_baselines.multi_policy.pcn.pcn import PCN
    except Exception as exc:
        raise RuntimeError("No se pudo importar MORL-Baselines. Instala con: pip install morl-baselines") from exc

    config = load_config(args.config)
    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = select_device(config.get("device", "auto"))
    out_dir = Path(args.out_dir) if args.out_dir else Path(config["results_dir"]) / "morl_baselines_pcn"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_episode_steps = int(config["max_episode_steps"])
    env = make_env(config["env_id"], seed, max_episode_steps)
    eval_env = make_env(config["env_id"], seed + 9999, max_episode_steps)
    reward_dim = infer_reward_dim(env)

    ref_point = auto_vector(config.get("baseline_ref_point", "auto"), reward_dim, "ref")
    max_return = auto_vector(config.get("baseline_max_return", "auto"), reward_dim, "max_return")
    scaling_factor = auto_vector(config.get("baseline_scaling_factor", "auto"), reward_dim, "scaling")

    agent = PCN(
        env=env,
        scaling_factor=scaling_factor,
        learning_rate=float(config.get("baseline_learning_rate", config["learning_rate"])),
        gamma=float(config["gamma"]),
        batch_size=int(config.get("baseline_batch_size", config["batch_size"])),
        hidden_dim=int(config.get("baseline_hidden_dim", config.get("hidden_dim", 64))),
        noise=float(config.get("baseline_noise", 0.2)),
        log=False,
        seed=seed,
        device=str(device),
    )
    agent.save = lambda *_, **__: None

    agent.train(
        total_timesteps=int(config["baseline_total_timesteps"]),
        eval_env=eval_env,
        ref_point=ref_point,
        known_pareto_front=None,
        num_eval_weights_for_eval=25,
        num_er_episodes=int(config["baseline_num_er_episodes"]),
        num_step_episodes=int(config["baseline_num_step_episodes"]),
        num_model_updates=int(config["baseline_num_model_updates"]),
        max_return=max_return,
        max_buffer_size=int(config["baseline_max_buffer_size"]),
        num_points_pf=int(config["baseline_num_points_pf"]),
    )

    e_returns, targets, distances = agent.evaluate(eval_env, max_return=max_return, n=int(config["baseline_num_points_pf"]))
    e_returns = np.asarray(e_returns, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)

    rows: List[Dict] = []
    for i, ret in enumerate(e_returns):
        row = {"target_id": i, "distance_to_command": float(distances[i]) if i < len(distances) else np.nan}
        if i < len(targets):
            for j, v in enumerate(targets[i]):
                row[f"target_obj_{j}"] = float(v)
        for j, v in enumerate(ret):
            row[f"mean_obj_{j}"] = float(v)
            row[f"std_obj_{j}"] = 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "evaluation_summary.csv", index=False)
    plot_points(e_returns, out_dir / "morl_baselines_pcn_front.png")

    metrics: Dict[str, object] = {}
    if e_returns.ndim == 2:
        metrics["num_nondominated"] = int(len(pareto_front(e_returns)))
        metrics["pareto_points"] = pareto_front(e_returns).tolist()
        if e_returns.shape[1] == 2:
            hv, ref = hypervolume_2d(e_returns)
            metrics["hypervolume_2d_auto_reference"] = float(hv)
            metrics["reference_point"] = ref.tolist()
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "config_used.json", config)
    print("Baseline MORL-Baselines PCN terminado.")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
