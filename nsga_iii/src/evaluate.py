from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate_policy import evaluate_genome
from .metrics import monte_carlo_hypervolume, normalize_for_max, pareto_front, pareto_front_mask, spacing_metric
from .plotting import plot_3d_front, plot_pairwise, plot_value_path
from .policy import PolicySpec
from .utils import save_json


def load_spec(data) -> PolicySpec:
    return PolicySpec(
        obs_dim=int(data["obs_dim"]),
        action_dim=int(data["action_dim"]),
        hidden_sizes=[int(x) for x in data["hidden_sizes"].tolist()],
        action_low=data["action_low"].astype(np.float32),
        action_high=data["action_high"].astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa la población final de NSGA-III.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--eval-episodes", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "config_used.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    data = np.load(run_dir / "checkpoints" / "final_population.npz", allow_pickle=True)
    spec = load_spec(data)
    genomes = data["genomes"]

    eval_episodes = int(args.eval_episodes or config.get("eval_episodes", 3))
    max_episode_steps = config.get("max_episode_steps", None)
    rows = []
    for i, genome in enumerate(genomes):
        fit, length = evaluate_genome(
            genome,
            spec,
            env_id=config["env_id"],
            episodes=eval_episodes,
            seed=int(config["seed"]) + 50000 + i * 37,
            max_episode_steps=max_episode_steps,
        )
        row = {"individual": i, "eval_episodes": eval_episodes, "mean_episode_length": length}
        for j, value in enumerate(fit):
            row[f"mean_obj_{j}"] = float(value)
        row["mean_sum_return"] = float(np.sum(fit))
        rows.append(row)

    df = pd.DataFrame(rows)
    obj_cols = [c for c in df.columns if c.startswith("mean_obj_")]
    points = df[obj_cols].to_numpy(dtype=np.float64)
    mask = pareto_front_mask(points)
    df["is_nondominated"] = mask
    df.to_csv(run_dir / "evaluation_summary.csv", index=False)

    norm, ideal, nadir = normalize_for_max(points)
    metrics = {
        "env_id": config["env_id"],
        "eval_episodes": eval_episodes,
        "n_solutions": int(len(points)),
        "n_nondominated": int(mask.sum()),
        "ideal": ideal.tolist(),
        "nadir": nadir.tolist(),
        "normalized_hv_mc": monte_carlo_hypervolume(norm, samples=int(config.get("hv_samples", 50000)), seed=int(config["seed"]) + 77),
        "spacing_normalized": spacing_metric(norm),
        "pareto_points": pareto_front(points).tolist(),
    }
    save_json(run_dir / "metrics.json", metrics)

    plot_3d_front(df, run_dir / "front_3d.png")
    plot_pairwise(df, run_dir / "objective_pairplots.png")
    plot_value_path(df, run_dir / "value_path.png")

    print("Evaluación terminada.")
    print(df.head())
    print(f"Resumen: {run_dir / 'evaluation_summary.csv'}")
    print(f"Métricas: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
