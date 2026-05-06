from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pareto import hypervolume_2d, pareto_front
from .utils import save_json


def load_run(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "evaluation_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe {csv_path}. Primero corre evaluate.py.")
    df = pd.read_csv(csv_path)
    df["run_dir"] = str(run_dir)
    cfg_path = run_dir / "config_used.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        df["seed"] = int(cfg.get("seed", -1))
        df["env_id"] = cfg.get("env_id", "unknown")
    else:
        df["seed"] = -1
        df["env_id"] = "unknown"
    return df


def plot_all_points(df: pd.DataFrame, out_path: Path) -> None:
    obj_cols = [c for c in df.columns if c.startswith("mean_obj_")]
    if len(obj_cols) != 2:
        return
    points = df[obj_cols].to_numpy(dtype=np.float64)
    front = pareto_front(points)
    plt.figure(figsize=(7, 5))
    for seed, sub in df.groupby("seed"):
        plt.scatter(sub["mean_obj_0"], sub["mean_obj_1"], label=f"seed={seed}")
    if len(front) > 0:
        front_sorted = front[np.argsort(front[:, 0])]
        plt.plot(front_sorted[:, 0], front_sorted[:, 1], marker="o", linewidth=2, label="Frente no dominado global")
    plt.xlabel("Objetivo 0")
    plt.ylabel("Objetivo 1")
    plt.title("PCN con múltiples semillas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Agrega evaluaciones de PCN con varias semillas.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.runs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_df = pd.concat([load_run(p) for p in run_dirs], ignore_index=True)
    all_df.to_csv(out_dir / "all_seed_evaluations.csv", index=False)

    obj_cols = [c for c in all_df.columns if c.startswith("mean_obj_")]
    metrics: Dict[str, object] = {}
    if len(obj_cols) == 2:
        points = all_df[obj_cols].to_numpy(dtype=np.float64)
        hv, ref = hypervolume_2d(points)
        metrics["global_hypervolume_2d_auto_reference"] = hv
        metrics["reference_point"] = ref.tolist()
        metrics["global_pareto_points"] = pareto_front(points).tolist()
        metrics["num_global_nondominated"] = int(len(pareto_front(points)))

        hv_by_seed: List[Dict[str, float]] = []
        for seed, sub in all_df.groupby("seed"):
            hv_s, _ = hypervolume_2d(sub[obj_cols].to_numpy(dtype=np.float64), reference=ref)
            hv_by_seed.append({"seed": int(seed), "hypervolume_2d": float(hv_s)})
        hv_df = pd.DataFrame(hv_by_seed)
        hv_df.to_csv(out_dir / "hypervolume_by_seed.csv", index=False)
        metrics["mean_hypervolume_2d"] = float(hv_df["hypervolume_2d"].mean())
        metrics["std_hypervolume_2d"] = float(hv_df["hypervolume_2d"].std(ddof=1)) if len(hv_df) > 1 else 0.0
        plot_all_points(all_df, out_dir / "multi_seed_pcn_front.png")

    summary_cols = obj_cols + ["mean_episode_length"]
    agg = all_df.groupby("seed", as_index=False)[summary_cols].mean()
    agg.to_csv(out_dir / "summary_by_seed.csv", index=False)
    save_json(out_dir / "aggregate_metrics.json", metrics)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("Resumen por semilla:")
    print(agg.round(3).to_string(index=False))
    if metrics:
        print("\nMétricas globales:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.3f}")
            else:
                print(f"{k}: {v}")


if __name__ == "__main__":
    main()
