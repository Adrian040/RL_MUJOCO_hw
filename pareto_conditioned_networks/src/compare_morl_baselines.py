from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pareto import hypervolume_2d, normalize_points, pareto_front
from .utils import save_json


def read_points(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    obj_cols = [c for c in df.columns if c.startswith("mean_obj_")]
    obj_cols = sorted(obj_cols, key=lambda x: int(x.split("_")[-1]))
    return df[obj_cols].to_numpy(dtype=np.float64)


def method_summary(name: str, points: np.ndarray, hv_norm: float | None) -> dict:
    front = pareto_front(points)
    row = {
        "Método": name,
        "Soluciones evaluadas": len(points),
        "Soluciones no dominadas": len(front),
    }
    for i in range(points.shape[1]):
        row[f"Mejor objetivo {i}"] = float(points[:, i].max())
    if hv_norm is not None:
        row["HV normalizado"] = float(hv_norm)
    return row


def plot_comparison(local_points: np.ndarray, baseline_points: np.ndarray, out_path: Path) -> None:
    if local_points.shape[1] < 2 or baseline_points.shape[1] < 2:
        return
    local_front = pareto_front(local_points)
    baseline_front = pareto_front(baseline_points)
    plt.figure(figsize=(7, 5))
    plt.scatter(local_points[:, 0], local_points[:, 1], alpha=0.7, s=70, label="PCN local")
    plt.scatter(baseline_points[:, 0], baseline_points[:, 1], alpha=0.7, s=70, label="PCN MORL-Baselines")
    if len(local_front) > 0 and local_front.shape[1] >= 2:
        f = local_front[np.argsort(local_front[:, 0])]
        plt.plot(f[:, 0], f[:, 1], marker="o", linewidth=2, label="Local no dominado")
    if len(baseline_front) > 0 and baseline_front.shape[1] >= 2:
        f = baseline_front[np.argsort(baseline_front[:, 0])]
        plt.plot(f[:, 0], f[:, 1], marker="o", linewidth=2, label="Baseline no dominado")
    plt.xlabel("Objetivo 0")
    plt.ylabel("Objetivo 1")
    plt.title("Comparación PCN local vs PCN MORL-Baselines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara PCN local contra el PCN oficial de MORL-Baselines.")
    parser.add_argument("--local-run", type=str, required=True)
    parser.add_argument("--baseline-run", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    local_run = Path(args.local_run)
    baseline_run = Path(args.baseline_run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_points = read_points(local_run / "evaluation_summary.csv")
    baseline_points = read_points(baseline_run / "evaluation_summary.csv")
    if local_points.shape[1] != baseline_points.shape[1]:
        raise ValueError("Los métodos tienen distinta dimensión de objetivos.")

    hv_local = None
    hv_baseline = None
    metrics = {}
    all_points = np.vstack([local_points, baseline_points])
    ideal = all_points.max(axis=0)
    nadir = all_points.min(axis=0)
    metrics["ideal"] = ideal.tolist()
    metrics["nadir"] = nadir.tolist()

    if local_points.shape[1] == 2:
        ref_norm = np.array([-0.05, -0.05], dtype=np.float64)
        local_norm = normalize_points(local_points, ideal, nadir)
        baseline_norm = normalize_points(baseline_points, ideal, nadir)
        hv_local, _ = hypervolume_2d(local_norm, reference=ref_norm)
        hv_baseline, _ = hypervolume_2d(baseline_norm, reference=ref_norm)
        metrics["reference_normalized"] = ref_norm.tolist()
        metrics["hypervolume_local_normalized"] = float(hv_local)
        metrics["hypervolume_morl_baselines_normalized"] = float(hv_baseline)

    comparison = pd.DataFrame([
        method_summary("PCN local", local_points, hv_local),
        method_summary("PCN MORL-Baselines", baseline_points, hv_baseline),
    ])
    comparison.to_csv(out_dir / "comparison_local_vs_morl_baselines.csv", index=False)
    plot_comparison(local_points, baseline_points, out_dir / "comparison_local_vs_morl_baselines.png")
    save_json(out_dir / "comparison_metrics.json", metrics)
    print(comparison.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
