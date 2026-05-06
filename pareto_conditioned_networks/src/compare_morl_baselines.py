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
    return df[["mean_obj_0", "mean_obj_1"]].to_numpy(dtype=np.float64)


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
    all_points = np.vstack([local_points, baseline_points])
    ideal = all_points.max(axis=0)
    nadir = all_points.min(axis=0)
    ref_norm = np.array([-0.05, -0.05], dtype=np.float64)

    local_norm = normalize_points(local_points, ideal, nadir)
    baseline_norm = normalize_points(baseline_points, ideal, nadir)
    hv_local, _ = hypervolume_2d(local_norm, reference=ref_norm)
    hv_baseline, _ = hypervolume_2d(baseline_norm, reference=ref_norm)

    local_front = pareto_front(local_points)
    baseline_front = pareto_front(baseline_points)
    comparison = pd.DataFrame({
        "Método": ["PCN local", "PCN MORL-Baselines"],
        "Soluciones evaluadas": [len(local_points), len(baseline_points)],
        "Soluciones no dominadas": [len(local_front), len(baseline_front)],
        "Mejor objetivo 0": [local_points[:, 0].max(), baseline_points[:, 0].max()],
        "Mejor objetivo 1": [local_points[:, 1].max(), baseline_points[:, 1].max()],
        "HV normalizado": [hv_local, hv_baseline],
    })
    comparison.to_csv(out_dir / "comparison_local_vs_morl_baselines.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(local_points[:, 0], local_points[:, 1], alpha=0.7, s=70, label="PCN local")
    plt.scatter(baseline_points[:, 0], baseline_points[:, 1], alpha=0.7, s=70, label="PCN MORL-Baselines")
    if len(local_front) > 0:
        f = local_front[np.argsort(local_front[:, 0])]
        plt.plot(f[:, 0], f[:, 1], marker="o", linewidth=2, label="Local no dominado")
    if len(baseline_front) > 0:
        f = baseline_front[np.argsort(baseline_front[:, 0])]
        plt.plot(f[:, 0], f[:, 1], marker="o", linewidth=2, label="Baseline no dominado")
    plt.xlabel("Objetivo 0")
    plt.ylabel("Objetivo 1")
    plt.title("Comparación PCN local vs PCN MORL-Baselines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_local_vs_morl_baselines.png", dpi=300)
    plt.close()

    save_json(out_dir / "comparison_metrics.json", {
        "ideal": ideal.tolist(),
        "nadir": nadir.tolist(),
        "reference_normalized": ref_norm.tolist(),
        "hypervolume_local_normalized": float(hv_local),
        "hypervolume_morl_baselines_normalized": float(hv_baseline),
    })
    print(comparison.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
