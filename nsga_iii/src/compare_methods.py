from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import monte_carlo_hypervolume, normalize_for_max, pareto_front, pareto_front_mask, spacing_metric
from .utils import save_json


def load_points(path: str | Path) -> np.ndarray:
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.startswith("mean_obj_")]
    if not cols:
        raise ValueError(f"No hay columnas mean_obj_* en {path}")
    return df[cols].to_numpy(dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara NSGA-III con un baseline MORL.")
    parser.add_argument("--nsga-path", type=str, required=True)
    parser.add_argument("--baseline-path", type=str, required=True)
    parser.add_argument("--baseline-name", type=str, default="PGMORL")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--hv-samples", type=int, default=100000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nsga = load_points(args.nsga_path)
    baseline = load_points(args.baseline_path)
    all_points = np.vstack([nsga, baseline])
    _, ideal, nadir = normalize_for_max(all_points)
    nsga_norm, _, _ = normalize_for_max(nsga, ideal=ideal, nadir=nadir)
    baseline_norm, _, _ = normalize_for_max(baseline, ideal=ideal, nadir=nadir)

    nsga_mask = pareto_front_mask(nsga)
    baseline_mask = pareto_front_mask(baseline)

    metrics = {
        "ideal_common": ideal.tolist(),
        "nadir_common": nadir.tolist(),
        "NSGAIII": {
            "n_solutions": int(len(nsga)),
            "n_nondominated": int(nsga_mask.sum()),
            "normalized_hv_mc": monte_carlo_hypervolume(nsga_norm, samples=args.hv_samples, seed=123),
            "spacing_normalized": spacing_metric(nsga_norm),
            "best_objectives": nsga.max(axis=0).tolist(),
        },
        args.baseline_name: {
            "n_solutions": int(len(baseline)),
            "n_nondominated": int(baseline_mask.sum()),
            "normalized_hv_mc": monte_carlo_hypervolume(baseline_norm, samples=args.hv_samples, seed=456),
            "spacing_normalized": spacing_metric(baseline_norm),
            "best_objectives": baseline.max(axis=0).tolist(),
        },
    }
    save_json(out_dir / "comparison_metrics.json", metrics)

    pd.DataFrame([
        {"Método": "NSGA-III", **metrics["NSGAIII"]},
        {"Método": args.baseline_name, **metrics[args.baseline_name]},
    ]).to_csv(out_dir / "comparison_table.csv", index=False)

    if nsga.shape[1] >= 3:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(nsga[:, 0], nsga[:, 1], nsga[:, 2], alpha=0.55, label="NSGA-III")
        ax.scatter(baseline[:, 0], baseline[:, 1], baseline[:, 2], alpha=0.55, label=args.baseline_name)
        ax.set_xlabel("Objetivo 0")
        ax.set_ylabel("Objetivo 1")
        ax.set_zlabel("Objetivo 2")
        ax.set_title("Comparación de soluciones evaluadas")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "comparison_3d.png", dpi=250)
        plt.close()

    pairs = [(0, 1)] if nsga.shape[1] == 2 else [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(nsga[:, i], nsga[:, j], alpha=0.55, label="NSGA-III")
        ax.scatter(baseline[:, i], baseline[:, j], alpha=0.55, label=args.baseline_name)
        ax.set_xlabel(f"Objetivo {i}")
        ax.set_ylabel(f"Objetivo {j}")
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_pairplots.png", dpi=250)
    plt.close()

    print("Comparación terminada.")
    print(pd.read_csv(out_dir / "comparison_table.csv"))


if __name__ == "__main__":
    main()
