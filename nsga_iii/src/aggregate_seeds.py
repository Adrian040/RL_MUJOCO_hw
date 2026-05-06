from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import monte_carlo_hypervolume, normalize_for_max, pareto_front, pareto_front_mask, spacing_metric
from .plotting import plot_3d_front, plot_pairwise, plot_value_path
from .utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Agrega resultados de varias semillas NSGA-III.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--hv-samples", type=int, default=80000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    seed_metrics = []
    for run in args.runs:
        run_dir = Path(run)
        df = pd.read_csv(run_dir / "evaluation_summary.csv")
        with open(run_dir / "config_used.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        df["seed"] = int(cfg.get("seed", len(frames)))
        df["run_dir"] = str(run_dir)
        frames.append(df)
        with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:
            m = json.load(f)
        m["seed"] = int(cfg.get("seed", len(frames)))
        seed_metrics.append(m)

    all_df = pd.concat(frames, ignore_index=True)
    obj_cols = [c for c in all_df.columns if c.startswith("mean_obj_")]
    points = all_df[obj_cols].to_numpy(dtype=np.float64)
    mask = pareto_front_mask(points)
    all_df["global_is_nondominated"] = mask
    all_df.to_csv(out_dir / "all_evaluations.csv", index=False)

    norm, ideal, nadir = normalize_for_max(points)
    metrics = {
        "n_total_solutions": int(len(points)),
        "n_global_nondominated": int(mask.sum()),
        "ideal": ideal.tolist(),
        "nadir": nadir.tolist(),
        "normalized_hv_mc": monte_carlo_hypervolume(norm, samples=args.hv_samples, seed=2026),
        "spacing_normalized": spacing_metric(norm),
        "seed_metrics": seed_metrics,
        "pareto_points": pareto_front(points).tolist(),
    }
    save_json(out_dir / "aggregate_metrics.json", metrics)

    # Agregado por semilla para una tabla compacta.
    summary_rows = []
    for seed, group in all_df.groupby("seed"):
        pts = group[obj_cols].to_numpy(dtype=np.float64)
        norm_seed, _, _ = normalize_for_max(pts, ideal=ideal, nadir=nadir)
        row = {
            "seed": int(seed),
            "n_solutions": int(len(group)),
            "n_nondominated_local": int(pareto_front_mask(pts).sum()),
            "n_nondominated_global": int(group["global_is_nondominated"].sum()),
            "normalized_hv_mc_common_scale": monte_carlo_hypervolume(norm_seed, samples=max(20000, args.hv_samples // 2), seed=int(seed) + 9),
        }
        for c in obj_cols:
            row[f"{c}_mean"] = float(group[c].mean())
            row[f"{c}_max"] = float(group[c].max())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "aggregated_by_seed.csv", index=False)

    plot_3d_front(all_df.rename(columns={c: c for c in obj_cols}), out_dir / "aggregate_front_3d.png")
    plot_pairwise(all_df, out_dir / "aggregate_pairplots.png")
    plot_value_path(all_df, out_dir / "aggregate_value_path.png")

    print("Agregación terminada.")
    print(f"Evaluaciones: {out_dir / 'all_evaluations.csv'}")
    print(f"Métricas: {out_dir / 'aggregate_metrics.json'}")


if __name__ == "__main__":
    main()
