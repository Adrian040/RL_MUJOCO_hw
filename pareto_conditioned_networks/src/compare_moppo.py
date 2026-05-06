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
    parser = argparse.ArgumentParser(description="Compara PCN contra una corrida de MOPPO.")
    parser.add_argument("--pcn-run", type=str, required=True)
    parser.add_argument("--moppo-run", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    pcn_run = Path(args.pcn_run)
    moppo_run = Path(args.moppo_run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pcn_points = read_points(pcn_run / "evaluation_summary.csv")
    moppo_points = read_points(moppo_run / "evaluation_summary.csv")

    all_points = np.vstack([pcn_points, moppo_points])
    ideal = all_points.max(axis=0)
    nadir = all_points.min(axis=0)
    ref_norm = np.array([-0.05, -0.05], dtype=np.float64)

    pcn_norm = normalize_points(pcn_points, ideal, nadir)
    moppo_norm = normalize_points(moppo_points, ideal, nadir)
    hv_pcn, _ = hypervolume_2d(pcn_norm, reference=ref_norm)
    hv_moppo, _ = hypervolume_2d(moppo_norm, reference=ref_norm)

    pcn_front = pareto_front(pcn_points)
    moppo_front = pareto_front(moppo_points)

    comparison = pd.DataFrame({
        "Método": ["PCN", "MOPPO"],
        "No. soluciones evaluadas": [len(pcn_points), len(moppo_points)],
        "No. soluciones no dominadas": [len(pcn_front), len(moppo_front)],
        "Mejor objetivo 0": [pcn_points[:, 0].max(), moppo_points[:, 0].max()],
        "Mejor objetivo 1": [pcn_points[:, 1].max(), moppo_points[:, 1].max()],
        "HV normalizado": [hv_pcn, hv_moppo],
    })
    comparison.to_csv(out_dir / "comparison_pcn_vs_moppo.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(pcn_points[:, 0], pcn_points[:, 1], alpha=0.6, s=70, label="PCN")
    plt.scatter(moppo_points[:, 0], moppo_points[:, 1], alpha=0.6, s=70, label="MOPPO")
    if len(pcn_front) > 0:
        f = pcn_front[np.argsort(pcn_front[:, 0])]
        plt.plot(f[:, 0], f[:, 1], marker="o", linewidth=2, label="PCN no dominado")
    if len(moppo_front) > 0:
        f = moppo_front[np.argsort(moppo_front[:, 0])]
        plt.plot(f[:, 0], f[:, 1], marker="o", linewidth=2, label="MOPPO no dominado")
    plt.xlabel("Objetivo 0")
    plt.ylabel("Objetivo 1")
    plt.title("Comparación PCN vs MOPPO")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_pcn_vs_moppo.png", dpi=300)
    plt.close()

    save_json(out_dir / "comparison_metrics.json", {
        "ideal": ideal.tolist(),
        "nadir": nadir.tolist(),
        "reference_normalized": ref_norm.tolist(),
        "hypervolume_pcn_normalized": float(hv_pcn),
        "hypervolume_moppo_normalized": float(hv_moppo),
    })
    print(comparison.round(3).to_string(index=False))
    print(f"Archivos guardados en: {out_dir}")


if __name__ == "__main__":
    main()
