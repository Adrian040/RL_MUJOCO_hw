from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_comparison(env_dir: Path) -> pd.DataFrame | None:
    path = env_dir / "comparison_morl_baselines" / "comparison_local_vs_morl_baselines.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.insert(0, "Ambiente", env_dir.name)
    return df


def plot_hv(df: pd.DataFrame, out_path: Path) -> None:
    if "HV normalizado" not in df.columns:
        return
    pivot = df.pivot(index="Ambiente", columns="Método", values="HV normalizado")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("HV normalizado")
    ax.set_title("Comparación multi-ambiente")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Agrega comparaciones por ambiente.")
    parser.add_argument("--root-dir", type=str, required=True)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    frames = []
    for env_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        df = load_comparison(env_dir)
        if df is not None:
            frames.append(df)
    if not frames:
        print("No se encontraron comparaciones para agregar.")
        return
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(root_dir / "multi_env_comparison_summary.csv", index=False)
    plot_hv(out, root_dir / "multi_env_hv_summary.png")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
