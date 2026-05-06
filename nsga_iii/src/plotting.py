from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import pareto_front, pareto_front_mask


def plot_3d_front(df: pd.DataFrame, out_path: str | Path) -> None:
    obj_cols = [c for c in df.columns if c.startswith("mean_obj_")]
    if len(obj_cols) < 3:
        return
    pts = df[obj_cols[:3]].to_numpy(dtype=float)
    mask = pareto_front_mask(pts)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.55, label="Población evaluada")
    ax.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2], s=70, label="No dominadas")
    ax.set_xlabel("Objetivo 0")
    ax.set_ylabel("Objetivo 1")
    ax.set_zlabel("Objetivo 2")
    ax.set_title("Frente aproximado NSGA-III")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def plot_pairwise(df: pd.DataFrame, out_path: str | Path) -> None:
    obj_cols = [c for c in df.columns if c.startswith("mean_obj_")]
    if len(obj_cols) < 2:
        return
    pts = df[obj_cols].to_numpy(dtype=float)
    mask = pareto_front_mask(pts)
    pairs = [(0, 1)] if len(obj_cols) == 2 else [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(pts[:, i], pts[:, j], alpha=0.55, label="Evaluadas")
        ax.scatter(pts[mask, i], pts[mask, j], s=65, label="No dominadas")
        ax.set_xlabel(f"Objetivo {i}")
        ax.set_ylabel(f"Objetivo {j}")
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def plot_value_path(df: pd.DataFrame, out_path: str | Path) -> None:
    obj_cols = [c for c in df.columns if c.startswith("mean_obj_")]
    if len(obj_cols) < 2:
        return
    pts = df[obj_cols].to_numpy(dtype=float)
    front = pareto_front(pts)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    denom = np.where(np.abs(maxs - mins) < 1e-12, 1.0, maxs - mins)
    norm_front = (front - mins) / denom

    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(obj_cols))
    for row in norm_front:
        plt.plot(x, row, marker="o", alpha=0.75)
    plt.xticks(x, [f"Obj. {i}" for i in range(len(obj_cols))])
    plt.ylim(-0.05, 1.05)
    plt.ylabel("Valor normalizado")
    plt.title("Value path de soluciones no dominadas")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
