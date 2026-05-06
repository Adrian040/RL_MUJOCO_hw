from __future__ import annotations

import numpy as np


def pareto_front_mask(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        others = np.delete(points, i, axis=0)
        if len(others) == 0:
            continue
        dominated = np.any(np.all(others >= p, axis=1) & np.any(others > p, axis=1))
        if dominated:
            keep[i] = False
    return keep


def pareto_front(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64)[pareto_front_mask(points)]


def normalize_for_max(points: np.ndarray, ideal: np.ndarray | None = None, nadir: np.ndarray | None = None):
    points = np.asarray(points, dtype=np.float64)
    if ideal is None:
        ideal = points.max(axis=0)
    if nadir is None:
        nadir = points.min(axis=0)
    denom = np.where(np.abs(ideal - nadir) < 1e-12, 1.0, ideal - nadir)
    norm = (points - nadir) / denom
    return np.clip(norm, 0.0, 1.0), ideal, nadir


def monte_carlo_hypervolume(points: np.ndarray, reference: np.ndarray | None = None, samples: int = 50000, seed: int = 0) -> float:
    points = pareto_front(np.asarray(points, dtype=np.float64))
    if reference is None:
        reference = np.full(points.shape[1], -0.05, dtype=np.float64)
    upper = np.ones(points.shape[1], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.uniform(reference, upper, size=(samples, points.shape[1]))
    dominated = np.zeros(samples, dtype=bool)
    for p in points:
        dominated |= np.all(sample <= p, axis=1)
    volume_box = np.prod(upper - reference)
    return float(volume_box * dominated.mean())


def spacing_metric(points: np.ndarray) -> float:
    points = pareto_front(np.asarray(points, dtype=np.float64))
    if len(points) <= 2:
        return 0.0
    dists = []
    for i, p in enumerate(points):
        others = np.delete(points, i, axis=0)
        dists.append(float(np.min(np.linalg.norm(others - p, axis=1))))
    dists = np.asarray(dists)
    return float(np.sqrt(np.mean((dists - dists.mean()) ** 2)))
