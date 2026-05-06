from __future__ import annotations

from typing import Tuple

import numpy as np


def dominated_mask(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    mask = np.zeros(len(points), dtype=bool)
    for i, p in enumerate(points):
        others = np.delete(points, i, axis=0)
        if len(others) == 0:
            continue
        mask[i] = bool(np.any(np.all(others >= p, axis=1) & np.any(others > p, axis=1)))
    return mask


def pareto_mask(points: np.ndarray) -> np.ndarray:
    return ~dominated_mask(points)


def pareto_front(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return points
    return points[pareto_mask(points)]


def crowding_distance(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n <= 2:
        return np.ones(n, dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    denom = np.where(maxs - mins == 0.0, 1.0, maxs - mins)
    norm = (points - mins) / denom
    dist = np.zeros(n, dtype=np.float64)
    for obj in range(points.shape[1]):
        order = np.argsort(norm[:, obj])
        dist[order[0]] += 1.0
        dist[order[-1]] += 1.0
        for j in range(1, n - 1):
            dist[order[j]] += norm[order[j + 1], obj] - norm[order[j - 1], obj]
    return dist / max(points.shape[1], 1)


def pruning_scores(points: np.ndarray, crowding_threshold: float = 0.2, crowding_penalty: float = 0.01) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.array([], dtype=np.float64)
    front = pareto_front(points)
    if len(front) == 0:
        return np.zeros(len(points), dtype=np.float64)
    distances = np.zeros(len(points), dtype=np.float64)
    for i, p in enumerate(points):
        distances[i] = np.min(np.linalg.norm(front - p, axis=1))
    il2 = -distances
    cd = crowding_distance(points)
    scores = il2.copy()
    crowded = cd <= crowding_threshold
    scores[crowded] = 2.0 * (il2[crowded] - crowding_penalty)
    return scores


def hypervolume_2d(points: np.ndarray, reference: np.ndarray | None = None) -> Tuple[float, np.ndarray]:
    pts = pareto_front(np.asarray(points, dtype=np.float64))
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("hypervolume_2d requiere puntos de dimensión 2.")
    if len(pts) == 0:
        ref = np.zeros(2, dtype=np.float64) if reference is None else np.asarray(reference, dtype=np.float64)
        return 0.0, ref
    if reference is None:
        margin = 0.05 * np.maximum(np.ptp(pts, axis=0), 1.0)
        reference = pts.min(axis=0) - margin
    reference = np.asarray(reference, dtype=np.float64)
    pts = pts[np.all(pts > reference, axis=1)]
    if len(pts) == 0:
        return 0.0, reference
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    previous_x = reference[0]
    for x, y in pts:
        width = max(x - previous_x, 0.0)
        height = max(y - reference[1], 0.0)
        hv += width * height
        previous_x = max(previous_x, x)
    return float(hv), reference


def normalize_points(points: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    ideal = np.asarray(ideal, dtype=np.float64)
    nadir = np.asarray(nadir, dtype=np.float64)
    denom = np.where(ideal - nadir == 0.0, 1.0, ideal - nadir)
    return (points - nadir) / denom
