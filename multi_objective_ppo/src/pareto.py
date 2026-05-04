from __future__ import annotations

from typing import Tuple

import numpy as np


def is_dominated(point: np.ndarray, others: np.ndarray) -> bool:
    """Indica si un punto está dominado en maximización."""
    return bool(np.any(np.all(others >= point, axis=1) & np.any(others > point, axis=1)))


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Devuelve los puntos no dominados para un problema de maximización."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return points
    mask = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if is_dominated(p, np.delete(points, i, axis=0)):
            mask[i] = False
    return points[mask]


def hypervolume_2d(points: np.ndarray, reference: np.ndarray | None = None) -> Tuple[float, np.ndarray]:
    """Calcula hipervolumen 2D aproximado para maximización.

    Si no se proporciona punto de referencia, se usa un punto ligeramente peor
    que el mínimo observado en cada objetivo. El resultado es útil para comparar
    corridas del mismo experimento, no como valor absoluto universal.
    """
    pts = pareto_front(np.asarray(points, dtype=np.float64))
    if pts.shape[1] != 2:
        raise ValueError("hypervolume_2d solo acepta puntos de dimensión 2.")
    if reference is None:
        margin = 0.05 * np.maximum(np.ptp(pts, axis=0), 1.0)
        reference = pts.min(axis=0) - margin
    reference = np.asarray(reference, dtype=np.float64)

    pts = pts[np.all(pts > reference, axis=1)]
    if len(pts) == 0:
        return 0.0, reference

    # Para maximización en 2D, ordenamos por objetivo 0 ascendente.
    # En el frente no dominado, el objetivo 1 queda de forma descendente.
    # El área se suma en franjas: (x_i - x_{i-1}) * (y_i - y_ref).
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    previous_x = reference[0]
    for x, y in pts:
        width = max(x - previous_x, 0.0)
        height = max(y - reference[1], 0.0)
        hv += width * height
        previous_x = max(previous_x, x)
    return float(hv), reference
