from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence

import numpy as np


@dataclass
class Individual:
    genome: np.ndarray
    fitness: np.ndarray | None = None
    episode_length: float = 0.0
    rank: int = -1
    ref_index: int = -1
    ref_distance: float = np.inf


def uniform_reference_points(n_obj: int, p: int) -> np.ndarray:
    points = []

    def rec(left: int, total: int, prefix: list[int]):
        if len(prefix) == n_obj - 1:
            points.append(prefix + [left])
            return
        for v in range(left + 1):
            rec(left - v, total, prefix + [v])

    rec(p, p, [])
    return np.asarray(points, dtype=np.float64) / float(p)


def dominates_min(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(objectives_min: np.ndarray) -> List[List[int]]:
    n = len(objectives_min)
    domination_count = np.zeros(n, dtype=np.int64)
    dominated_sets = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates_min(objectives_min[p], objectives_min[q]):
                dominated_sets[p].append(q)
            elif dominates_min(objectives_min[q], objectives_min[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_sets[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]


def normalize_objectives(objectives_min: np.ndarray) -> np.ndarray:
    z_min = objectives_min.min(axis=0)
    translated = objectives_min - z_min
    n_obj = objectives_min.shape[1]

    extreme_indices = []
    eps = 1e-6
    for j in range(n_obj):
        w = np.full(n_obj, eps)
        w[j] = 1.0
        asf = np.max(translated / w, axis=1)
        extreme_indices.append(int(np.argmin(asf)))

    extreme_points = translated[extreme_indices]
    try:
        b = np.ones(n_obj)
        plane = np.linalg.solve(extreme_points, b)
        intercepts = 1.0 / plane
        if np.any(~np.isfinite(intercepts)) or np.any(intercepts <= 1e-12):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts = translated.max(axis=0)

    intercepts = np.where(intercepts <= 1e-12, 1.0, intercepts)
    return translated / intercepts


def associate_to_reference_points(normalized: np.ndarray, ref_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref_norm = np.linalg.norm(ref_points, axis=1)
    ref_norm = np.where(ref_norm <= 1e-12, 1.0, ref_norm)
    distances = []
    for z in ref_points:
        w = z / max(np.linalg.norm(z), 1e-12)
        proj = normalized @ w
        perpendicular = normalized - np.outer(proj, w)
        distances.append(np.linalg.norm(perpendicular, axis=1))
    distances = np.asarray(distances).T
    pi = np.argmin(distances, axis=1)
    d = distances[np.arange(len(normalized)), pi]
    return pi.astype(int), d.astype(float)


def nsga3_select(
    population: Sequence[Individual],
    population_size: int,
    ref_points: np.ndarray,
    rng: np.random.Generator,
) -> list[Individual]:
    fitness = np.asarray([ind.fitness for ind in population], dtype=np.float64)
    objectives_min = -fitness
    fronts = non_dominated_sort(objectives_min)

    selected_indices: list[int] = []
    last_front: list[int] | None = None
    rank_by_index = {}

    for rank, front in enumerate(fronts):
        for idx in front:
            rank_by_index[idx] = rank
        if len(selected_indices) + len(front) <= population_size:
            selected_indices.extend(front)
        else:
            last_front = front
            break

    for idx, rank in rank_by_index.items():
        population[idx].rank = rank

    if last_front is None or len(selected_indices) == population_size:
        out = [population[i] for i in selected_indices[:population_size]]
        return out

    st_indices = selected_indices + last_front
    st_objectives = objectives_min[st_indices]
    normalized = normalize_objectives(st_objectives)
    assoc, dist = associate_to_reference_points(normalized, ref_points)

    local_pos = {idx: pos for pos, idx in enumerate(st_indices)}
    for idx in st_indices:
        pos = local_pos[idx]
        population[idx].ref_index = int(assoc[pos])
        population[idx].ref_distance = float(dist[pos])

    selected_set = set(selected_indices)
    last_set = set(last_front)
    niche_count = np.zeros(len(ref_points), dtype=np.int64)
    for idx in selected_indices:
        niche_count[population[idx].ref_index] += 1

    remaining = population_size - len(selected_indices)
    chosen: list[int] = []
    active_refs = set(range(len(ref_points)))

    while remaining > 0 and last_set and active_refs:
        min_count = min(niche_count[j] for j in active_refs)
        candidates_refs = [j for j in active_refs if niche_count[j] == min_count]
        ref_j = int(rng.choice(candidates_refs))
        candidates = [idx for idx in last_set if population[idx].ref_index == ref_j]
        if not candidates:
            active_refs.remove(ref_j)
            continue
        if niche_count[ref_j] == 0:
            chosen_idx = min(candidates, key=lambda idx: population[idx].ref_distance)
        else:
            chosen_idx = int(rng.choice(candidates))
        chosen.append(chosen_idx)
        last_set.remove(chosen_idx)
        niche_count[ref_j] += 1
        remaining -= 1

    if remaining > 0 and last_set:
        leftovers = list(last_set)
        rng.shuffle(leftovers)
        chosen.extend(leftovers[:remaining])

    return [population[i] for i in selected_indices + chosen]
