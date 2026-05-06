from __future__ import annotations

import numpy as np


def sbx_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator,
    low: float,
    high: float,
    eta_c: float,
    prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() > prob:
        return parent1.copy(), parent2.copy()

    x1 = parent1.astype(np.float64).copy()
    x2 = parent2.astype(np.float64).copy()
    c1 = x1.copy()
    c2 = x2.copy()

    for i in range(len(x1)):
        if rng.random() <= 0.5 and abs(x1[i] - x2[i]) > 1e-14:
            y1, y2 = sorted((x1[i], x2[i]))
            rand = rng.random()
            beta = 1.0 + (2.0 * (y1 - low) / (y2 - y1))
            alpha = 2.0 - beta ** (-(eta_c + 1.0))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
            child1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

            beta = 1.0 + (2.0 * (high - y2) / (y2 - y1))
            alpha = 2.0 - beta ** (-(eta_c + 1.0))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
            child2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

            child1 = np.clip(child1, low, high)
            child2 = np.clip(child2, low, high)
            if rng.random() <= 0.5:
                c1[i], c2[i] = child2, child1
            else:
                c1[i], c2[i] = child1, child2
    return c1.astype(np.float32), c2.astype(np.float32)


def polynomial_mutation(
    genome: np.ndarray,
    rng: np.random.Generator,
    low: float,
    high: float,
    eta_m: float,
    prob: float,
) -> np.ndarray:
    y = genome.astype(np.float64).copy()
    span = high - low
    if span <= 0:
        return genome.copy()
    for i in range(len(y)):
        if rng.random() <= prob:
            delta1 = (y[i] - low) / span
            delta2 = (high - y[i]) / span
            rand = rng.random()
            mut_pow = 1.0 / (eta_m + 1.0)
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                deltaq = 1.0 - val ** mut_pow
            y[i] = np.clip(y[i] + deltaq * span, low, high)
    return y.astype(np.float32)
