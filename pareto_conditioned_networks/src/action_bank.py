from __future__ import annotations

import numpy as np


def build_action_bank(action_low: np.ndarray, action_high: np.ndarray, bank_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    action_dim = low.shape[0]

    actions = []
    actions.append(np.zeros(action_dim, dtype=np.float32))

    for dim in range(action_dim):
        a_pos = np.zeros(action_dim, dtype=np.float32)
        a_neg = np.zeros(action_dim, dtype=np.float32)
        a_pos[dim] = high[dim]
        a_neg[dim] = low[dim]
        actions.extend([a_pos, a_neg])

    remaining = max(0, bank_size - len(actions))
    if remaining > 0:
        random_actions = rng.uniform(low=low, high=high, size=(remaining, action_dim)).astype(np.float32)
        actions.extend(list(random_actions))

    bank = np.stack(actions[:bank_size], axis=0).astype(np.float32)
    return np.clip(bank, low, high)


def action_from_index(action_bank: np.ndarray, index: int) -> np.ndarray:
    return np.asarray(action_bank[int(index)], dtype=np.float32)
