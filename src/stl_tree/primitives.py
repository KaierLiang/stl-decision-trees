"""Tree indexing and inclusive temporal-window statistics."""

import numpy as np


def generate_nodes(depth):
    total = 2 ** (depth + 1) - 1
    first_leaf = 2**depth - 1
    return total, list(range(first_leaf)), list(range(first_leaf, total))


def generate_windows(horizon, max_span=500):
    """All inclusive [start, end] with end - start <= max_span, including points."""
    if horizon < 1 or max_span < 0:
        raise ValueError("horizon must be positive and max_span nonnegative")
    return [
        (start, end)
        for start in range(horizon)
        for end in range(start, min(horizon, start + max_span + 1))
    ]


def window_extrema(signals, windows):
    """Cache min/max coefficients once, without removing or coarsening windows."""
    minima = np.stack([signals[:, :, a : b + 1].min(axis=2) for a, b in windows], axis=2)
    maxima = np.stack([signals[:, :, a : b + 1].max(axis=2) for a, b in windows], axis=2)
    return minima, maxima
