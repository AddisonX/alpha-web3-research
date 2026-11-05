"""Daily equal-weight selection for the independent PIT rank-feature strategy."""
from __future__ import annotations

import math

import numpy as np

STRATEGY_ID = "pit_rank64_daily_excess_long_only"
STRATEGY_KEY = f"{STRATEGY_ID}@v1"


class PITRankExcessLongOnlyStrategy:
    strategy_key = STRATEGY_KEY

    def __init__(self, fraction: float = 0.10, stop_loss: float | None = 0.10) -> None:
        if not 0 < fraction <= 1 or (stop_loss is not None and not 0 < stop_loss < 1):
            raise ValueError("fraction must be in (0,1], stop_loss must be None or in (0,1)")
        self.fraction = fraction
        self.stop_loss = stop_loss

    def select_weights(self, scores: np.ndarray, symbols: list[str]) -> np.ndarray:
        if len(scores) != len(symbols) or len(set(symbols)) != len(symbols):
            raise ValueError("scores must match unique symbol identifiers")
        eligible = [index for index, score in enumerate(scores) if np.isfinite(score)]
        weights = np.zeros(len(symbols), dtype=float)
        if not eligible:
            return weights
        count = max(1, math.ceil(len(eligible) * self.fraction))
        chosen = sorted(eligible, key=lambda index: (-float(scores[index]), symbols[index]))[:count]
        weights[chosen] = 1.0 / len(chosen)
        return weights
