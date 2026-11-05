from __future__ import annotations

import math

import numpy as np


class DailyCrossSectionalQuantileStrategy:
    """Preserved policy used by the original 4h/h6 Freqtrade strategies."""

    def __init__(self, long_only: bool = False, fraction: float = 0.10) -> None:
        if not 0 < fraction < 0.5:
            raise ValueError("fraction must be in (0, 0.5)")
        self.long_only = long_only
        self.fraction = fraction
        suffix = "long_only" if long_only else "long_short"
        self.strategy_key = f"cross_sectional_4h_h6_daily_{suffix}@v1"

    def target_positions(self, scores: np.ndarray, symbols: list[str]) -> np.ndarray:
        if len(scores) != len(symbols):
            raise ValueError("scores and symbols must have equal length")
        targets = np.zeros(len(scores), dtype=np.int8)
        valid = np.flatnonzero(np.isfinite(scores))
        if not valid.size:
            return targets
        count = max(1, math.ceil(valid.size * self.fraction))
        ordered = sorted(valid, key=lambda index: (-float(scores[index]), symbols[index]))
        targets[ordered[:count]] = 1
        if not self.long_only:
            targets[ordered[-count:]] = -1
        return targets
