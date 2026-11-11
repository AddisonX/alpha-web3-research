from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import math

import polars as pl

DEFAULT_STRATEGY_ID = "tokyo-lightgbm-long-only-v1"


@dataclass(frozen=True)
class DataRequirements:
    features: tuple[str, ...]
    timeframe: str = "4h"
    lookback_bars: int = 40
    min_closed_bars: int = 25

    def __post_init__(self) -> None:
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("features must be nonempty and unique")
        if self.lookback_bars < self.min_closed_bars or self.min_closed_bars < 1:
            raise ValueError("invalid lookback/minimum bar requirements")


@dataclass(frozen=True)
class StrategyMetadata:
    source_experiment: str
    model_count: int = 0
    horizon_bars: int = 0
    label_column: str = ""


@dataclass(frozen=True)
class StrategyContext:
    decision_timestamp: int
    features: pl.DataFrame
    allow_short: bool = False
    previous_targets: pl.DataFrame | None = None


@dataclass(frozen=True)
class StrategyDecision:
    """Complete targets; signed weights require explicit opt-in on both sides."""

    decision_timestamp: int
    targets: pl.DataFrame
    allow_short: bool = False

    def validate(self, context: StrategyContext) -> None:
        if self.decision_timestamp != context.decision_timestamp:
            raise ValueError("strategy decision timestamp does not match context")
        if not {"pair", "target_position"}.issubset(self.targets.columns):
            raise ValueError("decision requires pair and target_position")
        pairs = self.targets["pair"].to_list()
        expected = context.features["symbol"].to_list()
        if (not pairs or any(not isinstance(pair, str) or not pair for pair in pairs)
                or len(expected) != len(set(expected))
                or len(pairs) != len(set(pairs)) or set(pairs) != set(expected)):
            raise ValueError("decision must contain each context symbol exactly once")
        targets = self.targets["target_position"].to_list()
        if self.allow_short:
            if not context.allow_short:
                raise ValueError("short targets require an explicitly enabled strategy")
            validate_signed_weights(self.targets)
            return
        if any(value is None or value not in (0, 1) for value in targets):
            raise ValueError("only long-only binary targets are supported")
        if not any(targets):
            raise ValueError("the existing safety adapter requires at least one selected pair")


class LongOnlyStrategy(Protocol):
    strategy_id: str
    requirements: DataRequirements
    metadata: StrategyMetadata

    def decide(self, context: StrategyContext) -> StrategyDecision: ...


def validate_signed_weights(targets: pl.DataFrame) -> None:
    """Signed, unlevered, equal-weight 50/50 targets; zeros cover unselected coins."""
    if "target_weight" not in targets.columns:
        raise ValueError("short targets require target_weight")
    positions = targets["target_position"].to_list()
    weights = targets["target_weight"].to_list()
    if any(p not in (-1, 0, 1) for p in positions):
        raise ValueError("invalid signed target")
    if any(w is None or not math.isfinite(w) or (1 if w > 0 else -1 if w < 0 else 0) != p
           for p, w in zip(positions, weights)):
        raise ValueError("invalid target weight or direction")
    for sign in (-1, 1):
        side = [abs(w) for p, w in zip(positions, weights) if p == sign]
        if not side or not math.isclose(sum(side), .5, abs_tol=1e-9):
            raise ValueError("each side must sum to 50 percent")
        if any(not math.isclose(w, .5 / len(side), abs_tol=1e-9) for w in side):
            raise ValueError("each side must be equal weighted")
