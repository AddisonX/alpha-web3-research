from pathlib import Path

from src.strategies.base import DEFAULT_STRATEGY_ID, LongOnlyStrategy


def create_strategy(strategy_id: str, production_root: Path) -> LongOnlyStrategy:
    # Explicit allowlist; configuration cannot execute arbitrary imports.
    if strategy_id == "xgb15-120-long-short-v1":
        from src.strategies.xgb15 import XGB15LongShortStrategy
        return XGB15LongShortStrategy(production_root.parent / "strategies" / strategy_id / "v1")
    if strategy_id == DEFAULT_STRATEGY_ID:
        from src.strategies.lightgbm import LightGBMLongOnlyStrategy

        return LightGBMLongOnlyStrategy(production_root)
    if strategy_id == "return-1-reversal-long-only-v1":
        from src.strategies.reversal import ReturnOneReversalStrategy

        return ReturnOneReversalStrategy()
    raise ValueError(f"unknown strategy: {strategy_id}")
