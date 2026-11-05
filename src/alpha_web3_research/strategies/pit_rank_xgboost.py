"""Preserve the daily equal-weight policy while isolating the XGBoost experiment."""
from alpha_web3_research.strategies.pit_rank_excess import PITRankExcessLongOnlyStrategy

STRATEGY_ID = "pit_rank64_xgboost_daily_excess_long_only"
STRATEGY_KEY = f"{STRATEGY_ID}@v1"


class PITRankXGBoostLongOnlyStrategy(PITRankExcessLongOnlyStrategy):
    strategy_key = STRATEGY_KEY

    def __init__(self, fraction: float = 0.10) -> None:
        super().__init__(fraction=fraction, stop_loss=None)
