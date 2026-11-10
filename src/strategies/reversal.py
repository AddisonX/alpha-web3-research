"""A model-free policy used to validate strategy compatibility, not profitability."""
import math

import polars as pl

from src.strategies.base import DataRequirements, StrategyContext, StrategyDecision, StrategyMetadata

STRATEGY_ID = "return-1-reversal-long-only-v1"


class ReturnOneReversalStrategy:
    strategy_id = STRATEGY_ID
    requirements = DataRequirements(("return_1",))
    metadata = StrategyMetadata("rule:return_1-reversal-v1", horizon_bars=6)

    def decide(self, context: StrategyContext) -> StrategyDecision:
        frame = context.features.select(
            "symbol", pl.when(pl.col("return_1").is_finite()).then(pl.col("return_1"))
            .otherwise(None).alias("return_1"),
        )
        valid_count = frame["return_1"].len() - frame["return_1"].null_count()
        if not valid_count:
            raise ValueError("no finite return_1 values; keep existing signals frozen")
        count = math.ceil(valid_count * 0.10)
        ordered = frame.sort(["return_1", "symbol"], nulls_last=True).with_row_index("score_position", offset=1)
        return StrategyDecision(context.decision_timestamp, ordered.select(
            pl.col("symbol").alias("pair"),
            ((pl.col("score_position") <= count) & pl.col("return_1").is_not_null()).cast(pl.Int8).alias("target_position"),
            "score_position",
        ))
