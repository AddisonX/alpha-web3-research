"""Runnable contract example, not a researched or production-ready strategy."""

import math

import polars as pl

from src.strategies.base import (
    DataRequirements,
    StrategyContext,
    StrategyDecision,
    StrategyMetadata,
)

STRATEGY_ID = "template-momentum-long-only-v1"


class ImportedLongOnlyStrategy:
    strategy_id = STRATEGY_ID
    requirements = DataRequirements(
        features=("return_1",),
        timeframe="4h",
        lookback_bars=40,
        min_closed_bars=25,
    )
    metadata = StrategyMetadata(
        source_experiment="template:momentum-contract-example-v1",
        model_count=0,
        horizon_bars=6,
        label_column="",
    )
    top_fraction = 0.10
    max_selected = 9

    def decide(self, context: StrategyContext) -> StrategyDecision:
        features = context.features
        required = {"symbol", *self.requirements.features}
        if not required.issubset(features.columns):
            raise ValueError("missing required feature columns")
        if features.is_empty():
            raise ValueError("empty feature matrix")
        symbols = features["symbol"].to_list()
        if (
            any(not isinstance(symbol, str) or not symbol for symbol in symbols)
            or len(set(symbols)) != len(symbols)
        ):
            raise ValueError("symbols must be nonempty unique strings")
        dtype = features.schema["return_1"]
        if dtype != pl.Null and not dtype.is_numeric():
            raise ValueError("required features must be numeric")

        # Replace this scoring block with the new policy, preserving the contract.
        scored = features.select(
            "symbol",
            pl.col("return_1").cast(pl.Float64).alias("score"),
        ).with_columns(
            pl.when(pl.col("score").is_finite())
            .then(pl.col("score"))
            .otherwise(None)
            .alias("score"),
        )
        valid_count = scored["score"].len() - scored["score"].null_count()
        if valid_count == 0:
            raise ValueError("no finite scores; do not publish replacement targets")
        selected_count = min(self.max_selected, math.ceil(valid_count * self.top_fraction))
        ordered = scored.sort(
            ["score", "symbol"], descending=[True, False], nulls_last=True,
        ).with_row_index("score_position", offset=1)
        targets = ordered.select(
            pl.col("symbol").alias("pair"),
            (
                (pl.col("score_position") <= selected_count)
                & pl.col("score").is_not_null()
            ).cast(pl.Int8).alias("target_position"),
            "score_position",
        )
        decision = StrategyDecision(context.decision_timestamp, targets)
        decision.validate(context)
        return decision
