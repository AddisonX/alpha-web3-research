from __future__ import annotations
from datetime import datetime, timezone
import polars as pl
from src.config import Settings
FACTOR_ID_COLUMNS = ["exchange", "market_type", "symbol", "timeframe", "timestamp"]
GROUP_COLUMNS = ["exchange", "market_type", "symbol", "timeframe"]
class FactorBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _build_factor_long_frame(self, data: pl.DataFrame) -> pl.DataFrame:
        generated_at = datetime.now(timezone.utc).isoformat()
        base = (
            data.sort(GROUP_COLUMNS + ["timestamp"])
            .with_columns(
                (pl.col("close") / pl.col("close").shift(1).over(GROUP_COLUMNS) - 1.0).alias("ret_1"),
                ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("candle_body"),
                ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("hl_range_1"),
                ((pl.col("high") - pl.max_horizontal("open", "close")) / pl.col("close")).alias("upper_shadow"),
                ((pl.min_horizontal("open", "close") - pl.col("low")) / pl.col("close")).alias("lower_shadow"),
                (pl.col("close") * pl.col("volume")).alias("quote_volume"),
            )
        )

        factor_exprs: list[pl.Expr] = [
            pl.col("ret_1").alias("return_1"),
            pl.col("ret_1").log1p().alias("log_return_1"),
            pl.col("candle_body").alias("candle_body_1"),
            pl.col("hl_range_1").alias("range_1"),
            pl.col("upper_shadow").alias("upper_shadow_1"),
            pl.col("lower_shadow").alias("lower_shadow_1"),
        ]

        for window in [2, 3, 4, 6, 8, 12, 24]:
            factor_exprs.extend(
                [
                    (pl.col("close") / pl.col("close").shift(window).over(GROUP_COLUMNS) - 1.0).alias(f"return_{window}"),
                    pl.col("ret_1").rolling_mean(window_size=window, min_samples=2).over(GROUP_COLUMNS).alias(f"mean_return_{window}"),
                    pl.col("ret_1").rolling_std(window_size=window, min_samples=2).over(GROUP_COLUMNS).alias(f"volatility_{window}"),
                    (pl.col("close") / pl.col("close").rolling_mean(window_size=window, min_samples=2).over(GROUP_COLUMNS) - 1.0).alias(f"ma_gap_{window}"),
                    (pl.col("close") / pl.col("close").rolling_max(window_size=window, min_samples=2).over(GROUP_COLUMNS) - 1.0).alias(f"distance_to_high_{window}"),
                    (pl.col("close") / pl.col("close").rolling_min(window_size=window, min_samples=2).over(GROUP_COLUMNS) - 1.0).alias(f"distance_to_low_{window}"),
                    pl.col("hl_range_1").rolling_mean(window_size=window, min_samples=2).over(GROUP_COLUMNS).alias(f"avg_range_{window}"),
                    (pl.col("volume") / pl.col("volume").shift(window).over(GROUP_COLUMNS) - 1.0).alias(f"volume_change_{window}"),
                    zscore_expr("volume", window).alias(f"volume_zscore_{window}"),
                    zscore_expr("quote_volume", window).alias(f"quote_volume_zscore_{window}"),
                    (pl.col("hl_range_1") / (pl.col("quote_volume").rolling_mean(window_size=window, min_samples=2).over(GROUP_COLUMNS) + 1.0)).alias(f"range_to_quote_volume_{window}"),
                ]
            )

        wide = base.with_columns(factor_exprs)
        factor_columns = [column for column in wide.columns if column not in FACTOR_ID_COLUMNS and column not in {"open", "high", "low", "close", "volume", "ret_1", "candle_body", "hl_range_1", "upper_shadow", "lower_shadow", "quote_volume", "fetched_at", "source"}]
        if not factor_columns:
            return self._empty_long_frame()

        return (
            wide.select(FACTOR_ID_COLUMNS + factor_columns)
            .unpivot(index=FACTOR_ID_COLUMNS, on=factor_columns, variable_name="factor_name", value_name="factor_value")
            .with_columns(
                pl.lit(generated_at).alias("generated_at"),
                pl.col("factor_name").map_elements(factor_family, return_dtype=pl.String).alias("factor_family"),
            )
            .select(["generated_at", *FACTOR_ID_COLUMNS, "factor_family", "factor_name", "factor_value"])
            .sort(["timeframe", "factor_name", "timestamp", "symbol"])
        )


    @staticmethod
    def _empty_long_frame() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "generated_at": pl.String,
                "exchange": pl.String,
                "market_type": pl.String,
                "symbol": pl.String,
                "timeframe": pl.String,
                "timestamp": pl.Int64,
                "factor_family": pl.String,
                "factor_name": pl.String,
                "factor_value": pl.Float64,
            }
        )


def zscore_expr(column: str, window: int) -> pl.Expr:
    mean = pl.col(column).rolling_mean(window_size=window, min_samples=2).over(GROUP_COLUMNS)
    std = pl.col(column).rolling_std(window_size=window, min_samples=2).over(GROUP_COLUMNS)
    return pl.when(std > 0).then((pl.col(column) - mean) / std).otherwise(None)


def factor_family(name: str) -> str:
    if name.startswith(("return_", "log_return", "mean_return", "ma_gap", "distance_to")):
        return "momentum_reversal"
    if name.startswith("range_to_quote_volume"):
        return "liquidity_proxy"
    if name.startswith(("volatility_", "range_", "avg_range", "candle", "upper", "lower")):
        return "volatility_range"
    if name.startswith(("volume_", "quote_volume")):
        return "volume"
    return "other"

