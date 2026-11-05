from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from alpha_web3_research.config import Settings
from alpha_web3_research.data.storage import ParquetOHLCVStore
from alpha_web3_research.data.contracts import BarQuery
from alpha_web3_research.data.reader import MarketDataReader
from alpha_web3_research.universe.builder import UniverseBuilder

HOUR_MS = 60 * 60 * 1_000
ASSET_COLUMNS = ["exchange", "market_type", "symbol"]
FACTOR_WINDOWS = [2, 3, 4, 6, 8, 12, 24]


@dataclass(frozen=True)
class HourlyFeatureBuildResult:
    rows: int
    feature_count: int
    symbol_count: int
    path: Path


class HourlyMultiTimeframeFactorBuilder:
    """Build hourly as-of features from 1h candles and rolling 4h candles."""

    def __init__(self, settings: Settings, store: ParquetOHLCVStore | None = None) -> None:
        self.settings = settings
        self.store = store or ParquetOHLCVStore(settings=settings)
        self.output_path = settings.resolved_data_path() / "factors" / "hourly_multitimeframe.parquet"

    def build(self, universe_only: bool = True, write: bool = True) -> tuple[pl.DataFrame, HourlyFeatureBuildResult]:
        one_hour = MarketDataReader(self.store).read_bars(
            BarQuery(self.settings.exchange_id, self.settings.market_type, "1h")
        )
        if universe_only and not one_hour.is_empty():
            one_hour = self._filter_to_latest_universe(one_hour)
        if one_hour.is_empty():
            features = pl.DataFrame()
        else:
            one_hour = self._with_decision_timestamp(one_hour)
            one_hour_features = build_factor_wide(
                one_hour,
                group_columns=[*ASSET_COLUMNS, "segment_id"],
                prefix="1h__",
            ).drop("segment_id")
            rolling_four_hour = build_rolling_four_hour_candles(one_hour)
            four_hour_features = build_factor_wide(
                rolling_four_hour,
                group_columns=[*ASSET_COLUMNS, "segment_id", "four_hour_phase"],
                prefix="4h__",
            ).drop(["segment_id", "four_hour_phase"])
            features = (
                one_hour_features.join(
                    four_hour_features,
                    on=[*ASSET_COLUMNS, "decision_timestamp"],
                    how="inner",
                    validate="1:1",
                )
                .with_columns(pl.lit(datetime.now(timezone.utc).isoformat()).alias("generated_at"))
                .sort(["decision_timestamp", "symbol"])
            )

        feature_columns = [column for column in features.columns if column.startswith(("1h__", "4h__"))]
        if write:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            features.write_parquet(self.output_path)
        result = HourlyFeatureBuildResult(
            rows=features.height,
            feature_count=len(feature_columns),
            symbol_count=features.get_column("symbol").n_unique() if "symbol" in features.columns else 0,
            path=self.output_path,
        )
        return features, result

    def _filter_to_latest_universe(self, frame: pl.DataFrame) -> pl.DataFrame:
        universe = UniverseBuilder(settings=self.settings, store=self.store).read_latest()
        if universe.is_empty():
            return frame
        included = universe.filter(pl.col("included")).select(ASSET_COLUMNS).unique()
        return frame.join(included, on=ASSET_COLUMNS, how="inner")

    @staticmethod
    def _with_decision_timestamp(frame: pl.DataFrame) -> pl.DataFrame:
        return (
            frame.sort([*ASSET_COLUMNS, "timestamp"])
            .with_columns(
                (
                    (pl.col("timestamp").diff().over(ASSET_COLUMNS) != HOUR_MS)
                    .fill_null(True)
                    .cast(pl.Int64)
                    .cum_sum()
                    .over(ASSET_COLUMNS)
                ).alias("segment_id"),
                (pl.col("timestamp") + HOUR_MS).alias("decision_timestamp"),
                (pl.col("close") * pl.col("volume")).alias("quote_volume"),
            )
            .select([*ASSET_COLUMNS, "segment_id", "decision_timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
        )


def build_rolling_four_hour_candles(one_hour: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the four most recently closed 1h candles at every hourly decision."""

    groups = [*ASSET_COLUMNS, "segment_id"]
    frame = (
        one_hour.sort([*groups, "decision_timestamp"])
        .with_columns(
            pl.col("decision_timestamp").shift(3).over(groups).alias("window_start_timestamp"),
            pl.col("open").shift(3).over(groups).alias("open_4h"),
            pl.col("high").rolling_max(window_size=4, min_samples=4).over(groups).alias("high_4h"),
            pl.col("low").rolling_min(window_size=4, min_samples=4).over(groups).alias("low_4h"),
            pl.col("volume").rolling_sum(window_size=4, min_samples=4).over(groups).alias("volume_4h"),
            pl.col("quote_volume").rolling_sum(window_size=4, min_samples=4).over(groups).alias("quote_volume_4h"),
        )
        .filter(pl.col("decision_timestamp") - pl.col("window_start_timestamp") == 3 * HOUR_MS)
        .with_columns(((pl.col("decision_timestamp") // HOUR_MS) % 4).cast(pl.Int8).alias("four_hour_phase"))
        .select(
            [
                *groups,
                "decision_timestamp",
                "four_hour_phase",
                pl.col("open_4h").alias("open"),
                pl.col("high_4h").alias("high"),
                pl.col("low_4h").alias("low"),
                "close",
                pl.col("volume_4h").alias("volume"),
                pl.col("quote_volume_4h").alias("quote_volume"),
            ]
        )
    )
    return frame


def build_factor_wide(data: pl.DataFrame, group_columns: list[str], prefix: str) -> pl.DataFrame:
    base = data.sort([*group_columns, "decision_timestamp"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over(group_columns) - 1.0).alias("ret_1"),
        ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("candle_body"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("hl_range_1"),
        ((pl.col("high") - pl.max_horizontal("open", "close")) / pl.col("close")).alias("upper_shadow"),
        ((pl.min_horizontal("open", "close") - pl.col("low")) / pl.col("close")).alias("lower_shadow"),
    )
    expressions: list[pl.Expr] = [
        pl.col("ret_1").alias(f"{prefix}return_1"),
        pl.col("ret_1").log1p().alias(f"{prefix}log_return_1"),
        pl.col("candle_body").alias(f"{prefix}candle_body_1"),
        pl.col("hl_range_1").alias(f"{prefix}range_1"),
        pl.col("upper_shadow").alias(f"{prefix}upper_shadow_1"),
        pl.col("lower_shadow").alias(f"{prefix}lower_shadow_1"),
    ]
    for window in FACTOR_WINDOWS:
        expressions.extend(
            [
                (pl.col("close") / pl.col("close").shift(window).over(group_columns) - 1.0).alias(f"{prefix}return_{window}"),
                pl.col("ret_1").rolling_mean(window_size=window, min_samples=window).over(group_columns).alias(f"{prefix}mean_return_{window}"),
                pl.col("ret_1").rolling_std(window_size=window, min_samples=window).over(group_columns).alias(f"{prefix}volatility_{window}"),
                (pl.col("close") / pl.col("close").rolling_mean(window_size=window, min_samples=window).over(group_columns) - 1.0).alias(f"{prefix}ma_gap_{window}"),
                (pl.col("close") / pl.col("close").rolling_max(window_size=window, min_samples=window).over(group_columns) - 1.0).alias(f"{prefix}distance_to_high_{window}"),
                (pl.col("close") / pl.col("close").rolling_min(window_size=window, min_samples=window).over(group_columns) - 1.0).alias(f"{prefix}distance_to_low_{window}"),
                pl.col("hl_range_1").rolling_mean(window_size=window, min_samples=window).over(group_columns).alias(f"{prefix}avg_range_{window}"),
                (pl.col("volume") / pl.col("volume").shift(window).over(group_columns) - 1.0).alias(f"{prefix}volume_change_{window}"),
                zscore_expr("volume", window, group_columns).alias(f"{prefix}volume_zscore_{window}"),
                zscore_expr("quote_volume", window, group_columns).alias(f"{prefix}quote_volume_zscore_{window}"),
                (
                    pl.col("hl_range_1")
                    / (pl.col("quote_volume").rolling_mean(window_size=window, min_samples=window).over(group_columns) + 1.0)
                ).alias(f"{prefix}range_to_quote_volume_{window}"),
            ]
        )
    feature_names = [expression.meta.output_name() for expression in expressions]
    return base.with_columns(expressions).select([*group_columns, "decision_timestamp", *feature_names])


def zscore_expr(column: str, window: int, group_columns: list[str]) -> pl.Expr:
    mean = pl.col(column).rolling_mean(window_size=window, min_samples=window).over(group_columns)
    std = pl.col(column).rolling_std(window_size=window, min_samples=window).over(group_columns)
    return pl.when(std > 0).then((pl.col(column) - mean) / std).otherwise(None)
