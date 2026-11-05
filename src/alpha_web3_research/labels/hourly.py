from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from alpha_web3_research.config import Settings
from alpha_web3_research.data.storage import ParquetOHLCVStore
from alpha_web3_research.data.contracts import BarQuery
from alpha_web3_research.data.reader import MarketDataReader
from alpha_web3_research.factors.hourly import ASSET_COLUMNS, HOUR_MS
from alpha_web3_research.universe.builder import UniverseBuilder

DEFAULT_HORIZONS = (1, 4, 12, 24)


@dataclass(frozen=True)
class HourlyLabelBuildResult:
    rows: int
    symbol_count: int
    horizons: tuple[int, ...]
    path: Path


class HourlyMultiHorizonLabelBuilder:
    """Build absolute-return and future-path-risk labels on hourly decision timestamps."""

    def __init__(self, settings: Settings, store: ParquetOHLCVStore | None = None) -> None:
        self.settings = settings
        self.store = store or ParquetOHLCVStore(settings=settings)
        self.output_path = settings.resolved_data_path() / "labels" / "hourly_multihorizon.parquet"

    def build(
        self,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        universe_only: bool = True,
        write: bool = True,
    ) -> tuple[pl.DataFrame, HourlyLabelBuildResult]:
        if not horizons or any(horizon <= 0 for horizon in horizons):
            raise ValueError("horizons must contain positive hours")
        one_hour = MarketDataReader(self.store).read_bars(
            BarQuery(self.settings.exchange_id, self.settings.market_type, "1h")
        )
        if universe_only and not one_hour.is_empty():
            universe = UniverseBuilder(settings=self.settings, store=self.store).read_latest()
            if not universe.is_empty():
                included = universe.filter(pl.col("included")).select(ASSET_COLUMNS).unique()
                one_hour = one_hour.join(included, on=ASSET_COLUMNS, how="inner")
        labels = build_hourly_labels(one_hour, horizons)
        if write:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            labels.write_parquet(self.output_path)
        result = HourlyLabelBuildResult(
            rows=labels.height,
            symbol_count=labels.get_column("symbol").n_unique() if not labels.is_empty() else 0,
            horizons=horizons,
            path=self.output_path,
        )
        return labels, result


def build_hourly_labels(one_hour: pl.DataFrame, horizons: tuple[int, ...]) -> pl.DataFrame:
    groups = [*ASSET_COLUMNS, "segment_id"]
    if one_hour.is_empty():
        return pl.DataFrame()
    base = (
        one_hour.sort([*ASSET_COLUMNS, "timestamp"])
        .with_columns(
            (
                (pl.col("timestamp").diff().over(ASSET_COLUMNS) != HOUR_MS)
                .fill_null(True)
                .cast(pl.Int64)
                .cum_sum()
                .over(ASSET_COLUMNS)
            ).alias("segment_id"),
            (pl.col("timestamp") + HOUR_MS).alias("decision_timestamp"),
        )
        .select([*groups, "decision_timestamp", "close"])
    )
    expressions: list[pl.Expr] = []
    for horizon in horizons:
        future_close = pl.col("close").shift(-horizon).over(groups)
        squared_path_returns = [
            (
                (
                    pl.col("close").shift(-step).over(groups)
                    / pl.col("close").shift(-(step - 1)).over(groups)
                ).log()
                ** 2
            )
            for step in range(1, horizon + 1)
        ]
        expressions.extend(
            [
                (future_close / pl.col("close") - 1.0).alias(f"forward_return_h{horizon}"),
                pl.when(future_close.is_not_null())
                .then(pl.sum_horizontal(squared_path_returns).sqrt())
                .otherwise(None)
                .alias(f"forward_volatility_h{horizon}"),
            ]
        )
    return (
        base.with_columns(expressions)
        .with_columns(
            pl.col("forward_return_h1").alias("next_hour_return"),
            pl.lit(datetime.now(timezone.utc).isoformat()).alias("generated_at"),
        )
        .drop("segment_id")
        .sort(["decision_timestamp", "symbol"])
    )
