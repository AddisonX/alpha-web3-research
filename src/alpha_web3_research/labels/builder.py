from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from alpha_web3_research.config import Settings
from alpha_web3_research.data.storage import ParquetOHLCVStore
from alpha_web3_research.universe.builder import UniverseBuilder


@dataclass(frozen=True)
class LabelBuildResult:
    rows: int
    path: Path
    latest_path: Path
    horizons: list[int]
    timeframe: str


class LabelBuilder:
    def __init__(self, settings: Settings, store: ParquetOHLCVStore | None = None) -> None:
        self.settings = settings
        self.store = store or ParquetOHLCVStore(settings=settings)
        self.labels_root = settings.resolved_data_path() / "labels"

    def build(
        self,
        timeframe: str,
        horizons: list[int],
        volatility_lookback: int,
        universe_only: bool = True,
    ) -> LabelBuildResult:
        data = self.store.read_ohlcv()
        if data.is_empty():
            frame = self._empty_frame()
        else:
            frame = data.filter(pl.col("timeframe") == timeframe)
            if universe_only:
                frame = self._filter_to_latest_universe(frame)
            frame = self._build_labels(frame, timeframe, horizons, volatility_lookback)

        path = self.labels_root / f"timeframe={timeframe}" / "labels.parquet"
        latest_path = self.labels_root / "latest_labels.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        self._combined_labels().write_parquet(latest_path)
        return LabelBuildResult(rows=frame.height, path=path, latest_path=latest_path, horizons=horizons, timeframe=timeframe)

    def _combined_labels(self) -> pl.DataFrame:
        files = sorted(self.labels_root.glob("timeframe=*/labels.parquet"))
        if not files:
            return self._empty_frame()
        return pl.concat([pl.read_parquet(file) for file in files], how="diagonal_relaxed").sort(
            ["timeframe", "horizon_bars", "timestamp", "symbol"]
        )

    def _filter_to_latest_universe(self, frame: pl.DataFrame) -> pl.DataFrame:
        universe = UniverseBuilder(settings=self.settings, store=self.store).read_latest()
        if universe.is_empty():
            return frame
        included = universe.filter(pl.col("included")).select(["exchange", "market_type", "symbol"]).unique()
        return frame.join(included, on=["exchange", "market_type", "symbol"], how="inner")

    def _build_labels(
        self,
        data: pl.DataFrame,
        timeframe: str,
        horizons: list[int],
        volatility_lookback: int,
    ) -> pl.DataFrame:
        base = (
            data.sort(["exchange", "market_type", "symbol", "timestamp"])
            .with_columns(
                (pl.col("close") / pl.col("close").shift(1).over(["exchange", "market_type", "symbol"]) - 1.0).alias("historical_return_1bar")
            )
            .with_columns(
                pl.col("historical_return_1bar")
                .rolling_std(window_size=volatility_lookback, min_samples=max(2, min(volatility_lookback, 2)))
                .over(["exchange", "market_type", "symbol"])
                .alias("realized_volatility")
            )
        )

        frames: list[pl.DataFrame] = []
        for horizon in horizons:
            label = (
                base.with_columns(
                    pl.col("close").shift(-horizon).over(["exchange", "market_type", "symbol"]).alias("future_close"),
                    pl.lit(horizon).alias("horizon_bars"),
                    pl.lit(f"{horizon}bar").alias("horizon_name"),
                )
                .with_columns((pl.col("future_close") / pl.col("close") - 1.0).alias("forward_return"))
                .with_columns(
                    pl.when((pl.col("realized_volatility").is_not_null()) & (pl.col("realized_volatility") > 0))
                    .then(pl.col("forward_return") / pl.col("realized_volatility"))
                    .otherwise(None)
                    .alias("volatility_adjusted_return")
                )
                .filter(pl.col("future_close").is_not_null())
            )
            frames.append(label)

        if not frames:
            return self._empty_frame()

        labels = pl.concat(frames, how="diagonal_relaxed")
        group_cols = ["exchange", "market_type", "timeframe", "timestamp", "horizon_bars"]
        return (
            labels.with_columns(
                (pl.len().over(group_cols)).alias("cross_section_count"),
                pl.col("forward_return").rank(method="average").over(group_cols).alias("forward_return_rank"),
            )
            .with_columns(
                pl.when(pl.col("cross_section_count") > 1)
                .then((pl.col("forward_return_rank") - 1.0) / (pl.col("cross_section_count") - 1.0))
                .otherwise(None)
                .alias("forward_return_rank_pct")
            )
            .with_columns(pl.lit(datetime.now(timezone.utc).isoformat()).alias("generated_at"))
            .select(
                [
                    "generated_at",
                    "exchange",
                    "market_type",
                    "symbol",
                    "timeframe",
                    "timestamp",
                    "close",
                    "horizon_bars",
                    "horizon_name",
                    "future_close",
                    "forward_return",
                    "forward_return_rank",
                    "forward_return_rank_pct",
                    "realized_volatility",
                    "volatility_adjusted_return",
                    "cross_section_count",
                ]
            )
            .sort(["timeframe", "horizon_bars", "timestamp", "symbol"])
        )

    @staticmethod
    def _empty_frame() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "generated_at": pl.String,
                "exchange": pl.String,
                "market_type": pl.String,
                "symbol": pl.String,
                "timeframe": pl.String,
                "timestamp": pl.Int64,
                "close": pl.Float64,
                "horizon_bars": pl.Int64,
                "horizon_name": pl.String,
                "future_close": pl.Float64,
                "forward_return": pl.Float64,
                "forward_return_rank": pl.Float64,
                "forward_return_rank_pct": pl.Float64,
                "realized_volatility": pl.Float64,
                "volatility_adjusted_return": pl.Float64,
                "cross_section_count": pl.UInt32,
            }
        )
