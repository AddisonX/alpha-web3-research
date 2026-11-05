from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from alpha_web3_research.config import Settings
from alpha_web3_research.data.storage import ParquetOHLCVStore
from alpha_web3_research.data.time import parse_timestamp_ms
from alpha_web3_research.universe.rules import (
    LEVERAGED_SUFFIXES,
    NON_CRYPTO_BASE_ASSETS,
    STABLE_BASE_ASSETS,
    is_non_crypto_base,
    is_stable_base,
)

DAY_MS = 86_400_000


@dataclass(frozen=True)
class UniverseBuildResult:
    snapshot_id: str
    generated_at: str
    included_count: int
    total_count: int
    tier: str
    path: Path
    latest_path: Path


class UniverseBuilder:
    def __init__(self, settings: Settings, store: ParquetOHLCVStore | None = None) -> None:
        self.settings = settings
        self.store = store or ParquetOHLCVStore(settings=settings)
        self.universe_root = settings.resolved_data_path() / "universe"

    def build(
        self,
        timeframes: list[str],
        min_rows: int,
        min_quote_volume: float = 0.0,
        min_days: int | None = None,
        liquidity_lookback_days: int = 30,
        liquidity_timeframe: str = "1h",
        min_median_daily_quote_volume: float = 0.0,
        min_mean_daily_quote_volume: float = 0.0,
        max_symbols: int | None = None,
        tier: str = "custom",
        exclude_non_crypto: bool = True,
        as_of: str | None = None,
    ) -> UniverseBuildResult:
        data = self.store.read_ohlcv()
        generated_at = datetime.now(timezone.utc).isoformat()
        snapshot_id = generated_at.replace(":", "").replace("-", "").split(".")[0].replace("+0000", "Z")
        if data.is_empty():
            frame = self._empty_frame()
        else:
            as_of_ms = parse_timestamp_ms(as_of)
            filtered = data.filter(pl.col("timeframe").is_in(timeframes))
            if as_of_ms is not None:
                filtered = filtered.filter(pl.col("timestamp") <= as_of_ms)
            frame = self._build_frame(
                data=filtered,
                timeframes=timeframes,
                min_rows=min_rows,
                min_quote_volume=min_quote_volume,
                min_days=min_days,
                liquidity_lookback_days=liquidity_lookback_days,
                liquidity_timeframe=liquidity_timeframe,
                min_median_daily_quote_volume=min_median_daily_quote_volume,
                min_mean_daily_quote_volume=min_mean_daily_quote_volume,
                max_symbols=max_symbols,
                tier=tier,
                exclude_non_crypto=exclude_non_crypto,
                as_of_ms=as_of_ms,
                generated_at=generated_at,
                snapshot_id=snapshot_id,
            )

        path = self.universe_root / f"snapshot_id={snapshot_id}" / "universe.parquet"
        latest_path = self.universe_root / "latest_universe.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        frame.write_parquet(latest_path)
        return UniverseBuildResult(
            snapshot_id=snapshot_id,
            generated_at=generated_at,
            included_count=frame.filter(pl.col("included")).height if "included" in frame.columns else 0,
            total_count=frame.height,
            tier=tier,
            path=path,
            latest_path=latest_path,
        )

    def read_latest(self) -> pl.DataFrame:
        path = self.universe_root / "latest_universe.parquet"
        if not path.exists():
            return self._empty_frame()
        return pl.read_parquet(path)

    def _build_frame(
        self,
        data: pl.DataFrame,
        timeframes: list[str],
        min_rows: int,
        min_quote_volume: float,
        min_days: int | None,
        liquidity_lookback_days: int,
        liquidity_timeframe: str,
        min_median_daily_quote_volume: float,
        min_mean_daily_quote_volume: float,
        max_symbols: int | None,
        tier: str,
        exclude_non_crypto: bool,
        as_of_ms: int | None,
        generated_at: str,
        snapshot_id: str,
    ) -> pl.DataFrame:
        required_timeframe_count = len(set(timeframes))
        required_rows = pl.DataFrame(
            {
                "timeframe": timeframes,
                "required_rows": [required_rows_for_timeframe(timeframe, min_rows, min_days) for timeframe in timeframes],
            }
        )
        enriched = data.with_columns(
                (pl.col("close") * pl.col("volume")).alias("quote_volume"),
                pl.col("symbol").map_elements(base_asset, return_dtype=pl.String).alias("base_asset"),
                pl.col("symbol").map_elements(quote_asset, return_dtype=pl.String).alias("quote_asset"),
            )
        effective_as_of_ms = as_of_ms if as_of_ms is not None else enriched.select(pl.max("timestamp")).item()
        liquidity_start_ms = effective_as_of_ms - liquidity_lookback_days * DAY_MS
        group_columns = ["exchange", "market_type", "symbol", "base_asset", "quote_asset"]

        timeframe_stats = (
            enriched.group_by(group_columns + ["timeframe"])
            .agg(
                pl.len().alias("rows"),
                pl.min("timestamp").alias("first_timestamp"),
                pl.max("timestamp").alias("last_timestamp"),
                pl.mean("quote_volume").alias("mean_quote_volume"),
            )
            .join(required_rows, on="timeframe", how="left")
            .with_columns(
                pl.col("required_rows").fill_null(min_rows).alias("required_rows"),
                (pl.col("rows") >= pl.col("required_rows").fill_null(min_rows)).alias("timeframe_has_min_rows"),
            )
        )

        symbol_stats = (
            timeframe_stats.group_by(group_columns)
            .agg(
                pl.col("timeframe").n_unique().alias("timeframe_count"),
                pl.col("timeframe").unique().sort().alias("available_timeframes"),
                pl.sum("rows").alias("rows_total"),
                pl.min("first_timestamp").alias("first_timestamp"),
                pl.max("last_timestamp").alias("last_timestamp"),
                pl.mean("mean_quote_volume").alias("mean_quote_volume"),
                pl.all("timeframe_has_min_rows").alias("has_min_rows"),
                pl.min("required_rows").alias("min_required_rows_per_timeframe"),
            )
            .with_columns(
                (pl.col("timeframe_count") >= required_timeframe_count).alias("has_required_timeframes"),
                (pl.col("mean_quote_volume") >= min_quote_volume).alias("has_min_quote_volume"),
                pl.col("base_asset").is_in(STABLE_BASE_ASSETS).alias("is_stable_base"),
                pl.col("base_asset").is_in(NON_CRYPTO_BASE_ASSETS).alias("is_non_crypto_base"),
                pl.any_horizontal([pl.col("base_asset").str.ends_with(suffix) for suffix in LEVERAGED_SUFFIXES]).alias("is_leveraged_base"),
            )
        )

        liquidity = (
            enriched.filter((pl.col("timeframe") == liquidity_timeframe) & (pl.col("timestamp") >= liquidity_start_ms))
            .with_columns((pl.col("timestamp") // DAY_MS).alias("day"))
            .group_by(group_columns + ["day"])
            .agg(pl.sum("quote_volume").alias("daily_quote_volume"))
            .group_by(group_columns)
            .agg(
                pl.median("daily_quote_volume").alias("median_daily_quote_volume"),
                pl.mean("daily_quote_volume").alias("mean_daily_quote_volume"),
                pl.len().alias("liquidity_day_count"),
            )
        )

        base = (
            symbol_stats.join(liquidity, on=group_columns, how="left")
            .with_columns(
                pl.col("median_daily_quote_volume").fill_null(0.0),
                pl.col("mean_daily_quote_volume").fill_null(0.0),
                pl.col("liquidity_day_count").fill_null(0),
            )
            .with_columns(
                (pl.col("median_daily_quote_volume") >= min_median_daily_quote_volume).alias("has_min_median_daily_quote_volume"),
                (pl.col("mean_daily_quote_volume") >= min_mean_daily_quote_volume).alias("has_min_mean_daily_quote_volume"),
            )
            .with_columns(
                (
                    pl.col("has_required_timeframes")
                    & pl.col("has_min_rows")
                    & pl.col("has_min_quote_volume")
                    & pl.col("has_min_median_daily_quote_volume")
                    & pl.col("has_min_mean_daily_quote_volume")
                    & ~pl.col("is_stable_base")
                    & ~pl.col("is_leveraged_base")
                    & ~(pl.lit(exclude_non_crypto) & pl.col("is_non_crypto_base"))
                ).alias("pre_rank_included")
            )
            .with_columns(
                pl.when(pl.col("pre_rank_included"))
                .then(pl.col("median_daily_quote_volume").rank(method="ordinal", descending=True))
                .otherwise(None)
                .alias("liquidity_rank")
            )
            .with_columns(
                (
                    pl.col("pre_rank_included")
                    & (pl.lit(max_symbols is None) | (pl.col("liquidity_rank") <= pl.lit(max_symbols or 0)))
                ).alias("included")
            )
            .with_columns(
                pl.struct(
                    [
                        "has_required_timeframes",
                        "has_min_rows",
                        "has_min_quote_volume",
                        "has_min_median_daily_quote_volume",
                        "has_min_mean_daily_quote_volume",
                        "is_stable_base",
                        "is_leveraged_base",
                        "is_non_crypto_base",
                        "pre_rank_included",
                        "liquidity_rank",
                    ]
                )
                .map_elements(lambda row: exclusion_reason(row, max_symbols=max_symbols, exclude_non_crypto=exclude_non_crypto), return_dtype=pl.String)
                .alias("exclusion_reason"),
                pl.lit(tier).alias("tier"),
                pl.lit("current_active_universe").alias("universe_mode"),
                pl.lit(True).alias("survivorship_bias"),
                pl.lit(min_days).alias("min_days"),
                pl.lit(liquidity_lookback_days).alias("liquidity_lookback_days"),
                pl.lit(liquidity_timeframe).alias("liquidity_timeframe"),
                pl.lit(min_median_daily_quote_volume).alias("min_median_daily_quote_volume"),
                pl.lit(min_mean_daily_quote_volume).alias("min_mean_daily_quote_volume"),
                pl.lit(max_symbols).alias("max_symbols"),
                pl.lit(effective_as_of_ms).alias("as_of_timestamp"),
                pl.lit(snapshot_id).alias("snapshot_id"),
                pl.lit(generated_at).alias("generated_at"),
            )
            .sort(["included", "liquidity_rank", "symbol"], descending=[True, False, False], nulls_last=True)
        )
        return base.select(
            [
                "snapshot_id",
                "generated_at",
                "tier",
                "universe_mode",
                "survivorship_bias",
                "exchange",
                "market_type",
                "symbol",
                "base_asset",
                "quote_asset",
                "included",
                "exclusion_reason",
                "liquidity_rank",
                "available_timeframes",
                "rows_total",
                "min_required_rows_per_timeframe",
                "first_timestamp",
                "last_timestamp",
                "as_of_timestamp",
                "mean_quote_volume",
                "median_daily_quote_volume",
                "mean_daily_quote_volume",
                "liquidity_day_count",
                "min_days",
                "liquidity_lookback_days",
                "liquidity_timeframe",
                "min_median_daily_quote_volume",
                "min_mean_daily_quote_volume",
                "max_symbols",
            ]
        )

    @staticmethod
    def _empty_frame() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "snapshot_id": pl.String,
                "generated_at": pl.String,
                "tier": pl.String,
                "universe_mode": pl.String,
                "survivorship_bias": pl.Boolean,
                "exchange": pl.String,
                "market_type": pl.String,
                "symbol": pl.String,
                "base_asset": pl.String,
                "quote_asset": pl.String,
                "included": pl.Boolean,
                "exclusion_reason": pl.String,
                "liquidity_rank": pl.Int64,
                "available_timeframes": pl.List(pl.String),
                "rows_total": pl.Int64,
                "min_required_rows_per_timeframe": pl.Int64,
                "first_timestamp": pl.Int64,
                "last_timestamp": pl.Int64,
                "as_of_timestamp": pl.Int64,
                "mean_quote_volume": pl.Float64,
                "median_daily_quote_volume": pl.Float64,
                "mean_daily_quote_volume": pl.Float64,
                "liquidity_day_count": pl.Int64,
                "min_days": pl.Int64,
                "liquidity_lookback_days": pl.Int64,
                "liquidity_timeframe": pl.String,
                "min_median_daily_quote_volume": pl.Float64,
                "min_mean_daily_quote_volume": pl.Float64,
                "max_symbols": pl.Int64,
            }
        )


def base_asset(symbol: str) -> str:
    return symbol.split("/")[0]


def quote_asset(symbol: str) -> str:
    if "/" not in symbol:
        return ""
    rest = symbol.split("/", maxsplit=1)[1]
    return rest.split(":", maxsplit=1)[0]


def required_rows_for_timeframe(timeframe: str, min_rows: int, min_days: int | None) -> int:
    if min_days is None:
        return min_rows
    return max(min_rows, int((min_days * DAY_MS) / timeframe_to_ms(timeframe)))


def timeframe_to_ms(timeframe: str) -> int:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 3_600_000
    if unit == "d":
        return value * DAY_MS
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def exclusion_reason(row: dict[str, object], max_symbols: int | None, exclude_non_crypto: bool) -> str:
    reasons: list[str] = []
    if not row["has_required_timeframes"]:
        reasons.append("missing_required_timeframe")
    if not row["has_min_rows"]:
        reasons.append("insufficient_history")
    if not row["has_min_quote_volume"]:
        reasons.append("insufficient_quote_volume")
    if not row["has_min_median_daily_quote_volume"]:
        reasons.append("insufficient_median_daily_quote_volume")
    if not row["has_min_mean_daily_quote_volume"]:
        reasons.append("insufficient_mean_daily_quote_volume")
    if row["is_stable_base"]:
        reasons.append("stable_base_asset")
    if row["is_leveraged_base"]:
        reasons.append("leveraged_token")
    if exclude_non_crypto and row["is_non_crypto_base"]:
        reasons.append("non_crypto_underlying")
    rank = row.get("liquidity_rank")
    if row.get("pre_rank_included") and max_symbols is not None and rank is not None and rank > max_symbols:
        reasons.append("outside_liquidity_rank_limit")
    return ",".join(reasons) if reasons else ""
