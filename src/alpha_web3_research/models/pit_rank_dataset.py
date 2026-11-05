"""Rank features and exact-clock daily excess labels for inception-universe research."""
from __future__ import annotations

import polars as pl

from alpha_web3_research.config import Settings
from alpha_web3_research.data.binance_archive import BAR_MS
from alpha_web3_research.data.ohlcv import OHLCV_COLUMNS
from alpha_web3_research.factors.builder import FactorBuilder

DAY_MS = 86_400_000


def rank_cross_section(features: pl.DataFrame, names: list[str], min_cross_section: int) -> pl.DataFrame:
    frame = features.filter(pl.all_horizontal([pl.col(name).is_finite() for name in names]))
    frame = frame.with_columns(pl.len().over("decision_timestamp").alias("cross_section_count"))
    return frame.filter(pl.col("cross_section_count") >= min_cross_section).with_columns([
        ((pl.col(name).rank(method="average").over("decision_timestamp") - 1.0)
         / (pl.col("cross_section_count") - 1.0)).alias(name)
        for name in names
    ])


def build_daily_excess_labels(bars: pl.DataFrame, decision_hour_utc: int = 4) -> pl.DataFrame:
    ordered = bars.sort(["symbol", "timestamp"])
    labels = ordered.with_columns(
        pl.col("timestamp").shift(-6).over("symbol").alias("exit_timestamp"),
        pl.col("open").shift(-6).over("symbol").alias("exit_open"),
    ).filter(
        ((pl.col("timestamp") % DAY_MS) == decision_hour_utc * 3_600_000)
        & (pl.col("exit_timestamp") == pl.col("timestamp") + DAY_MS)
    ).with_columns(
        (pl.col("exit_open") / pl.col("open") - 1.0).alias("forward_return"),
        pl.col("timestamp").alias("decision_timestamp"),
        pl.col("exit_timestamp").alias("label_end_timestamp"),
    )
    labels = labels.filter(pl.col("forward_return").is_finite())
    return labels.with_columns(
        pl.col("forward_return").mean().over("decision_timestamp").alias("universe_forward_return"),
        pl.len().over("decision_timestamp").alias("label_universe_count"),
    ).with_columns(
        (pl.col("forward_return") - pl.col("universe_forward_return")).alias("excess_forward_return")
    ).select(
        "symbol", "decision_timestamp", "label_end_timestamp", "forward_return",
        "universe_forward_return", "excess_forward_return", "label_universe_count",
    )


def build_pit_rank_dataset(
    bars: pl.DataFrame, names: list[str], settings: Settings,
    min_cross_section: int = 20, decision_hour_utc: int = 4,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if len(names) != len(set(names)) or not names:
        raise ValueError("factor names must be nonempty and unique")
    if bars.select(pl.struct("symbol", "timestamp").n_unique()).item() != bars.height:
        raise ValueError("duplicate symbol/timestamp bars")
    builder = FactorBuilder(settings)
    outputs = []
    for symbol_frame in bars.sort(["symbol", "timestamp"]).partition_by("symbol", maintain_order=True):
        segmented = symbol_frame.with_columns(
            (pl.col("timestamp").diff().fill_null(BAR_MS) != BAR_MS).cum_sum().alias("segment_id")
        )
        for segment in segmented.partition_by("segment_id", maintain_order=True):
            if segment.height < 25:
                continue
            first_ready_timestamp = segment["timestamp"][24]
            long = builder._build_factor_long_frame(segment.select(OHLCV_COLUMNS)).filter(
                pl.col("factor_name").is_in(names)
                & (pl.col("timestamp") >= first_ready_timestamp)
                & (((pl.col("timestamp") + BAR_MS) % DAY_MS) == decision_hour_utc * 3_600_000)
            )
            if long.is_empty():
                continue
            wide = long.pivot(index=["symbol", "timestamp"], on="factor_name", values="factor_value")
            if any(name not in wide.columns for name in names):
                raise ValueError("requested factor missing from factor builder")
            outputs.append(wide.with_columns((pl.col("timestamp") + BAR_MS).alias("decision_timestamp")))
    if not outputs:
        raise ValueError("no daily factors after continuous-history warmup")
    ranked = rank_cross_section(pl.concat(outputs, how="vertical_relaxed"), names, min_cross_section)
    labels = build_daily_excess_labels(bars, decision_hour_utc)
    # Future-label availability must not decide whether a test-time prediction exists.
    dataset = ranked.join(labels, on=["symbol", "decision_timestamp"], how="left").sort(
        ["decision_timestamp", "symbol"]
    )
    return dataset, labels
