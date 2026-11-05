"""Build a fixed inception universe using only bars closed at the as-of time."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from alpha_web3_research.universe.rules import is_non_crypto_base, is_stable_base

DAY_MS = 86_400_000
BAR_MS = 14_400_000


@dataclass(frozen=True)
class PointInTimeUniverseRules:
    min_history_days: int = 180
    liquidity_lookback_days: int = 30
    min_median_daily_quote_volume: float = 5_000_000
    min_coverage: float = 0.98
    max_symbols: int = 120


def build_point_in_time_universe(
    bars: pl.DataFrame, as_of_ms: int, rules: PointInTimeUniverseRules | None = None,
) -> pl.DataFrame:
    rules = rules or PointInTimeUniverseRules()
    if rules.min_history_days < rules.liquidity_lookback_days or not 0 < rules.min_coverage <= 1 or rules.max_symbols < 1:
        raise ValueError("invalid point-in-time universe rules")
    history_start = as_of_ms - rules.min_history_days * DAY_MS
    liquidity_start = as_of_ms - rules.liquidity_lookback_days * DAY_MS
    historical = bars.filter(pl.col("timestamp") + BAR_MS <= as_of_ms)
    if historical.is_empty():
        raise ValueError("no closed historical bars available at the universe as-of time")
    recent = historical.filter(pl.col("timestamp") >= history_start)
    stats = recent.group_by("symbol").agg(
        pl.col("timestamp").n_unique().alias("history_bars"),
        pl.col("timestamp").max().alias("last_closed_bar_open"),
    )
    first_seen = historical.group_by("symbol").agg(pl.col("timestamp").min().alias("first_seen_bar_open"))
    daily = (
        historical.filter(pl.col("timestamp") >= liquidity_start)
        .with_columns((pl.col("timestamp") // DAY_MS).alias("day"))
        .group_by(["symbol", "day"])
        .agg(pl.col("quote_volume").sum().alias("daily_quote_volume"), pl.len().alias("bar_count"))
        .filter(pl.col("bar_count") == 6)
        .group_by("symbol")
        .agg(pl.col("daily_quote_volume").median().alias("median_daily_quote_volume"),
             pl.len().alias("complete_liquidity_days"))
    )
    stats = first_seen.join(stats, on="symbol", how="left").join(daily, on="symbol", how="left")
    rows = []
    for row in stats.sort("symbol").iter_rows(named=True):
        base = row["symbol"].split("/")[0]
        reasons = []
        if is_stable_base(base):
            reasons.append("stable_base_asset")
        if is_non_crypto_base(base) or base in {"PAXG", "XAUT"}:
            reasons.append("non_crypto_underlying")
        # Multiplier crypto contracts such as 1000SHIB are deliberately retained.
        if base.endswith(("BULL", "BEAR", "2L", "2S", "3L", "3S")):
            reasons.append("leveraged_token")
        if row["first_seen_bar_open"] > history_start:
            reasons.append("listing_history_under_minimum")
        if (row["history_bars"] or 0) < rules.min_history_days * 6 * rules.min_coverage:
            reasons.append("insufficient_history_coverage")
        if row["last_closed_bar_open"] != as_of_ms - BAR_MS:
            reasons.append("not_observed_trading_at_inception")
        if (row["complete_liquidity_days"] or 0) < rules.liquidity_lookback_days:
            reasons.append("incomplete_liquidity_window")
        if (row["median_daily_quote_volume"] or 0) < rules.min_median_daily_quote_volume:
            reasons.append("insufficient_daily_liquidity")
        rows.append({**row, "pre_rank_included": not reasons, "exclusion_reason": ",".join(reasons)})
    eligible = sorted([row for row in rows if row["pre_rank_included"]],
                      key=lambda row: (-row["median_daily_quote_volume"], row["symbol"]))
    ranks = {row["symbol"]: index + 1 for index, row in enumerate(eligible)}
    for row in rows:
        rank = ranks.get(row["symbol"])
        row.update({"liquidity_rank": rank, "included": rank is not None and rank <= rules.max_symbols,
                    "as_of_timestamp": as_of_ms, "universe_mode": "fixed_inception_point_in_time",
                    "future_survival_used_for_selection": False})
        if rank is not None and rank > rules.max_symbols:
            row["exclusion_reason"] = "outside_liquidity_rank_limit"
    return pl.DataFrame(rows).sort(["included", "liquidity_rank", "symbol"],
                                  descending=[True, False, False], nulls_last=True)


def audit_universe_survival(universe: pl.DataFrame, bars: pl.DataFrame, audit_end_ms: int) -> pl.DataFrame:
    coverage = bars.group_by("symbol").agg(
        pl.col("timestamp").min().alias("observed_start_ms"),
        pl.col("timestamp").max().alias("observed_last_open_ms"), pl.len().alias("observed_bars"),
    )
    return universe.filter(pl.col("included")).join(coverage, on="symbol", how="left").with_columns(
        pl.lit(audit_end_ms).alias("audit_end_ms"),
        (pl.col("observed_last_open_ms") + BAR_MS >= audit_end_ms).fill_null(False).alias("data_reaches_audit_end"),
        pl.lit("archive coverage only; missing data is not proof of official delisting").alias("audit_scope"),
    )
