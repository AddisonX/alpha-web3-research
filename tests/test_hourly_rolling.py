from __future__ import annotations

import numpy as np
import polars as pl

from alpha_web3_research.backtest.hourly_rolling import (
    HourlyPortfolioRules,
    apply_trade_controls,
    capped_normalized_weights,
)
from alpha_web3_research.factors.hourly import HOUR_MS, build_rolling_four_hour_candles
from alpha_web3_research.labels.hourly import build_hourly_labels
from alpha_web3_research.models.hourly_rolling import split_hourly_dataset


def _candles(count: int, gap_after: int | None = None) -> pl.DataFrame:
    timestamps: list[int] = []
    current = 0
    for index in range(count):
        timestamps.append(current)
        current += HOUR_MS
        if gap_after is not None and index == gap_after:
            current += 10 * HOUR_MS
    closes = [100.0 + index for index in range(count)]
    return pl.DataFrame(
        {
            "exchange": ["binanceusdm"] * count,
            "market_type": ["swap"] * count,
            "symbol": ["BTC/USDT:USDT"] * count,
            "timestamp": timestamps,
            "decision_timestamp": [item + HOUR_MS for item in timestamps],
            "segment_id": [1 if gap_after is None or index <= gap_after else 2 for index in range(count)],
            "open": [value - 0.5 for value in closes],
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "volume": [10.0] * count,
            "quote_volume": [value * 10.0 for value in closes],
        }
    )


def test_rolling_four_hour_candle_uses_latest_four_closed_hours() -> None:
    rolling = build_rolling_four_hour_candles(_candles(5))
    first = rolling.row(0, named=True)

    assert rolling.height == 2
    assert first["decision_timestamp"] == 4 * HOUR_MS
    assert first["open"] == 99.5
    assert first["high"] == 104.0
    assert first["low"] == 99.0
    assert first["close"] == 103.0
    assert first["volume"] == 40.0


def test_hourly_labels_do_not_bridge_market_data_gaps() -> None:
    raw = _candles(8, gap_after=3).drop(["decision_timestamp", "segment_id", "quote_volume"])
    labels = build_hourly_labels(raw, (1, 4))
    before_gap = labels.filter(pl.col("decision_timestamp") == 4 * HOUR_MS).row(0, named=True)

    assert before_gap["forward_return_h1"] is None
    assert before_gap["forward_return_h4"] is None
    assert before_gap["forward_volatility_h4"] is None


def test_target_weighting_has_no_position_count_cap() -> None:
    weights = capped_normalized_weights(np.ones(20), gross_exposure=0.90, max_asset_weight=0.12)

    assert np.count_nonzero(weights) == 20
    assert np.isclose(weights.sum(), 0.90)
    assert weights.max() <= 0.12


def test_trade_controls_enforce_hourly_turnover_and_gross_limits() -> None:
    current = np.full(20, 0.04)
    desired = np.concatenate([np.zeros(10), np.full(10, 0.09)])
    rules = HourlyPortfolioRules(max_hourly_turnover=0.25)
    post_trade, turnover = apply_trade_controls(current, desired, rules)

    assert turnover <= 0.25 + 1e-12
    assert post_trade.sum() <= rules.gross_exposure + 1e-12
    assert post_trade.min() >= 0


def test_asset_cap_reduction_bypasses_minimum_trade_band() -> None:
    rules = HourlyPortfolioRules(max_asset_weight=0.12, minimum_trade_weight=0.02)
    current = np.array([0.135, 0.05])
    desired = np.array([0.12, 0.05])
    post_trade, _ = apply_trade_controls(current, desired, rules)

    assert np.isclose(post_trade[0], 0.12)


def test_single_time_split_purges_longest_label_before_boundaries() -> None:
    dataset = pl.DataFrame(
        {
            "decision_timestamp": list(range(200)),
            "symbol": ["BTC/USDT:USDT"] * 200,
        }
    )
    split, metadata = split_hourly_dataset(dataset, train_fraction=0.70, valid_fraction=0.15, embargo_hours=24)
    train_end = split.filter(pl.col("split") == "train").get_column("decision_timestamp").max()
    valid_start = split.filter(pl.col("split") == "valid").get_column("decision_timestamp").min()
    valid_end = split.filter(pl.col("split") == "valid").get_column("decision_timestamp").max()
    test_start = split.filter(pl.col("split") == "test").get_column("decision_timestamp").min()

    assert valid_start - train_end > 24
    assert test_start - valid_end > 24
    assert metadata["embargo_hours"] == 24
