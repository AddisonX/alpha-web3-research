from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from alpha_web3_research.backtest.freqtrade import (
    HOUR_MS,
    FreqtradeSignalExporter,
    _freqtrade_config,
)


def _timestamp(day: int, hour: int) -> int:
    value = datetime(2026, 4, day, hour, tzinfo=timezone.utc)
    return int(value.timestamp() * 1_000)


def test_build_signals_selects_exact_daily_tails_and_holds_for_24_hours() -> None:
    symbols = [f"COIN{index}/USDT:USDT" for index in range(10)]
    rows: list[dict[str, object]] = []
    for day in (21, 22):
        for hour in range(0, 24, 4):
            for index, symbol in enumerate(symbols):
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": _timestamp(day, hour),
                        "fold": 1,
                        "model_score": float(9 - index),
                        "model_score_rank_pct": 1.0 - (index / 9),
                    }
                )

    signals = FreqtradeSignalExporter._build_signals(
        predictions=pl.DataFrame(rows),
        top_quantile_fraction=0.10,
        rebalance_interval_hours=24,
        rebalance_hour_utc=0,
    )
    decisions = signals.filter(pl.col("is_rebalance"))

    assert decisions.get_column("decision_timestamp").n_unique() == 2
    assert decisions.filter(pl.col("target_position") == 1).height == 2
    assert decisions.filter(pl.col("target_position") == -1).height == 2
    assert signals.filter(pl.col("pair") == symbols[0]).get_column("target_position").to_list() == [1] * 12
    assert signals.filter(pl.col("pair") == symbols[-1]).get_column("target_position").to_list() == [-1] * 12
    assert signals.get_column("signal_valid").all()
    assert (
        decisions.get_column("signal_valid_until") - decisions.get_column("decision_timestamp")
    ).unique().to_list() == [24 * HOUR_MS]


def test_build_signals_breaks_score_ties_without_expanding_tail_count() -> None:
    rows = [
        {
            "symbol": f"COIN{index}/USDT:USDT",
            "timestamp": _timestamp(21, 0),
            "fold": 1,
            "model_score": 1.0,
            "model_score_rank_pct": 0.5,
        }
        for index in range(20)
    ]

    decisions = FreqtradeSignalExporter._build_signals(
        predictions=pl.DataFrame(rows),
        top_quantile_fraction=0.10,
        rebalance_interval_hours=24,
        rebalance_hour_utc=0,
    ).filter(pl.col("is_rebalance"))

    assert decisions.filter(pl.col("target_position") == 1).height == 2
    assert decisions.filter(pl.col("target_position") == -1).height == 2
    assert decisions.filter(pl.col("target_position") == 0).height == 16


def test_freqtrade_config_supports_market_order_bias_checks() -> None:
    config = _freqtrade_config(
        pairs=["BTC/USDT:USDT"],
        timeframe="4h",
        use_testnet_market_metadata=True,
    )

    assert config["trading_mode"] == "futures"
    assert config["margin_mode"] == "isolated"
    assert config["entry_pricing"]["price_side"] == "other"
    assert config["exit_pricing"]["price_side"] == "other"
    assert config["futures_funding_rate"] == 0.0
    assert config["pairlists"] == [{"method": "StaticPairList", "allow_inactive": True}]
    assert config["exchange"]["ccxt_config"]["options"]["fetchMarkets"] == {"types": ["linear"]}


def test_freqtrade_config_supports_long_only_position_slots() -> None:
    config = _freqtrade_config(
        pairs=["BTC/USDT:USDT"],
        timeframe="4h",
        use_testnet_market_metadata=True,
        max_open_trades=9,
        bot_name="alpha-cross-sectional-long-only-backtest",
    )

    assert config["max_open_trades"] == 9
    assert config["bot_name"] == "alpha-cross-sectional-long-only-backtest"


@pytest.mark.parametrize(
    ("fraction", "interval", "hour"),
    [(0.0, 24, 0), (0.5, 24, 0), (0.1, 0, 0), (0.1, 24, 24), (0.1, 4, 4)],
)
def test_invalid_signal_export_settings_are_rejected(
    fraction: float,
    interval: int,
    hour: int,
) -> None:
    with pytest.raises(ValueError):
        FreqtradeSignalExporter._validate_inputs(
            top_quantile_fraction=fraction,
            rebalance_interval_hours=interval,
            rebalance_hour_utc=hour,
        )
