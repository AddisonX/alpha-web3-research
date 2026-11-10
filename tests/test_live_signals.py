from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl

from src.live.signals import (
    HOUR_MS,
    build_decision_feature_matrix,
    build_signal_timeline,
    latest_closed_decision_timestamp,
)
from src.strategies.lightgbm import rank_long_only_scores


def _timestamp(day: int, hour: int) -> int:
    return int(datetime(2026, 7, day, hour, tzinfo=timezone.utc).timestamp() * 1000)


def test_latest_decision_waits_until_daily_candle_is_closed() -> None:
    timeframe_ms = 4 * HOUR_MS

    assert latest_closed_decision_timestamp(
        now_ms=_timestamp(27, 3),
        timeframe_ms=timeframe_ms,
        rebalance_interval_hours=24,
        rebalance_hour_utc=0,
    ) == _timestamp(26, 0)
    assert latest_closed_decision_timestamp(
        now_ms=_timestamp(27, 4) + 1,
        timeframe_ms=timeframe_ms,
        rebalance_interval_hours=24,
        rebalance_hour_utc=0,
    ) == _timestamp(27, 0)


def test_live_ranking_selects_exact_top_decile_and_persists_for_24_hours() -> None:
    pairs = [f"COIN{index}/USDT:USDT" for index in range(88)]
    ranked = rank_long_only_scores(
        pairs=pairs,
        scores=np.arange(88, dtype=float),
        top_quantile_fraction=0.10,
    )

    assert ranked.filter(pl.col("target_position") == 1).height == 9
    assert ranked.row(0, named=True)["pair"] == pairs[-1]
    assert ranked.row(0, named=True)["model_score_rank_pct"] == 1.0

    timeline = build_signal_timeline(
        ranked=ranked,
        decision_timestamp=_timestamp(27, 0),
        generated_at="2026-07-27T04:00:10+00:00",
        timeframe_ms=4 * HOUR_MS,
        rebalance_interval_hours=24,
        source_experiment="walk_forward_test",
    )
    assert timeline.height == 88 * 7
    assert timeline.filter(pl.col("is_rebalance")).height == 88
    assert timeline.filter(pl.col("target_position") == 1).height == 9 * 7


def test_feature_matrix_preserves_manifest_feature_order() -> None:
    factors = pl.DataFrame(
        {
            "symbol": ["B/USDT:USDT", "A/USDT:USDT", "A/USDT:USDT"],
            "timestamp": [_timestamp(27, 0)] * 3,
            "factor_name": ["return_1", "return_1", "range_1"],
            "factor_value": [2.0, 1.0, 3.0],
        }
    )

    matrix = build_decision_feature_matrix(
        factors=factors,
        decision_timestamp=_timestamp(27, 0),
        feature_names=["range_1", "return_1", "missing_factor"],
    )

    assert matrix.columns == ["symbol", "range_1", "return_1", "missing_factor"]
    assert matrix.get_column("symbol").to_list() == ["A/USDT:USDT", "B/USDT:USDT"]
    assert matrix.get_column("missing_factor").null_count() == 2
