from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.live.dashboard import (
    LiveDashboardProvider,
    build_equity_history_response,
    history_max_drawdown_pct,
    position_snapshot,
)


def test_position_snapshot_includes_round_trip_fees() -> None:
    position = position_snapshot(
        trade={
            "id": 1,
            "pair": "BTC/USDT:USDT",
            "is_short": False,
            "fee_open": 0.0005,
            "fee_close": 0.0005,
            "open_rate": 100.0,
            "stake_amount": 1000.0,
            "amount": 10.0,
            "open_date": "2026-07-27 00:00:00",
            "stop_loss": 90.0,
            "leverage": 1.0,
            "enter_tag": "model_top_quantile",
        },
        current_rate=110.0,
        signal={"model_score_rank_pct": 1.0, "score_position": 1},
        now=datetime(2026, 7, 27, 1, tzinfo=timezone.utc),
    )

    assert 9.8 < position["pnl_pct"] < 10.0
    assert 98 < position["pnl_abs"] < 100
    assert position["duration_minutes"] == 60
    assert position["model_score_rank_pct"] == 100.0


def test_history_max_drawdown_uses_running_equity_peak() -> None:
    history = [
        {"equity": 10_000},
        {"equity": 12_000},
        {"equity": 10_800},
        {"equity": 11_500},
    ]

    assert history_max_drawdown_pct(history) == 10.0


def test_history_max_drawdown_includes_starting_balance() -> None:
    assert history_max_drawdown_pct([{"equity": 9_900}]) == 1.0


def test_equity_history_filters_and_aggregates() -> None:
    history = [
        {
            "timestamp": datetime(2026, 7, 1, hour, minute, tzinfo=timezone.utc).isoformat(),
            "equity": 10_000 + hour * 60 + minute,
            "total_pnl": hour * 60 + minute,
            "realized_pnl": 0.0,
            "unrealized_pnl": hour * 60 + minute,
            "return_pct": (hour * 60 + minute) / 100.0,
        }
        for hour in range(3)
        for minute in range(60)
    ]

    response = build_equity_history_response(history, "24h", "1h")

    assert response["total_points"] == 180
    assert response["filtered_points"] == 180
    assert response["returned_points"] == 4
    assert [point["equity"] for point in response["points"]] == [
        10_000,
        10_059,
        10_119,
        10_179,
    ]


def test_equity_history_auto_interval_limits_large_series() -> None:
    history = [
        {
            "timestamp": datetime.fromtimestamp(index * 60, tz=timezone.utc).isoformat(),
            "equity": 10_000.0,
            "total_pnl": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "return_pct": 0.0,
        }
        for index in range(10_000)
    ]

    response = build_equity_history_response(history, "all", "auto")

    assert response["interval"] == "15m"
    assert response["returned_points"] < 1_500


def test_provider_migrates_legacy_json_without_truncating(tmp_path: Path) -> None:
    history_path = tmp_path / "equity.json"
    legacy = [
        {
            "timestamp": datetime.fromtimestamp(index * 60, tz=timezone.utc).isoformat(),
            "equity": 10_000.0 + index,
            "total_pnl": float(index),
            "realized_pnl": 0.0,
            "unrealized_pnl": float(index),
            "return_pct": index / 100.0,
        }
        for index in range(2_000)
    ]
    history_path.write_text(json.dumps(legacy), encoding="utf-8")
    provider = LiveDashboardProvider(
        database_path=tmp_path / "trades.sqlite",
        signal_path=tmp_path / "signals.parquet",
        status_path=tmp_path / "status.json",
        history_path=history_path,
        price_loader=lambda _: {},
    )

    response = provider.equity_history("all", "raw")

    assert response["total_points"] == 2_000
    assert response["returned_points"] == 2_000
    assert provider.history_database_path.exists()
