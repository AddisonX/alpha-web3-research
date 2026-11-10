from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.live.safety import check_live_signal_status
from src.live.signals import HOUR_MS, build_signal_timeline
from src.strategies.lightgbm import rank_long_only_scores

STRATEGIES = Path(__file__).resolve().parents[1] / "freqtrade" / "strategies"
sys.path.insert(0, str(STRATEGIES))
live_module = importlib.import_module("AlphaCrossSectionalLiveLongOnlyStrategy")
LiveStrategy = live_module.AlphaCrossSectionalLiveLongOnlyStrategy

NOW = datetime(2026, 7, 27, 12, 5, tzinfo=timezone.utc)
DECISION = int(datetime(2026, 7, 27, tzinfo=timezone.utc).timestamp() * 1000)
PAIR = "COIN59/USDT:USDT"
OLD_PAIR = "OLD/USDT:USDT"


@pytest.fixture
def live_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict]:
    pairs = [f"COIN{index:02d}/USDT:USDT" for index in range(60)]
    ranked = rank_long_only_scores(pairs, np.arange(60, dtype=float), 0.10)
    timeline = build_signal_timeline(
        ranked,
        DECISION,
        "2026-07-27T04:01:00+00:00",
        4 * HOUR_MS,
        24,
        "test",
    )
    signal_path = tmp_path / "alpha_live_long_only_signals.parquet"
    status_path = tmp_path / "alpha_live_long_only_status.json"
    timeline.write_parquet(signal_path)
    selected = ranked.filter(pl.col("target_position") == 1).get_column("pair").to_list()
    status = {
        "state": "healthy",
        "generated_at": "2026-07-27T04:01:00+00:00",
        "decision_timestamp": DECISION,
        "grace_valid_until": DECISION + 28 * HOUR_MS,
        "selected_pairs": selected,
        "selected_count": len(selected),
        "cross_section_count": 60,
    }
    status_path.write_text(json.dumps(status), encoding="utf-8")
    monkeypatch.setenv("ALPHA_FREQTRADE_LIVE_SIGNAL_PATH", str(signal_path))
    monkeypatch.setenv("ALPHA_FREQTRADE_LIVE_STATUS_PATH", str(status_path))
    monkeypatch.setattr(LiveStrategy, "_signals", None)
    monkeypatch.setattr(LiveStrategy, "_signals_mtime_ns", None)
    monkeypatch.setattr(LiveStrategy, "_validated_key", None)
    monkeypatch.setattr(LiveStrategy, "_last_health_reason", None)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz else NOW.replace(tzinfo=None)

    monkeypatch.setattr(live_module, "datetime", FixedDateTime)
    return signal_path, status_path, status


def _frame(strategy: LiveStrategy, pair: str = PAIR) -> pd.DataFrame:
    candle = NOW.replace(hour=8, minute=0)
    frame = pd.DataFrame({"date": [candle], "volume": [100.0]})
    return strategy.populate_indicators(frame, {"pair": pair})


def _confirm(strategy: LiveStrategy, pair: str = PAIR) -> bool:
    return strategy.confirm_trade_entry(pair, "limit", 1.0, 1.0, "gtc", NOW, None, "long")


def test_status_fails_closed_on_expiry_and_bad_counts(live_files: tuple[Path, Path, dict]) -> None:
    _, _, status = live_files
    assert check_live_signal_status(status, NOW).ready
    assert not check_live_signal_status(status, datetime(2026, 7, 28, 4, tzinfo=timezone.utc)).ready
    assert not check_live_signal_status({**status, "selected_count": 99}, NOW).ready
    assert not check_live_signal_status({**status, "state": "failed"}, NOW).ready


def test_restart_reconciles_selected_and_removed_targets(live_files: tuple[Path, Path, dict]) -> None:
    strategy = LiveStrategy({"dry_run": True})
    selected = _frame(strategy)
    assert strategy.populate_entry_trend(selected, {"pair": PAIR})["enter_long"].iloc[-1] == 1

    removed = _frame(strategy, OLD_PAIR)
    assert strategy.populate_exit_trend(removed, {"pair": OLD_PAIR})["exit_long"].iloc[-1] == 1


def test_bad_status_or_mismatched_parquet_freezes_model_trades(
    live_files: tuple[Path, Path, dict],
) -> None:
    signal_path, status_path, status = live_files
    strategy = LiveStrategy({"dry_run": True})
    status_path.write_text(json.dumps({**status, "state": "failed"}), encoding="utf-8")
    assert strategy.populate_entry_trend(_frame(strategy), {"pair": PAIR})["enter_long"].iloc[-1] == 0
    assert strategy.populate_exit_trend(_frame(strategy, OLD_PAIR), {"pair": OLD_PAIR})["exit_long"].iloc[-1] == 0

    status_path.write_text("[]", encoding="utf-8")
    assert strategy.populate_entry_trend(_frame(strategy), {"pair": PAIR})["enter_long"].iloc[-1] == 0

    status_path.write_text(
        json.dumps(
            {**status, "selected_pairs": [f"OTHER{i}/USDT:USDT" for i in range(6)]}
        ),
        encoding="utf-8",
    )
    assert strategy.populate_exit_trend(_frame(strategy, OLD_PAIR), {"pair": OLD_PAIR})["exit_long"].iloc[-1] == 0

    status_path.write_text(json.dumps(status), encoding="utf-8")
    signal_path.write_bytes(b"broken parquet")
    assert strategy.populate_entry_trend(_frame(strategy), {"pair": PAIR})["enter_long"].iloc[-1] == 0
    assert strategy.populate_exit_trend(_frame(strategy, OLD_PAIR), {"pair": OLD_PAIR})["exit_long"].iloc[-1] == 0


def test_entry_callback_rechecks_signal_and_blocks_same_decision_reentry(
    live_files: tuple[Path, Path, dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from freqtrade.persistence import Trade

    strategy = LiveStrategy({"dry_run": True})
    strategy.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (_frame(strategy, pair), NOW))
    monkeypatch.setattr(Trade, "get_trades_proxy", lambda **kwargs: [])
    assert _confirm(strategy)
    monkeypatch.setattr(Trade, "get_trades_proxy", lambda **kwargs: [object()])
    assert not _confirm(strategy)

    _, status_path, status = live_files
    status_path.write_text(json.dumps({**status, "grace_valid_until": DECISION + HOUR_MS}), encoding="utf-8")
    assert not _confirm(strategy)


def test_real_money_entry_is_hard_blocked(live_files: tuple[Path, Path, dict]) -> None:
    strategy = LiveStrategy({"dry_run": False})
    assert not _confirm(strategy)


def test_unexpected_entry_check_error_fails_closed(
    live_files: tuple[Path, Path, dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = LiveStrategy({"dry_run": True})

    def fail(*args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(strategy, "_health", fail)
    assert not _confirm(strategy)


def test_optional_audit_gate_blocks_only_new_entries(
    live_files: tuple[Path, Path, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from freqtrade.persistence import Trade

    strategy = LiveStrategy({"dry_run": True})
    strategy.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (_frame(strategy, pair), NOW))
    monkeypatch.setattr(Trade, "get_trades_proxy", lambda **kwargs: [])
    audit_path = tmp_path / "audit.json"
    monkeypatch.setenv("ALPHA_LIVE_AUDIT_PATH", str(audit_path))
    assert not _confirm(strategy)

    report = {
        "schema_version": 1,
        "generated_at": NOW.isoformat(),
        "state": "blocked",
        "signal_decision_timestamp": DECISION,
        "entry_allowed": False,
        "target_pairs": sorted(live_files[2]["selected_pairs"]),
        "issues": [],
    }
    audit_path.write_text(json.dumps(report), encoding="utf-8")
    assert not _confirm(strategy)
    assert strategy.populate_exit_trend(_frame(strategy, OLD_PAIR), {"pair": OLD_PAIR})["exit_long"].iloc[-1] == 1

    report["entry_allowed"] = True
    report["state"] = "healthy"
    report["signal_decision_timestamp"] = DECISION - 24 * HOUR_MS
    audit_path.write_text(json.dumps(report), encoding="utf-8")
    assert not _confirm(strategy)

    report["signal_decision_timestamp"] = DECISION
    report["generated_at"] = (NOW - pd.Timedelta(minutes=4)).isoformat()
    audit_path.write_text(json.dumps(report), encoding="utf-8")
    assert not _confirm(strategy)

    report["generated_at"] = NOW.isoformat()
    report["target_pairs"] = ["OTHER/USDT:USDT"]
    audit_path.write_text(json.dumps(report), encoding="utf-8")
    assert not _confirm(strategy)

    report["target_pairs"] = sorted(live_files[2]["selected_pairs"])
    audit_path.write_text(json.dumps(report), encoding="utf-8")
    assert _confirm(strategy)
