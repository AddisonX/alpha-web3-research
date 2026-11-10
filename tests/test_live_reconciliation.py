from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from src.cli import app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rehearse_live_safety import rehearse
from src.live.reconciliation import audit_dry_run, publish_audit
from src.live.signals import HOUR_MS, build_signal_timeline
from src.strategies.lightgbm import rank_long_only_scores

NOW = datetime(2026, 7, 27, 12, 5, tzinfo=timezone.utc)
DECISION = int(datetime(2026, 7, 27, tzinfo=timezone.utc).timestamp() * 1000)


@pytest.fixture
def audit_files(tmp_path: Path) -> dict:
    pairs = [f"COIN{i:02d}/USDT:USDT" for i in range(60)]
    ranked = rank_long_only_scores(pairs, np.arange(60, dtype=float), 0.1)
    selected = ranked.filter(pl.col("target_position") == 1).get_column("pair").to_list()
    timeline = build_signal_timeline(
        ranked, DECISION, "2026-07-27T04:01:00+00:00", 4 * HOUR_MS, 24, "test"
    )
    signal_path = tmp_path / "signals.parquet"
    status_path = tmp_path / "status.json"
    database_path = tmp_path / "trades.sqlite"
    timeline.write_parquet(signal_path)
    status_path.write_text(
        json.dumps(
            {
                "state": "healthy",
                "generated_at": "2026-07-27T04:01:00+00:00",
                "decision_timestamp": DECISION,
                "grace_valid_until": DECISION + 28 * HOUR_MS,
                "selected_pairs": selected,
                "selected_count": len(selected),
                "cross_section_count": 60,
            }
        ),
        encoding="utf-8",
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE trades (pair TEXT, is_open INTEGER, is_short INTEGER, close_date TEXT)"
        )
        connection.execute(
            "CREATE TABLE orders (id INTEGER, ft_pair TEXT, ft_order_side TEXT, "
            "order_date TEXT, ft_is_open INTEGER)"
        )
        connection.executemany(
            "INSERT INTO trades VALUES (?, 1, 0, NULL)", [(pair,) for pair in selected]
        )
    return {
        "status_path": status_path,
        "signal_path": signal_path,
        "database_path": database_path,
        "selected": selected,
    }


def _audit(files: dict, now: datetime = NOW) -> dict:
    return audit_dry_run(
        files["status_path"], files["signal_path"], files["database_path"], now=now
    )


def _codes(report: dict) -> set[str]:
    return {issue["code"] for issue in report["issues"]}


def test_healthy_and_rebalance_grace(audit_files: dict) -> None:
    assert _audit(audit_files)["state"] == "healthy"
    with sqlite3.connect(audit_files["database_path"]) as connection:
        connection.execute("DELETE FROM trades")
        connection.execute(
            "INSERT INTO trades VALUES ('OLD/USDT:USDT', 1, 0, NULL)"
        )
    during_grace = _audit(audit_files, datetime(2026, 7, 27, 4, 10, tzinfo=timezone.utc))
    assert during_grace["state"] == "healthy"
    assert during_grace["entry_allowed"]


def test_missing_targets_warn_but_do_not_block_recovery(audit_files: dict) -> None:
    with sqlite3.connect(audit_files["database_path"]) as connection:
        connection.execute("DELETE FROM trades WHERE pair = ?", (audit_files["selected"][0],))
    report = _audit(audit_files)
    assert report["state"] == "degraded"
    assert report["entry_allowed"]
    assert "missing_target_positions" in _codes(report)


def test_recently_closed_target_does_not_count_as_missing(audit_files: dict) -> None:
    pair = audit_files["selected"][0]
    with sqlite3.connect(audit_files["database_path"]) as connection:
        connection.execute("DELETE FROM trades WHERE pair = ?", (pair,))
        connection.execute(
            "INSERT INTO trades VALUES (?, 0, 0, '2026-07-27 06:00:00')", (pair,)
        )
    report = _audit(audit_files)
    assert report["state"] == "healthy"
    assert report["recently_closed_exempt_pairs"] == [pair]


def test_unexpected_short_duplicate_or_stale_order_blocks(audit_files: dict) -> None:
    with sqlite3.connect(audit_files["database_path"]) as connection:
        connection.execute("INSERT INTO trades VALUES ('OLD/USDT:USDT', 1, 0, NULL)")
        connection.execute(
            "INSERT INTO trades VALUES (?, 1, 1, NULL)", (audit_files["selected"][0],)
        )
        connection.execute(
            "INSERT INTO orders VALUES (1, ?, 'buy', '2026-07-27 11:40:00', 1)",
            (audit_files["selected"][1],),
        )
    report = _audit(audit_files)
    assert report["state"] == "blocked"
    assert not report["entry_allowed"]
    assert {
        "unexpected_open_positions",
        "unexpected_short_trades",
        "duplicate_open_trades",
        "stale_open_orders",
    } <= _codes(report)


def test_future_dated_open_order_blocks(audit_files: dict) -> None:
    with sqlite3.connect(audit_files["database_path"]) as connection:
        connection.execute(
            "INSERT INTO orders VALUES (1, ?, 'buy', '2026-07-27 12:10:00', 1)",
            (audit_files["selected"][0],),
        )
    report = _audit(audit_files)
    assert "stale_open_orders" in _codes(report)
    assert not report["entry_allowed"]


def test_bad_inputs_fail_closed(audit_files: dict) -> None:
    audit_files["signal_path"].write_bytes(b"not parquet")
    report = _audit(audit_files)
    assert "signal_file_unreadable" in _codes(report)
    assert not report["entry_allowed"]

    audit_files["status_path"].write_text("[]", encoding="utf-8")
    assert "signal_status_unreadable" in _codes(_audit(audit_files))

    audit_files["database_path"].unlink()
    assert "database_unreadable" in _codes(_audit(audit_files))


def test_truncated_signal_timeline_blocks_even_when_decision_rows_exist(audit_files: dict) -> None:
    signal_path = audit_files["signal_path"]
    frame = pl.read_parquet(signal_path)
    frame.filter(pl.col("timestamp") != DECISION + 8 * HOUR_MS).write_parquet(signal_path)
    report = _audit(audit_files)
    assert "signal_file_mismatch" in _codes(report)
    assert not report["entry_allowed"]


def test_alerts_on_change_and_recovery_and_retries_failed_delivery(
    audit_files: dict, tmp_path: Path,
) -> None:
    report_path = tmp_path / "audit.json"
    calls: list[str] = []

    def sender(url: str, report: dict) -> None:
        calls.append(report["state"])

    healthy = _audit(audit_files)
    publish_audit(healthy, report_path, "https://example.invalid/alert", sender)
    assert calls == []

    with sqlite3.connect(audit_files["database_path"]) as connection:
        connection.execute("INSERT INTO trades VALUES ('OLD/USDT:USDT', 1, 0, NULL)")
    blocked = _audit(audit_files)
    publish_audit(blocked, report_path, "https://example.invalid/alert", sender)
    publish_audit(blocked, report_path, "https://example.invalid/alert", sender)
    assert calls == ["blocked"]

    with sqlite3.connect(audit_files["database_path"]) as connection:
        connection.execute("DELETE FROM trades WHERE pair = 'OLD/USDT:USDT'")
    publish_audit(_audit(audit_files), report_path, "https://example.invalid/alert", sender)
    assert calls == ["blocked", "healthy"]

    def fail(url: str, report: dict) -> None:
        raise RuntimeError("webhook offline")

    failed = publish_audit(blocked, report_path, "https://example.invalid/alert", fail)
    assert "alert_delivery_failed" in _codes(failed)
    assert not failed["entry_allowed"]
    assert not failed["alert_delivered"]
    publish_audit(blocked, report_path, "https://example.invalid/alert", sender)
    assert calls[-1] == "blocked"

    report_path.write_text("[]", encoding="utf-8")
    publish_audit(blocked, report_path, "https://example.invalid/alert", sender)
    assert calls[-1] == "blocked"


def test_audit_cli_writes_report_and_uses_blocking_exit_code(
    audit_files: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.cli as cli

    original_audit = cli.audit_dry_run
    monkeypatch.setattr(cli, "audit_dry_run", lambda **kwargs: original_audit(now=NOW, **kwargs))
    report_path = tmp_path / "result.json"
    args = [
        "live", "audit-dry-run",
        "--status-path", str(audit_files["status_path"]),
        "--signal-path", str(audit_files["signal_path"]),
        "--database-path", str(audit_files["database_path"]),
        "--report-path", str(report_path),
    ]
    runner = CliRunner()
    assert runner.invoke(app, args).exit_code == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["entry_allowed"]

    with sqlite3.connect(audit_files["database_path"]) as connection:
        connection.execute("INSERT INTO trades VALUES ('OLD/USDT:USDT', 1, 0, NULL)")
    assert runner.invoke(app, args).exit_code == 2
    assert not json.loads(report_path.read_text(encoding="utf-8"))["entry_allowed"]


def test_rehearsal_uses_copies_and_passes_all_faults(audit_files: dict, tmp_path: Path) -> None:
    with sqlite3.connect(audit_files["database_path"]) as connection:
        connection.execute(
            "INSERT INTO orders VALUES (1, ?, 'buy', '2026-07-27 04:10:00', 0)",
            (audit_files["selected"][0],),
        )
    summary_path = rehearse(
        audit_files["status_path"],
        audit_files["signal_path"],
        audit_files["database_path"],
        tmp_path / "faults",
        now=NOW,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["passed"]
    assert len(summary["scenarios"]) == 6
    assert _audit(audit_files)["state"] == "healthy"
