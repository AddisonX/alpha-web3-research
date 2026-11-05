from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

import pandas as pd

from alpha_web3_research.live.safety import check_live_signal_status, validate_signal_rows

logger = logging.getLogger(__name__)

SIGNAL_COLUMNS = [
    "date",
    "pair",
    "decision_timestamp",
    "signal_valid_until",
    "signal_valid",
    "target_position",
]


def _issue(code: str, severity: str, pairs: list[str] | None = None) -> dict[str, Any]:
    return {"code": code, "severity": severity, "pairs": sorted(pairs or [])}


def _read_database(database_path: Path, decision_close: datetime | None) -> dict[str, Any]:
    uri = database_path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        open_trades = connection.execute(
            "SELECT pair, is_short FROM trades WHERE is_open = 1"
        ).fetchall()
        open_orders = connection.execute(
            "SELECT id, ft_pair, ft_order_side, order_date FROM orders WHERE ft_is_open = 1"
        ).fetchall()
        recently_closed = []
        if decision_close is not None:
            recently_closed = connection.execute(
                "SELECT DISTINCT pair FROM trades WHERE is_open = 0 AND close_date >= ?",
                (decision_close.replace(tzinfo=None).isoformat(sep=" "),),
            ).fetchall()
    return {
        "open_trades": [(str(pair), bool(is_short)) for pair, is_short in open_trades],
        "open_orders": [
            {"id": int(order_id), "pair": str(pair), "side": str(side), "order_date": date}
            for order_id, pair, side, date in open_orders
        ],
        "recently_closed_pairs": {str(row[0]) for row in recently_closed},
    }


def audit_dry_run(
    status_path: Path,
    signal_path: Path,
    database_path: Path,
    now: datetime | None = None,
    grace_minutes: int = 15,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if grace_minutes < 0:
        raise ValueError("grace_minutes must be nonnegative")

    issues: list[dict[str, Any]] = []
    health = None
    status: dict[str, Any] = {}
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not isinstance(status, dict):
            raise ValueError("status must be a JSON object")
        health = check_live_signal_status(status, now)
        if not health.ready:
            issues.append(_issue("signal_unhealthy", "block"))
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Cannot read live signal status: %s", exc)
        issues.append(_issue("signal_status_unreadable", "block"))

    if health is not None and health.ready:
        try:
            signals = pd.read_parquet(signal_path, columns=SIGNAL_COLUMNS)
            signals["date"] = pd.to_datetime(signals["date"], utc=True)
            error = validate_signal_rows(signals, health)
            if error is not None:
                logger.warning("Live signal file mismatch: %s", error)
                issues.append(_issue("signal_file_mismatch", "block"))
        except Exception as exc:
            logger.warning("Cannot read live signal file: %s", exc)
            issues.append(_issue("signal_file_unreadable", "block"))

    decision_close = None
    if health is not None and health.ready and health.decision_timestamp is not None:
        decision_close = datetime.fromtimestamp(
            (health.decision_timestamp + 4 * 3_600_000) / 1000, timezone.utc
        )
    database = None
    try:
        database = _read_database(database_path, decision_close)
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        logger.warning("Cannot read Freqtrade dry-run database: %s", exc)
        issues.append(_issue("database_unreadable", "block"))

    open_pairs: set[str] = set()
    missing_pairs: set[str] = set()
    unexpected_pairs: set[str] = set()
    exempt_pairs: set[str] = set()
    open_orders: list[dict[str, Any]] = []
    if database is not None:
        open_trades = database["open_trades"]
        counts = Counter(pair for pair, _ in open_trades)
        open_pairs = set(counts)
        duplicate_pairs = [pair for pair, count in counts.items() if count > 1]
        short_pairs = [pair for pair, is_short in open_trades if is_short]
        if duplicate_pairs:
            issues.append(_issue("duplicate_open_trades", "block", duplicate_pairs))
        if short_pairs:
            issues.append(_issue("unexpected_short_trades", "block", short_pairs))

        for order in database["open_orders"]:
            item = dict(order)
            try:
                order_date = datetime.fromisoformat(str(item.pop("order_date")))
                if order_date.tzinfo is None:
                    order_date = order_date.replace(tzinfo=timezone.utc)
                age_seconds = (now - order_date).total_seconds()
            except ValueError:
                age_seconds = float("inf")
            item["age_seconds"] = None if age_seconds == float("inf") else int(age_seconds)
            open_orders.append(item)
        stale_orders = [
            order["pair"]
            for order in open_orders
            if order["age_seconds"] is None
            or order["age_seconds"] < -60
            or order["age_seconds"] > grace_minutes * 60
        ]
        if stale_orders:
            issues.append(_issue("stale_open_orders", "block", stale_orders))

        if health is not None and health.ready and not any(
            issue["code"].startswith("signal_file_") for issue in issues
        ):
            target_pairs = health.selected_pairs
            exempt_pairs = database["recently_closed_pairs"] & target_pairs
            missing_pairs = target_pairs - open_pairs - exempt_pairs
            unexpected_pairs = open_pairs - target_pairs
            generated_at = datetime.fromisoformat(str(status["generated_at"]))
            grace_elapsed = now >= generated_at + timedelta(minutes=grace_minutes)
            if grace_elapsed and unexpected_pairs:
                issues.append(_issue("unexpected_open_positions", "block", list(unexpected_pairs)))
            if grace_elapsed and missing_pairs:
                issues.append(_issue("missing_target_positions", "warning", list(missing_pairs)))

    blocked = any(issue["severity"] == "block" for issue in issues)
    state = "blocked" if blocked else "degraded" if issues else "healthy"
    report = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "state": state,
        "entry_allowed": not blocked,
        "signal_decision_timestamp": health.decision_timestamp if health and health.ready else None,
        "target_pairs": sorted(health.selected_pairs) if health and health.ready else [],
        "open_pairs": sorted(open_pairs),
        "missing_pairs": sorted(missing_pairs),
        "unexpected_pairs": sorted(unexpected_pairs),
        "recently_closed_exempt_pairs": sorted(exempt_pairs),
        "open_orders": open_orders,
        "issues": issues,
    }
    report["fingerprint"] = _fingerprint(report)
    return report


def _fingerprint(report: dict[str, Any]) -> str:
    payload = {"state": report["state"], "issues": report["issues"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _send_webhook(webhook_url: str, report: dict[str, Any]) -> None:
    payload = json.dumps(
        {
            "event": "alpha_live_safety",
            "generated_at": report["generated_at"],
            "state": report["state"],
            "entry_allowed": report["entry_allowed"],
            "issues": report["issues"],
            "target_pairs": report["target_pairs"],
            "open_pairs": report["open_pairs"],
        }
    ).encode("utf-8")
    request = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"alert webhook returned HTTP {response.status}")


def publish_audit(
    report: dict[str, Any],
    report_path: Path,
    webhook_url: str | None = None,
    alert_sender: Callable[[str, dict[str, Any]], None] = _send_webhook,
) -> dict[str, Any]:
    previous: dict[str, Any] = {}
    try:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(previous, dict):
            previous = {}
    except (OSError, ValueError, TypeError):
        pass
    channel = "webhook" if webhook_url else "journal"
    changed = previous.get("fingerprint") != report["fingerprint"]
    retry = previous.get("alert_delivered") is False
    should_alert = (changed or retry or previous.get("alert_channel") != channel) and (
        report["state"] != "healthy" or previous.get("state") not in (None, "healthy")
    )
    report = dict(report)
    report["alert_channel"] = channel
    report["alert_delivered"] = True
    if should_alert:
        try:
            if webhook_url:
                alert_sender(webhook_url, report)
            else:
                logger.warning("Live safety audit changed: %s %s", report["state"], report["issues"])
        except Exception as exc:
            logger.error("Live safety alert delivery failed: %s", type(exc).__name__)
            report["alert_delivered"] = False
            report["issues"] = [*report["issues"], _issue("alert_delivery_failed", "block")]
            report["state"] = "blocked"
            report["entry_allowed"] = False
            report["fingerprint"] = _fingerprint(report)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    return report
