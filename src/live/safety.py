from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import polars as pl
from src.strategies.base import validate_signed_weights


@dataclass(frozen=True)
class LiveSignalHealth:
    ready: bool
    reason: str
    decision_timestamp: int | None = None
    valid_until: int | None = None
    selected_pairs: frozenset[str] = frozenset()
    cross_section_count: int | None = None
    allow_short: bool = False
    short_pairs: frozenset[str] = frozenset()


def check_live_signal_status(status: dict[str, Any], now: datetime, *, allow_short: bool = False) -> LiveSignalHealth:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if status.get("state") != "healthy":
        return LiveSignalHealth(False, "signal status is missing or unhealthy")

    try:
        decision = int(status["decision_timestamp"])
        valid_until = int(status["grace_valid_until"])
        generated_at = datetime.fromisoformat(str(status["generated_at"]))
        selected = status["selected_pairs"]
        selected_count = int(status["selected_count"])
        cross_section_count = int(status["cross_section_count"])
    except (KeyError, TypeError, ValueError):
        return LiveSignalHealth(False, "signal status is incomplete")

    if generated_at.tzinfo is None or not isinstance(selected, list):
        return LiveSignalHealth(False, "signal status has invalid types")
    if any(not isinstance(pair, str) or not pair for pair in selected):
        return LiveSignalHealth(False, "signal status has invalid pairs")
    pairs = frozenset(selected)
    shorts = status.get("short_pairs", [])
    signed = status.get("allow_short", False) is True
    if signed and not allow_short:
        return LiveSignalHealth(False, "short execution is not enabled")
    if not isinstance(shorts, list) or any(not isinstance(p, str) for p in shorts):
        return LiveSignalHealth(False, "invalid short targets")
    if len(set(shorts)) != len(shorts) or not set(shorts).issubset(pairs):
        return LiveSignalHealth(False, "invalid short targets")
    if (shorts and not signed) or (signed and (not shorts or len(shorts) == len(pairs))):
        return LiveSignalHealth(False, "invalid short allocation")
    if len(pairs) != selected_count or selected_count < 1 or cross_section_count < 60:
        return LiveSignalHealth(False, "signal universe or selection count is invalid")

    now_ms = int(now.astimezone(timezone.utc).timestamp() * 1000)
    generated_ms = int(generated_at.timestamp() * 1000)
    if signed and generated_ms < decision + 4 * 3_600_000:
        return LiveSignalHealth(False, "decision candle has not closed")
    if decision > generated_ms or generated_ms > now_ms:
        return LiveSignalHealth(False, "signal generation time is invalid")
    if now_ms >= valid_until or decision >= valid_until:
        return LiveSignalHealth(False, "signal has expired")
    return LiveSignalHealth(True, "healthy", decision, valid_until, pairs, cross_section_count, signed, frozenset(shorts))


def validate_signal_rows(signals: pd.DataFrame, health: LiveSignalHealth) -> str | None:
    if not health.ready or health.decision_timestamp is None:
        return "signal status is not healthy"
    if signals.duplicated(subset=["date", "pair"]).any():
        return "signal file contains duplicate date/pair rows"
    decision_date = pd.Timestamp(health.decision_timestamp, unit="ms", tz="UTC")
    current = signals.loc[signals["decision_timestamp"] == health.decision_timestamp]
    timeframe_ms = 4 * 3_600_000
    duration = health.valid_until - health.decision_timestamp
    if duration <= 0 or duration % timeframe_ms:
        return "signal file has an invalid decision interval"
    expected_dates = set(pd.date_range(decision_date, periods=duration // timeframe_ms, freq="4h"))
    if len(current) != health.cross_section_count * len(expected_dates):
        return "signal file and status have different decisions or targets"
    if set(current["date"]) != expected_dates:
        return "signal file has missing or unexpected candle times"
    if not current["signal_valid"].all():
        return "decision rows are not valid"
    if not current["signal_valid_until"].eq(health.valid_until).all():
        return "signal file and status have different expiry times"
    if not current["target_position"].isin([-1, 0, 1] if health.allow_short else [0, 1]).all():
        return "live signal file contains non-long targets"
    decision_pairs = frozenset(current.loc[current["date"] == decision_date, "pair"])
    for _, rows in current.groupby("date"):
        selected = frozenset(rows.loc[rows["target_position"] != 0, "pair"])
        if health.allow_short:
            try:
                validate_signed_weights(pl.from_pandas(rows))
            except (ValueError, TypeError):
                return "invalid signed target weights"
            if frozenset(rows.loc[rows.target_position == -1, "pair"]) != health.short_pairs:
                return "short targets differ from status"
        if frozenset(rows["pair"]) != decision_pairs or selected != health.selected_pairs:
            return "signal file and status have different decisions or targets"
    return None


def check_entry_audit(
    report: dict[str, Any],
    now: datetime,
    decision_timestamp: int,
    selected_pairs: frozenset[str],
    max_age_seconds: int = 180,
) -> tuple[bool, str]:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    try:
        generated_at = datetime.fromisoformat(str(report["generated_at"]))
        report_decision = int(report["signal_decision_timestamp"])
    except (KeyError, TypeError, ValueError):
        return False, "audit report is incomplete"
    if generated_at.tzinfo is None:
        return False, "audit report has no timezone"
    age = (now - generated_at).total_seconds()
    if age < 0 or age > max_age_seconds:
        return False, "audit report is stale or from the future"
    if report_decision != decision_timestamp:
        return False, "audit report is for a different model decision"
    targets = report.get("target_pairs")
    if not isinstance(targets, list) or any(not isinstance(pair, str) for pair in targets):
        return False, "audit report has an invalid target list"
    if frozenset(targets) != selected_pairs:
        return False, "audit report has a different target list"
    if report.get("schema_version") != 1 or report.get("state") not in ("healthy", "degraded"):
        return False, "audit report has an invalid version or state"
    issues = report.get("issues")
    if not isinstance(issues, list) or any(not isinstance(issue, dict) for issue in issues):
        return False, "audit report has invalid issues"
    if report.get("entry_allowed") is not True or any(issue.get("severity") == "block" for issue in issues):
        return False, "audit report blocks new entries"
    return True, "healthy"
