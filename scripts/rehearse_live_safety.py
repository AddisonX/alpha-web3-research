from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alpha_web3_research.live.reconciliation import audit_dry_run
from alpha_web3_research.live.safety import check_entry_audit


def _copy_database(source: Path, destination: Path) -> None:
    uri = source.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as original:
        with sqlite3.connect(destination) as copied:
            original.backup(copied)


def _update_status(path: Path, **changes: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def rehearse(
    status_path: Path,
    signal_path: Path,
    database_path: Path,
    output_root: Path,
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now(timezone.utc)
    baseline = audit_dry_run(status_path, signal_path, database_path, now=now)
    if baseline["state"] != "healthy" or not baseline["entry_allowed"]:
        raise RuntimeError(f"baseline must be healthy before fault injection: {baseline['issues']}")
    if not baseline["open_pairs"]:
        raise RuntimeError("baseline needs open dry-run trades for restart/position checks")

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output_root))
    source_status = json.loads(status_path.read_text(encoding="utf-8"))
    generated_at = datetime.fromisoformat(source_status["generated_at"])
    after_grace = generated_at + timedelta(minutes=16)
    if after_grace >= datetime.fromtimestamp(
        source_status["grace_valid_until"] / 1000, timezone.utc
    ):
        raise RuntimeError("signal expires before the 15-minute reconciliation grace ends")

    scenarios = [
        ("status_failed", "blocked", "signal_unhealthy"),
        ("signal_expired", "blocked", "signal_unhealthy"),
        ("parquet_corrupt", "blocked", "signal_file_unreadable"),
        ("unexpected_position", "blocked", "unexpected_open_positions"),
        ("stale_order", "blocked", "stale_open_orders"),
        ("missing_target", "degraded", "missing_target_positions"),
    ]
    results = []
    for name, expected_state, expected_issue in scenarios:
        directory = run_dir / name
        directory.mkdir()
        case_status = directory / "status.json"
        case_signal = directory / "signals.parquet"
        case_database = directory / "trades.sqlite"
        shutil.copyfile(status_path, case_status)
        shutil.copyfile(signal_path, case_signal)
        _copy_database(database_path, case_database)

        case_time = after_grace if name in {"unexpected_position", "missing_target"} else now
        if name == "status_failed":
            _update_status(case_status, state="failed")
        elif name == "signal_expired":
            _update_status(case_status, grace_valid_until=int(now.timestamp() * 1000))
        elif name == "parquet_corrupt":
            case_signal.write_bytes(b"corrupted rehearsal fixture")
        elif name == "unexpected_position":
            with sqlite3.connect(case_database) as connection:
                connection.execute(
                    "UPDATE trades SET pair = 'UNEXPECTED/USDT:USDT' "
                    "WHERE rowid = (SELECT rowid FROM trades WHERE is_open = 1 LIMIT 1)"
                )
        elif name == "stale_order":
            with sqlite3.connect(case_database) as connection:
                connection.execute(
                    "UPDATE orders SET ft_is_open = 1, order_date = ? "
                    "WHERE rowid = (SELECT rowid FROM orders LIMIT 1)",
                    ((now - timedelta(minutes=30)).replace(tzinfo=None).isoformat(sep=" "),),
                )
        elif name == "missing_target":
            before_decision = datetime.fromtimestamp(
                (baseline["signal_decision_timestamp"] + 4 * 3_600_000) / 1000, timezone.utc
            ) - timedelta(days=1)
            with sqlite3.connect(case_database) as connection:
                connection.execute(
                    "UPDATE trades SET is_open = 0, close_date = ? "
                    "WHERE rowid = (SELECT rowid FROM trades WHERE is_open = 1 LIMIT 1)",
                    (before_decision.replace(tzinfo=None).isoformat(sep=" "),),
                )

        report = audit_dry_run(case_status, case_signal, case_database, now=case_time)
        issues = {issue["code"] for issue in report["issues"]}
        passed = report["state"] == expected_state and expected_issue in issues
        if expected_state == "blocked":
            passed = passed and not report["entry_allowed"]
        else:
            passed = passed and report["entry_allowed"]
        results.append({"scenario": name, "passed": passed, "state": report["state"], "issues": sorted(issues)})
        (directory / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    decision = int(baseline["signal_decision_timestamp"])
    targets = frozenset(baseline["target_pairs"])
    blocked_report = json.loads(
        (run_dir / "unexpected_position" / "audit.json").read_text(encoding="utf-8")
    )
    blocked_report["generated_at"] = now.isoformat()
    gate_checks = {}
    for name, report in {
        "healthy": baseline,
        "stale": {**baseline, "generated_at": (now - timedelta(seconds=181)).isoformat()},
        "wrong_targets": {**baseline, "target_pairs": ["OTHER/USDT:USDT"]},
        "blocked": blocked_report,
    }.items():
        allowed, _ = check_entry_audit(report, now, decision, targets)
        gate_checks[name] = allowed

    summary = {
        "mode": "isolated_dry_run_fixture",
        "generated_at": now.isoformat(),
        "baseline_open_count": len(baseline["open_pairs"]),
        "baseline_state": baseline["state"],
        "scenarios": results,
        "gate_checks": gate_checks,
    }
    expected_gates = {"healthy": True, "stale": False, "wrong_targets": False, "blocked": False}
    summary["passed"] = all(result["passed"] for result in results) and gate_checks == expected_gates
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rehearse safety faults against copies of a dry-run instance.")
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--signal-path", type=Path, required=True)
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary_path = rehearse(args.status_path, args.signal_path, args.database_path, args.output_root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for result in summary["scenarios"]:
        print(f"{result['scenario']}: {result['state']} ({'PASS' if result['passed'] else 'FAIL'})")
    print(f"gate checks: {summary['gate_checks']}")
    print(f"summary: {summary_path}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
