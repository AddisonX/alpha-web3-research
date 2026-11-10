"""Read-only discovery of local Freqtrade instances; no strategy imports."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class MonitoredInstance:
    id: str
    directory: Path
    strategy_id: str
    display_name: str
    starting_balance: float
    execution_strategy: str

    def metadata(self) -> dict:
        runtime = read_object(self.directory / "runtime/runtime_status.json")
        checked_at = runtime.get("checked_at")
        fresh = False
        try:
            checked = datetime.fromisoformat(str(checked_at).replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - checked).total_seconds()
            fresh = 0 <= age <= 180
        except (ValueError, TypeError):
            pass
        if fresh and runtime.get("state") == "running":
            try:
                for key in ("supervisor_pid", "worker_pid"):
                    pid = int(runtime.get(key, 0))
                    if pid <= 1:
                        raise ValueError("missing runtime PID")
                    os.kill(pid, 0)
            except (OSError, ValueError, TypeError):
                fresh = False
        return {
            "id": self.id,
            "display_name": self.display_name,
            "strategy_id": self.strategy_id,
            "mode": "dry_run",
            "runtime_state": runtime.get("state", "unknown") if fresh else "unknown",
            "runtime_updated_at": checked_at,
            "audit_state": runtime.get("audit_state", "unknown") if fresh else "unknown",
            "entry_allowed": runtime.get("entry_allowed", False) if fresh else False,
            "database_ready": (self.directory / "database/trades.dryrun.sqlite").is_file(),
        }


def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def discover_instances(root: Path) -> tuple[dict[str, MonitoredInstance], list[dict]]:
    base = root / "freqtrade/instances"
    instances: dict[str, MonitoredInstance] = {}
    errors = []
    if not base.is_dir():
        return instances, errors
    for directory in sorted(base.iterdir()):
        if directory.name.startswith(".") or not directory.is_dir():
            continue
        if directory.is_symlink() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", directory.name):
            errors.append({"id": directory.name, "error": "unsupported instance directory"})
            continue
        try:
            manifest = read_object(directory / "instance.json")
            config = read_object(directory / "config.json")
            if manifest.get("layout") != "structured" or not manifest.get("strategy_id"):
                raise ValueError("missing structured instance manifest")
            if config.get("dry_run") is not True:
                raise ValueError("only dry-run instances are supported")
            if config.get("exchange", {}).get("name") != "binance" or config.get("stake_currency") != "USDT":
                raise ValueError("requires Binance USDT instance")
            if config.get("exchange", {}).get("sandbox") or config.get("trading_mode") != "futures":
                raise ValueError("requires mainnet futures market data")
            database = directory / "database/trades.dryrun.sqlite"
            url = config.get("db_url", "")
            if not url.startswith("sqlite:///"):
                raise ValueError("requires instance SQLite database")
            configured = Path(url.removeprefix("sqlite:///"))
            if not configured.is_absolute():
                configured = root / configured
            if configured.resolve() != database.resolve():
                raise ValueError("database must belong to this instance")
            for relative in ("database", "database/trades.dryrun.sqlite", "database/equity_history.sqlite", "signals", "runtime", "config.json", "instance.json"):
                if not (directory / relative).resolve().is_relative_to(directory.resolve()):
                    raise ValueError("instance paths cannot escape their directory")
            wallet = config.get("dry_run_wallet", 10000)
            if isinstance(wallet, bool) or not isinstance(wallet, (int, float)) or not math.isfinite(wallet) or wallet <= 0:
                raise ValueError("requires positive numeric dry_run_wallet")
            instances[directory.name] = MonitoredInstance(
                id=directory.name,
                directory=directory,
                strategy_id=str(manifest["strategy_id"]),
                display_name=str(manifest.get("display_name") or directory.name),
                starting_balance=float(wallet),
                execution_strategy=str(config.get("strategy", "unknown")),
            )
        except (ValueError, TypeError, AttributeError) as error:
            errors.append({"id": directory.name, "error": str(error)})
    return instances, errors
