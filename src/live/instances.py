"""Prepare or run an isolated, mainnet-public-data Freqtrade dry-run instance."""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import Settings
from src.data.exchange import create_exchange
from src.live.config import write_live_long_only_config
from src.live.reconciliation import audit_dry_run, publish_audit
from src.strategies.reversal import STRATEGY_ID
from src.live.paths import default_instance_name, instance_directory

ROOT = Path(__file__).resolve().parents[2]
INSTANCE = instance_directory(ROOT, STRATEGY_ID)


def instance_info(instance: Path) -> dict:
    path = instance / "instance.json"
    return json.loads(path.read_text())


def instance_path(instance: Path, kind: str) -> Path:
    folders = {"database": "database", "runtime": "runtime", "checks": "checks", "user_data": "user_data"}
    return instance / folders[kind]


def database_path(instance: Path) -> Path:
    return instance_path(instance, "database") / "trades.dryrun.sqlite"


def prepare_instance(root: Path = ROOT, source_pairs: Path | None = None, *,
                     strategy_id: str = STRATEGY_ID, instance_name: str | None = None,
                     structured: bool = True) -> Path:
    import re
    name = instance_name or default_instance_name(strategy_id)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
        raise ValueError("invalid instance name")
    instance = root / "freqtrade/instances" / name
    instance.mkdir(parents=True, exist_ok=True)
    manifest = instance / "instance.json"
    if not structured:
        raise ValueError("legacy instance layout is no longer supported")
    expected = {"strategy_id": strategy_id, "layout": "structured"}
    if manifest.exists() and instance_info(instance) != expected:
        raise ValueError("instance strategy or layout differs; do not migrate a running instance")
    if not manifest.exists():
        if structured and (instance / "config.json").exists():
            raise ValueError("existing legacy configuration requires an explicit offline migration")
        manifest.write_text(json.dumps(expected, indent=2) + "\n")
    for kind in ["database", "runtime", "checks", "user_data"]:
        instance_path(instance, kind).mkdir(exist_ok=True)
    (instance / "signals").mkdir(exist_ok=True)
    (instance / "logs").mkdir(exist_ok=True)
    config_path = instance / "config.json"
    if not config_path.exists():
        pairs_path = instance / "pairs.json"
        pairs_path.write_text((source_pairs or instance_directory(root) / "pairs.json").read_text())
        write_live_long_only_config(pairs_path, config_path, use_testnet_market_data=False)
        config = json.loads(config_path.read_text())
        config.update({
            "bot_name": strategy_id, "strategy": "AlphaCrossSectionalLiveLongOnlyStrategy",
            "db_url": "sqlite:///" + str(database_path(instance).resolve()),
            "telegram": {"enabled": False, "token": "", "chat_id": ""},
        })
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    config = json.loads(config_path.read_text())
    expected_db = "sqlite:///" + str(database_path(instance).resolve())
    if config.get("dry_run") is not True or config.get("db_url") != expected_db:
        raise ValueError("instance must be dry-run with its own database")
    if config.get("exchange", {}).get("key") or config.get("exchange", {}).get("secret"):
        raise ValueError("this public-data simulation must not contain exchange credentials")
    if "testnet" in json.dumps(config):
        raise ValueError("this instance requires mainnet public data")
    # This instance follows the user's existing proxy environment; no proxy URL is added.
    for key in ["ccxt_config", "ccxt_async_config"]:
        options = config["exchange"].setdefault(key, {})
        options.update({"trust_env": True, "requests_trust_env": True, "aiohttp_trust_env": True, "timeout": 15000})
        options.setdefault("options", {}).update({"fetchMarkets": {"types": ["linear"]}, "fetchCurrencies": False})
    config.setdefault("telegram", {}).update({"enabled": False, "token": "", "chat_id": ""})
    config.pop("api_server", None)  # Absent means disabled; avoid an incomplete RPC configuration.
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return instance


def environment(instance: Path) -> dict[str, str]:
    return {**os.environ,
        "ALPHA_CCXT_TRUST_ENV": "true",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ALPHA_FREQTRADE_LIVE_SIGNAL_PATH": str(instance / "signals/alpha_live_long_only_signals.parquet"),
        "ALPHA_FREQTRADE_LIVE_STATUS_PATH": str(instance / "signals/alpha_live_long_only_status.json"),
        "ALPHA_LIVE_AUDIT_PATH": str(instance / "signals/alpha_live_safety_audit.json"),
    }


def trade_command(instance: Path, config_path: Path | None = None) -> list[str]:
    return [sys.executable, "-m", "freqtrade", "trade", "--dry-run", "--userdir", str(instance_path(instance, "user_data")),
            "--config", str(config_path or instance / "config.json"),
            "--strategy-path", str(ROOT / "freqtrade/strategies"),
            "--strategy", "AlphaCrossSectionalLiveLongOnlyStrategy", "--no-color"]


def stop_worker(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def runtime_status(instance: Path, state: str, **fields) -> dict:
    report = {"strategy_id": instance_info(instance)["strategy_id"], "state": state, "dry_run": True,
              "supervisor_pid": os.getpid(), "checked_at": datetime.now(timezone.utc).isoformat(), **fields}
    path = instance_path(instance, "runtime") / "runtime_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)
    return report


def startup_check(instance: Path, seconds: int = 45) -> dict:
    config = json.loads((instance / "config.json").read_text())
    config.update({"initial_state": "stopped", "db_url": "sqlite:///" + str(instance_path(instance, "checks") / "startup-check.sqlite")})
    path = instance_path(instance, "checks") / "config.startup-check.json"
    path.write_text(json.dumps(config, indent=2))
    log_path = instance / "logs/startup-check.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(trade_command(instance, path), cwd=ROOT, env=environment(instance), stdout=log, stderr=log)
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            pass
        finally:
            stop_worker(process)
    text = log_path.read_text()
    report = {"strategy_id": instance_info(instance)["strategy_id"], "checked_at": datetime.now(timezone.utc).isoformat(),
              "worker_exit_code": process.returncode, "dry_run": True,
              "ready": "Changing state to: STOPPED" in text and "451" not in text,
              "region_restricted": True if "451" in text or "restricted location" in text else None,
              "market_load_timeout": "RequestTimeout" in text,
              "log": str(log_path), "check_did_not_enable_trading": True}
    (instance_path(instance, "checks") / "startup_report.json").write_text(json.dumps(report, indent=2))
    return report


def run(instance: Path) -> int:
    # Check a single public endpoint before starting a potentially long pair loop.
    settings = Settings(project_root=ROOT, binance_api_key=None, binance_secret=None, ccxt_trust_env=True)
    exchange = create_exchange(settings, sandbox=False)
    runtime_status(instance, "checking_network", worker_started=False)
    try:
        exchange.fetch_time()
    except Exception as error:
        report = runtime_status(instance, "blocked", worker_started=False, reason=str(error))
        print(json.dumps(report, indent=2))
        return 2

    def generate():
        subprocess.run([sys.executable, "-m", "src", "live", "generate-signals", "--mainnet", "--strategy", instance_info(instance)["strategy_id"],
            "--pairs-path", str(instance / "pairs.json"),
            "--signal-path", str(instance / "signals/alpha_live_long_only_signals.parquet"),
            "--status-path", str(instance / "signals/alpha_live_long_only_status.json")], cwd=ROOT,
            env=environment(instance), check=True, timeout=300)

    runtime_status(instance, "generating_signals", worker_started=False)
    try:
        generate()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        runtime_status(instance, "blocked", worker_started=False, reason=str(error))
        return 2
    # Reconcile before allowing this worker to reuse any existing audit.
    initial = audit_dry_run(instance / "signals/alpha_live_long_only_status.json",
                           instance / "signals/alpha_live_long_only_signals.parquet",
                           database_path(instance))
    publish_audit(initial, instance / "signals/alpha_live_safety_audit.json")
    log = (instance / "logs/freqtrade.log").open("a")
    log_offset = log.tell()
    process = subprocess.Popen(trade_command(instance), cwd=ROOT, env=environment(instance), stdout=log, stderr=log)
    runtime_status(instance, "starting", worker_started=True, worker_pid=process.pid)
    next_signal = time.monotonic() + 900
    try:
        while process.poll() is None:
            if time.monotonic() >= next_signal:
                try:
                    generate()
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    print(f"Signal refresh failed; audit remains authoritative: {error}", flush=True)
                next_signal = time.monotonic() + 900
            report = audit_dry_run(instance / "signals/alpha_live_long_only_status.json",
                                   instance / "signals/alpha_live_long_only_signals.parquet",
                                   database_path(instance))
            publish_audit(report, instance / "signals/alpha_live_safety_audit.json")
            with (instance / "logs/freqtrade.log").open("rb") as reader:
                reader.seek(log_offset)
                worker_log = reader.read().decode(errors="replace")
            ready = "Changing state to: RUNNING" in worker_log
            runtime_status(instance, "running" if ready else "starting", worker_started=True,
                           worker_pid=process.pid, audit_state=report["state"], entry_allowed=report["entry_allowed"])
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        stop_worker(process)
        log.close()
        runtime_status(instance, "stopped", worker_started=False, worker_exit_code=process.returncode)
    return process.returncode or 0


