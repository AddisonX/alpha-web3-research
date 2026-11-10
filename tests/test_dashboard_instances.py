import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from src.live.dashboard import InstanceMonitor, create_app
from src.live.monitoring import discover_instances


def make_instance(root, name, wallet=10000, profit=100):
    directory = root / "freqtrade/instances" / name
    (directory / "database").mkdir(parents=True)
    (directory / "runtime").mkdir()
    (directory / "signals").mkdir()
    (directory / "instance.json").write_text(json.dumps({"layout": "structured", "strategy_id": name}))
    database = directory / "database/trades.dryrun.sqlite"
    (directory / "config.json").write_text(json.dumps({
        "dry_run": True, "dry_run_wallet": wallet, "trading_mode": "futures",
        "exchange": {"name": "binance"}, "stake_currency": "USDT",
        "db_url": f"sqlite:///{database}", "strategy": "SharedExecutionAdapter",
    }))
    (directory / "runtime/runtime_status.json").write_text(json.dumps({
        "state": "running", "audit_state": "healthy", "entry_allowed": True,
        "supervisor_pid": os.getpid(), "worker_pid": os.getpid(),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }))
    with sqlite3.connect(database) as connection:
        connection.execute("""CREATE TABLE trades (
            id INTEGER, pair TEXT, is_open INTEGER, fee_open REAL, fee_open_cost REAL,
            fee_close REAL, fee_close_cost REAL, open_rate REAL, close_rate REAL,
            close_profit REAL, close_profit_abs REAL, stake_amount REAL, amount REAL,
            open_date TEXT, close_date TEXT, stop_loss REAL, leverage REAL, is_short INTEGER,
            funding_fees REAL, enter_tag TEXT, exit_reason TEXT
        )""")
        connection.execute("""CREATE TABLE orders (
            id INTEGER, ft_trade_id INTEGER, ft_pair TEXT, ft_order_side TEXT,
            status TEXT, order_type TEXT, average REAL, filled REAL, cost REAL,
            order_date TEXT, order_filled_date TEXT
        )""")
        connection.execute("""INSERT INTO trades VALUES (
            1, 'BTC/USDT:USDT', 0, 0, 0, 0, 0, 100, 101, ?, ?, 1000, 10,
            '2026-11-01 00:00:00', '2026-11-02 00:00:00', 90, 1, 0, 0, 'test', 'signal'
        )""", (profit / 1000, profit))
    return directory


def test_discovery_is_dynamic_and_isolates_invalid_instances(tmp_path):
    make_instance(tmp_path, "one")
    app = create_app(tmp_path, price_loader=lambda _: {}, start_sampler=False)
    with TestClient(app) as client:
        assert [row["id"] for row in client.get("/api/instances").json()["instances"]] == ["one"]
        make_instance(tmp_path, "two")
        invalid = make_instance(tmp_path, "invalid")
        config = json.loads((invalid / "config.json").read_text())
        config["dry_run"] = False
        (invalid / "config.json").write_text(json.dumps(config))
        result = client.get("/api/instances").json()
        assert [row["id"] for row in result["instances"]] == ["one", "two"]
        assert result["errors"][0]["id"] == "invalid"
        assert client.get("/api/instances/missing/live").status_code == 404
        assert client.get("/api/instances/invalid/live").status_code == 404


def test_balances_and_histories_do_not_cross_instances(tmp_path):
    first = make_instance(tmp_path, "one", wallet=10000, profit=100)
    make_instance(tmp_path, "two", wallet=20000, profit=-400)
    original = (first / "database/trades.dryrun.sqlite").read_bytes()
    with TestClient(create_app(tmp_path, price_loader=lambda _: {}, start_sampler=False)) as client:
        one = client.get("/api/instances/one/live").json()
        two = client.get("/api/instances/two/live").json()
        assert one["instance"]["id"] == "one"
        assert one["summary"]["equity"] == 10100
        assert one["summary"]["return_pct"] == 1
        assert two["summary"]["equity"] == 19600
        assert two["summary"]["return_pct"] == -2
        assert two["summary"]["max_drawdown_pct"] == 2
        assert client.get("/api/live").json()["instance"]["id"] == "one"
        for name, equity in (("one", 10100), ("two", 19600)):
            response = client.get(f"/api/instances/{name}/equity")
            assert response.json()["points"][0]["equity"] == equity
        assert client.get("/api/instances/one/equity?interval=bad").status_code == 422
    assert original == (first / "database/trades.dryrun.sqlite").read_bytes()


def test_missing_database_is_not_an_empty_healthy_wallet(tmp_path):
    directory = make_instance(tmp_path, "one")
    (directory / "database/trades.dryrun.sqlite").unlink()
    with TestClient(create_app(tmp_path, start_sampler=False)) as client:
        assert client.get("/api/instances").json()["instances"][0]["database_ready"] is False
        assert client.get("/api/instances/one/live").status_code == 409


def test_stale_heartbeat_and_path_escape(tmp_path):
    directory = make_instance(tmp_path, "one")
    runtime_path = directory / "runtime/runtime_status.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["checked_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    runtime_path.write_text(json.dumps(runtime))
    instances, _ = discover_instances(tmp_path)
    assert instances["one"].metadata()["runtime_state"] == "unknown"
    assert instances["one"].metadata()["entry_allowed"] is False
    (directory.parent / "linked").symlink_to(directory, target_is_directory=True)
    instances, errors = discover_instances(tmp_path)
    assert "linked" not in instances
    assert errors[0]["id"] == "linked"


def test_sampling_collects_unselected_instances_and_survives_one_failure(tmp_path):
    make_instance(tmp_path, "one")
    second = make_instance(tmp_path, "two")
    broken = make_instance(tmp_path, "broken")
    with sqlite3.connect(broken / "database/trades.dryrun.sqlite") as connection:
        connection.execute("DROP TABLE trades")
    monitor = InstanceMonitor(tmp_path, lambda _: {})
    monitor.sample_all()
    with sqlite3.connect(second / "database/equity_history.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM equity_history").fetchone()[0] == 1
    with TestClient(create_app(tmp_path, start_sampler=False)) as client:
        assert client.get("/api/instances/broken/live").status_code == 503
        assert client.get("/api/instances/one/live").status_code == 200


def test_incomplete_prices_do_not_write_false_equity_history(tmp_path):
    directory = make_instance(tmp_path, "one")
    with sqlite3.connect(directory / "database/trades.dryrun.sqlite") as connection:
        connection.execute("UPDATE trades SET is_open = 1")
    with TestClient(create_app(tmp_path, price_loader=lambda _: {}, start_sampler=False)) as client:
        result = client.get("/api/instances/one/live").json()
        assert result["price_state"] == "degraded"
        assert result["equity_history"] == []


def test_dead_process_cannot_be_reported_as_running(tmp_path, monkeypatch):
    make_instance(tmp_path, "one")

    def missing_process(pid, signal):
        raise ProcessLookupError

    monkeypatch.setattr("src.live.monitoring.os.kill", missing_process)
    instances, _ = discover_instances(tmp_path)
    assert instances["one"].metadata()["runtime_state"] == "unknown"
    assert instances["one"].metadata()["audit_state"] == "unknown"
