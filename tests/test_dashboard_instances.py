import json
import os
import pytest
import sqlite3
from contextlib import closing
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
    with closing(sqlite3.connect(database)) as connection, connection:
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


@pytest.mark.parametrize("funding", [-12., 0., 17.])
def test_closed_funding_is_reported_without_adding_it_to_profit_twice(tmp_path, funding):
    directory = make_instance(tmp_path, "one", profit=100)
    with closing(sqlite3.connect(directory / "database/trades.dryrun.sqlite")) as c, c:
        c.execute("UPDATE trades SET funding_fees = ?", (funding,))
    with TestClient(create_app(tmp_path, price_loader=lambda _: {}, start_sampler=False)) as client:
        result = client.get("/api/instances/one/live").json()
        assert result["summary"]["funding_net"] == funding
        assert result["summary"]["equity"] == 10100
        assert result["summary"]["trading_fees"] == result["summary"]["fees"] == 0


@pytest.mark.parametrize("short,price,entry,exit", [(0, 110., "buy", "sell"), (1, 90., "sell", "buy")])
def test_partial_exits_do_not_double_count_realized_funding_or_running_funding(tmp_path, short, price, entry, exit):
    directory = make_instance(tmp_path, "one", profit=100)
    database = directory / "database/trades.dryrun.sqlite"
    with closing(sqlite3.connect(database)) as c, c:
        c.execute("ALTER TABLE trades ADD COLUMN realized_profit REAL DEFAULT 0")
        c.execute("ALTER TABLE trades ADD COLUMN funding_fee_running REAL DEFAULT 0")
        c.execute("ALTER TABLE orders ADD COLUMN funding_fee REAL")
        c.execute("ALTER TABLE orders ADD COLUMN ft_is_open INTEGER DEFAULT 0")
        c.execute("UPDATE trades SET funding_fees = -3")
        c.execute("""INSERT INTO trades (
            id, pair, is_open, fee_open, fee_close, open_rate, stake_amount, amount,
            open_date, leverage, is_short, funding_fees, realized_profit, funding_fee_running
        ) VALUES (2, 'ETH/USDT:USDT', 1, 0, 0, 100, 500, 5,
                  '2026-11-11 00:00:00', 1, ?, -7, 40, -1)""", (short,))
        for order_id, side, funding in [(1, entry, -1), (2, exit, -2), (3, entry, -1), (4, exit, -1), (5, entry, -1)]:
            c.execute("INSERT INTO orders (id, ft_trade_id, ft_order_side, filled, funding_fee) VALUES (?, 2, ?, 1, ?)",
                      (order_id, side, funding))
        c.execute("INSERT INTO orders (id, ft_trade_id, ft_order_side, filled, funding_fee) VALUES (6, 2, ?, 0, 999)", (exit,))
        c.execute("INSERT INTO orders (id, ft_trade_id, ft_order_side, filled, funding_fee, ft_is_open) VALUES (7, 2, ?, 1, 999, 1)", (exit,))
    original = database.read_bytes()
    app = create_app(tmp_path, price_loader=lambda _: {"ETH/USDT:USDT": price}, start_sampler=False)
    with TestClient(app) as client:
        result = client.get("/api/instances/one/live").json()
        assert result["positions"][0]["pnl_abs"] == 48  # 50 price PnL - 2 remaining funding
        assert result["summary"]["unrealized_pnl"] == 48
        assert result["summary"]["realized_pnl"] == 140
        assert result["summary"]["funding_net"] == -10
        assert result["summary"]["equity"] == 10188
        assert result["summary"]["wallet_balance"] == 10140
        assert client.get("/api/instances/one/equity").json()["points"][-1]["equity"] == 10188
    assert original == database.read_bytes()


def test_missing_database_is_not_an_empty_healthy_wallet(tmp_path):
    directory = make_instance(tmp_path, "one")
    (directory / "database/trades.dryrun.sqlite").unlink()
    with TestClient(create_app(tmp_path, start_sampler=False)) as client:
        assert client.get("/api/instances").json()["instances"][0]["database_ready"] is False
        assert client.get("/api/instances/one/live").status_code == 409


def test_historical_curve_is_separate_from_live_wallet(tmp_path):
    directory = make_instance(tmp_path, "xgb15-paper")
    (directory / "instance.json").write_text(json.dumps({"layout": "structured", "strategy_id": "xgb15-120-long-short-v1"}))
    (directory / "database/trades.dryrun.sqlite").unlink()
    make_instance(tmp_path, "another-strategy")
    report = tmp_path / "data/imports/xgb15/history_2026.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"kind": "historical_backtest", "strategy_id": "xgb15-120-long-short-v1", "points": [{"date": "2026-01-01", "equity": 1.}]}))
    with TestClient(create_app(tmp_path, start_sampler=False)) as client:
        assert client.get("/api/instances/xgb15-paper/live").status_code == 409
        history = client.get("/api/instances/xgb15-paper/backtest")
        assert history.status_code == 200
        assert history.json()["kind"] == "historical_backtest"
        assert client.get("/api/instances/another-strategy/backtest").status_code == 404
        assert client.get("/api/instances/missing/backtest").status_code == 404
        report.write_text(json.dumps({"kind": "live", "strategy_id": "xgb15-120-long-short-v1"}))
        assert client.get("/api/instances/xgb15-paper/backtest").status_code == 503


def test_stale_heartbeat_and_path_escape(tmp_path):
    directory = make_instance(tmp_path, "one")
    runtime_path = directory / "runtime/runtime_status.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["checked_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    runtime_path.write_text(json.dumps(runtime))
    instances, _ = discover_instances(tmp_path)
    assert instances["one"].metadata()["runtime_state"] == "unknown"
    assert instances["one"].metadata()["entry_allowed"] is False
    try:
        (directory.parent / "linked").symlink_to(directory, target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege unavailable")
        raise
    instances, errors = discover_instances(tmp_path)
    assert "linked" not in instances
    assert errors[0]["id"] == "linked"


def test_sampling_collects_unselected_instances_and_survives_one_failure(tmp_path):
    make_instance(tmp_path, "one")
    second = make_instance(tmp_path, "two")
    broken = make_instance(tmp_path, "broken")
    with closing(sqlite3.connect(broken / "database/trades.dryrun.sqlite")) as connection, connection:
        connection.execute("DROP TABLE trades")
    monitor = InstanceMonitor(tmp_path, lambda _: {})
    monitor.sample_all()
    with closing(sqlite3.connect(second / "database/equity_history.sqlite")) as connection, connection:
        assert connection.execute("SELECT COUNT(*) FROM equity_history").fetchone()[0] == 1
    with TestClient(create_app(tmp_path, start_sampler=False)) as client:
        assert client.get("/api/instances/broken/live").status_code == 503
        assert client.get("/api/instances/one/live").status_code == 200


def test_incomplete_prices_do_not_write_false_equity_history(tmp_path):
    directory = make_instance(tmp_path, "one")
    with closing(sqlite3.connect(directory / "database/trades.dryrun.sqlite")) as connection, connection:
        connection.execute("UPDATE trades SET is_open = 1")
    with TestClient(create_app(tmp_path, price_loader=lambda _: {}, start_sampler=False)) as client:
        result = client.get("/api/instances/one/live").json()
        assert result["price_state"] == "degraded"
        assert result["equity_history"] == []


def test_dead_process_cannot_be_reported_as_running(tmp_path, monkeypatch):
    make_instance(tmp_path, "one")

    monkeypatch.setattr("src.live.monitoring.process_alive", lambda pid: False)
    instances, _ = discover_instances(tmp_path)
    assert instances["one"].metadata()["runtime_state"] == "unknown"
    assert instances["one"].metadata()["audit_state"] == "unknown"
