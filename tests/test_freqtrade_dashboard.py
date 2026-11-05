from __future__ import annotations

import json
import zipfile

import pytest

from alpha_web3_research.backtest.dashboard import FreqtradeDashboardExporter
from alpha_web3_research.config import Settings


def test_dashboard_exporter_normalizes_all_strategy_runs(tmp_path) -> None:
    results_dir = tmp_path / "freqtrade" / "user_data" / "backtest_results"
    results_dir.mkdir(parents=True)
    archive_path = results_dir / "backtest-result-example.zip"
    payload = {
        "strategy": {
            "AlphaExample": {
                "backtest_start": "2026-01-01 00:00:00",
                "backtest_end": "2026-01-03 00:00:00",
                "backtest_days": 2,
                "timeframe": "4h",
                "trading_mode": "futures",
                "margin_mode": "isolated",
                "stake_currency": "USDT",
                "stake_amount": "unlimited",
                "max_open_trades": 2,
                "stoploss": -0.1,
                "starting_balance": 1_000,
                "final_balance": 1_030,
                "profit_total_abs": 30,
                "profit_total": 0.03,
                "max_drawdown_account": 0.01,
                "max_drawdown_abs": 10,
                "profit_factor": 1.4,
                "wins": 1,
                "draws": 0,
                "losses": 1,
                "total_trades": 2,
                "trade_count_long": 1,
                "trade_count_short": 1,
                "profit_total_long_abs": 40,
                "profit_total_long": 0.04,
                "profit_total_short_abs": -10,
                "profit_total_short": -0.01,
                "wallet_stats": {
                    "sharpe": 1.5,
                    "sortino": 2.0,
                    "calmar": 3.0,
                    "max_drawdown_account": 0.01,
                },
                "daily_profit": [["2026-01-01", -10], ["2026-01-02", 40]],
                "results_per_pair": [
                    {
                        "key": "BTC/USDT:USDT",
                        "trades": 2,
                        "profit_total_abs": 30,
                        "profit_total": 0.03,
                        "wins": 1,
                        "draws": 0,
                        "losses": 1,
                    },
                    {"key": "TOTAL", "trades": 2},
                ],
                "exit_reason_summary": [],
                "results_per_enter_tag": [],
                "periodic_breakdown": {"month": [], "weekday": []},
                "trades": [
                    {
                        "pair": "BTC/USDT:USDT",
                        "is_short": False,
                        "open_date": "2026-01-01 00:00:00+00:00",
                        "close_date": "2026-01-02 00:00:00+00:00",
                        "profit_abs": 40,
                        "profit_ratio": 0.04,
                        "fee_open": 0.0005,
                        "fee_close": 0.0005,
                        "leverage": 1,
                    }
                ],
            }
        },
        "strategy_comparison": [],
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("backtest-result-example.json", json.dumps(payload))
        archive.writestr("backtest-result-example_config.json", "{}")
    archive_path.with_suffix(".meta.json").write_text(
        json.dumps({"AlphaExample": {"run_id": "run-1"}}),
        encoding="utf-8",
    )
    (results_dir / ".last_result.json").write_text(
        json.dumps({"latest_backtest": archive_path.name}),
        encoding="utf-8",
    )

    output_path = tmp_path / "dashboard" / "public" / "data" / "backtests.json"
    result = FreqtradeDashboardExporter(Settings(project_root=tmp_path)).export(
        output_path=output_path
    )
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    run = exported["runs"][0]

    assert result.archive_count == 1
    assert result.strategy_count == 1
    assert result.run_count == 1
    assert result.trade_count == 2
    assert run["id"] == "AlphaExample:run-1"
    assert run["isLatest"] is True
    assert run["summary"]["returnPct"] == pytest.approx(3.0)
    assert run["summary"]["winRatePct"] == pytest.approx(50.0)
    assert run["execution"]["feePct"] == pytest.approx(0.05)
    assert [point["balance"] for point in run["equity"]] == [990.0, 1030.0]
    assert run["equity"][0]["drawdownPct"] == pytest.approx(-1.0)
    assert run["pairPerformance"][0]["pair"] == "BTC/USDT:USDT"
