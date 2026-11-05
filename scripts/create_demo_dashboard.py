"""Create a deterministic, synthetic dashboard fixture. No exchange access."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

from alpha_web3_research.backtest.dashboard import FreqtradeDashboardExporter
from alpha_web3_research.config import Settings


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = {
        "backtest_start": "2026-01-01 00:00:00", "backtest_end": "2026-01-03 00:00:00",
        "backtest_days": 2, "timeframe": "4h", "trading_mode": "futures", "margin_mode": "isolated",
        "stake_currency": "USDT", "stake_amount": "unlimited", "max_open_trades": 2, "stoploss": -.1,
        "starting_balance": 1000, "final_balance": 1030, "profit_total_abs": 30, "profit_total": .03,
        "max_drawdown_account": .01, "max_drawdown_abs": 10, "profit_factor": 4,
        "wins": 1, "draws": 0, "losses": 1, "total_trades": 2,
        "trade_count_long": 2, "trade_count_short": 0,
        "profit_total_long_abs": 30, "profit_total_long": .03,
        "profit_total_short_abs": 0, "profit_total_short": 0,
        "daily_profit": [["2026-01-01", -10], ["2026-01-02", 40]],
        "results_per_pair": [
            {"key": "BTC/USDT:USDT", "trades": 1, "profit_total_abs": 40, "profit_total": .04, "wins": 1, "losses": 0},
            {"key": "ETH/USDT:USDT", "trades": 1, "profit_total_abs": -10, "profit_total": -.01, "wins": 0, "losses": 1},
        ],
        "exit_reason_summary": [], "results_per_enter_tag": [],
        "periodic_breakdown": {"month": [], "weekday": []},
        "trades": [
            {"pair": "ETH/USDT:USDT", "is_short": False, "open_date": "2026-01-01 00:00:00+00:00",
             "close_date": "2026-01-01 04:00:00+00:00", "trade_duration": 240,
             "profit_abs": -10, "profit_ratio": -.02, "stake_amount": 500,
             "fee_open": .0005, "fee_close": .0005, "leverage": 1, "exit_reason": "synthetic_exit"},
            {"pair": "BTC/USDT:USDT", "is_short": False, "open_date": "2026-01-02 00:00:00+00:00",
             "close_date": "2026-01-02 04:00:00+00:00", "trade_duration": 240,
             "profit_abs": 40, "profit_ratio": .08, "stake_amount": 500,
             "fee_open": .0005, "fee_close": .0005, "leverage": 1, "exit_reason": "synthetic_exit"},
        ],
    }
    destination = root / "dashboard/public/data/backtests.example.json"
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        archive = folder / "synthetic-example.zip"
        with ZipFile(archive, "w") as stream:
            stream.writestr("result.json", json.dumps({"strategy": {"SyntheticExampleStrategy": result}}))
        (folder / ".last_result.json").write_text(json.dumps({"latest_backtest": archive.name}))
        FreqtradeDashboardExporter(Settings(project_root=root)).export(results_dir=folder, output_path=destination)
    data = json.loads(destination.read_text())
    data["sourceDirectory"] = "synthetic fixture; not real trades or historical performance"
    data["generatedAt"] = "2026-01-03T00:00:00+00:00"
    for run in data["runs"]:
        run["generatedAt"] = data["generatedAt"]
    destination.write_text(json.dumps(data, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
