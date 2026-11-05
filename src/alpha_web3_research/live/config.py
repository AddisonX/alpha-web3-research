from __future__ import annotations

import json
from pathlib import Path

from alpha_web3_research.backtest.freqtrade import _freqtrade_config


def write_live_long_only_config(
    pairs_path: Path,
    output_path: Path,
    max_open_trades: int = 9,
    use_testnet_market_data: bool = True,
) -> Path:
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    config = _freqtrade_config(
        pairs=[str(pair) for pair in pairs],
        timeframe="4h",
        use_testnet_market_metadata=use_testnet_market_data,
        max_open_trades=max_open_trades,
        bot_name="alpha-cross-sectional-live-long-only-dry-run",
    )
    config.update(
        {
            "dry_run": True,
            "initial_state": "running",
            "db_url": "sqlite:///freqtrade/user_data/tradesv3.live-long-only.dryrun.sqlite",
            "internals": {"process_throttle_secs": 15},
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path
