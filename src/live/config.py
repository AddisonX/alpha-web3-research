from __future__ import annotations

import json
from pathlib import Path

from typing import Any


def write_live_long_only_config(
    pairs_path: Path,
    output_path: Path,
    max_open_trades: int = 9,
    use_testnet_market_data: bool = False,
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
            "db_url": "sqlite:///freqtrade/instances/tokyo-lightgbm/database/trades.dryrun.sqlite",
            "internals": {"process_throttle_secs": 15},
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _freqtrade_config(
    pairs: list[str],
    timeframe: str,
    use_testnet_market_metadata: bool,
    max_open_trades: int = 18,
    bot_name: str = "alpha-cross-sectional-backtest",
) -> dict[str, Any]:
    ccxt_config: dict[str, Any] = {}
    if use_testnet_market_metadata:
        ccxt_config = {
            "options": {
                "defaultType": "swap",
                "fetchMarkets": {"types": ["linear"]},
                "fetchCurrencies": False,
            },
            "urls": {
                "api": {
                    "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPublicV2": "https://testnet.binancefuture.com/fapi/v2",
                    "fapiPublicV3": "https://testnet.binancefuture.com/fapi/v3",
                    "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPrivateV2": "https://testnet.binancefuture.com/fapi/v2",
                    "fapiPrivateV3": "https://testnet.binancefuture.com/fapi/v3",
                    "fapiData": "https://testnet.binancefuture.com/futures/data",
                }
            }
        }
    return {
        "$schema": "https://schema.freqtrade.io/schema.json",
        "bot_name": bot_name,
        "dry_run": True,
        "dry_run_wallet": 10_000,
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "liquidation_buffer": 0.05,
        "futures_funding_rate": 0.0,
        "stake_currency": "USDT",
        "stake_amount": "unlimited",
        "tradable_balance_ratio": 0.99,
        "max_open_trades": max_open_trades,
        "timeframe": timeframe,
        "dataformat_ohlcv": "feather",
        "cancel_open_orders_on_exit": True,
        "unfilledtimeout": {
            "entry": 10,
            "exit": 10,
            "exit_timeout_count": 0,
            "unit": "minutes",
        },
        "entry_pricing": {
            "price_side": "other",
            "use_order_book": True,
            "order_book_top": 1,
            "price_last_balance": 0.0,
            "check_depth_of_market": {"enabled": False, "bids_to_ask_delta": 1},
        },
        "exit_pricing": {
            "price_side": "other",
            "use_order_book": True,
            "order_book_top": 1,
        },
        "exchange": {
            "name": "binance",
            "key": "",
            "secret": "",
            "ccxt_config": ccxt_config,
            "ccxt_async_config": ccxt_config,
            "enable_ws": False,
            "pair_whitelist": pairs,
            "pair_blacklist": [],
        },
        "pairlists": [{"method": "StaticPairList", "allow_inactive": use_testnet_market_metadata}],
        "internals": {"process_throttle_secs": 5},
    }

