from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import ccxt

from alpha_web3_research.backtest.freqtrade import _freqtrade_config


TESTNET_API_URLS = {
    "sapi": "http://127.0.0.1:9/disabled",
    "sapiV2": "http://127.0.0.1:9/disabled",
    "sapiV3": "http://127.0.0.1:9/disabled",
    "sapiV4": "http://127.0.0.1:9/disabled",
    "dapiPublic": "https://testnet.binancefuture.com/dapi/v1",
    "dapiPrivate": "https://testnet.binancefuture.com/dapi/v1",
    "dapiPrivateV2": "https://testnet.binancefuture.com/dapi/v2",
    "dapiData": "https://testnet.binancefuture.com/futures/data",
    "eapiPublic": "http://127.0.0.1:9/disabled",
    "eapiPrivate": "http://127.0.0.1:9/disabled",
    "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
    "fapiPublicV2": "https://testnet.binancefuture.com/fapi/v2",
    "fapiPublicV3": "https://testnet.binancefuture.com/fapi/v3",
    "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
    "fapiPrivateV2": "https://testnet.binancefuture.com/fapi/v2",
    "fapiPrivateV3": "https://testnet.binancefuture.com/fapi/v3",
    "fapiData": "https://testnet.binancefuture.com/futures/data",
    "public": "https://testnet.binance.vision/api/v3",
    "private": "https://testnet.binance.vision/api/v3",
    "v1": "https://testnet.binance.vision/api/v1",
    "papi": "http://127.0.0.1:9/disabled",
    "papiV2": "http://127.0.0.1:9/disabled",
}


def write_testnet_config(pairs_path: Path, output_path: Path, database_path: Path) -> Path:
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or not pairs or any(not isinstance(pair, str) for pair in pairs):
        raise ValueError("pairs file must contain a nonempty list of pair strings")
    config = _freqtrade_config(
        pairs=sorted(set(pairs)),
        timeframe="4h",
        use_testnet_market_metadata=False,
        max_open_trades=1,
        bot_name="alpha-testnet-live-smoke",
    )
    for key in ("ccxt_config", "ccxt_async_config"):
        config["exchange"][key] = {
            "options": {"defaultType": "swap", "fetchMarkets": {"types": ["linear"]}, "fetchCurrencies": False},
            "urls": {"api": TESTNET_API_URLS},
        }
    config.update(
        {
            "dry_run": False,
            "initial_state": "running",
            "stake_amount": 20,
            "tradable_balance_ratio": 0.99,
            "db_url": f"sqlite:///{database_path.resolve()}",
            "stoploss": -0.10,
            "stoploss_price_type": "mark",
            "order_types": {
                "entry": "market",
                "exit": "market",
                "emergency_exit": "market",
                "force_entry": "market",
                "force_exit": "market",
                "stoploss": "market",
                "stoploss_on_exchange": True,
                "stoploss_on_exchange_interval": 60,
            },
        }
    )
    validate_testnet_config(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return output_path


def validate_testnet_config(config: dict[str, Any]) -> None:
    exchange = config.get("exchange")
    if not isinstance(exchange, dict):
        raise ValueError("exchange configuration is missing")
    if config.get("dry_run") is not False or config.get("trading_mode") != "futures":
        raise ValueError("testnet execution requires futures with dry_run=false")
    if config.get("margin_mode") != "isolated" or exchange.get("name") != "binance":
        raise ValueError("testnet execution requires isolated Binance futures")
    if config.get("stake_currency") != "USDT" or config.get("max_open_trades") != 1:
        raise ValueError("testnet execution requires USDT and one open trade maximum")
    if isinstance(config.get("stake_amount"), bool) or config.get("stake_amount") != 20:
        raise ValueError("testnet stake must be exactly 20 USDT")
    if config.get("stoploss") != -0.10 or config.get("stoploss_price_type") != "mark":
        raise ValueError("testnet stoploss configuration changed")
    types = config.get("order_types", {})
    if types.get("stoploss_on_exchange") is not True or types.get("stoploss") != "market":
        raise ValueError("testnet requires an exchange-side market stoploss")
    if not str(config.get("db_url", "")).startswith("sqlite:////") or "testnet" not in str(config["db_url"]):
        raise ValueError("testnet requires a separate absolute SQLite database")
    if exchange.get("enable_ws") is not False:
        raise ValueError("testnet execution requires REST-only exchange access")
    for key in ("ccxt_config", "ccxt_async_config"):
        urls = exchange.get(key, {}).get("urls", {}).get("api")
        if urls != TESTNET_API_URLS:
            raise ValueError(f"{key} is not pinned to the complete Testnet API map")
        effective = ccxt.binance(exchange[key]).urls["api"]
        if effective != TESTNET_API_URLS:
            raise ValueError(f"{key} leaves unpinned CCXT API endpoints")


def load_testnet_credentials(path: Path) -> tuple[str, str]:
    if path.stat().st_mode & 0o077:
        raise ValueError("testnet credential file must be readable only by its owner (mode 0600)")
    document = json.loads(path.read_text(encoding="utf-8"))
    if set(document) != {"exchange"} or set(document["exchange"]) != {"key", "secret"}:
        raise ValueError("credential file may contain only exchange.key and exchange.secret")
    key = document["exchange"]["key"]
    secret = document["exchange"]["secret"]
    if not isinstance(key, str) or not key or not isinstance(secret, str) or not secret:
        raise ValueError("testnet API key and secret are required")
    return key, secret


def preflight_testnet_account(config_path: Path, credentials_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_testnet_config(config)
    key, secret = load_testnet_credentials(credentials_path)
    exchange = ccxt.binance({
        "apiKey": key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {"defaultType": "future", "fetchMarkets": {"types": ["linear"]}, "fetchCurrencies": False},
    })
    exchange.set_sandbox_mode(True)
    sandbox_urls = exchange.urls["api"]
    if not sandbox_urls or set(sandbox_urls) - set(TESTNET_API_URLS) or any(
        sandbox_urls[key] != TESTNET_API_URLS[key] for key in sandbox_urls
    ):
        raise RuntimeError("CCXT Testnet API map does not match the pinned map")
    try:
        exchange.load_markets()
        if not exchange.fapiPrivateGetPositionSideDual().get("dualSidePosition") is False:
            raise RuntimeError("Testnet account must use one-way position mode")
        if not exchange.fapiPrivateGetMultiAssetsMargin().get("multiAssetsMargin") is False:
            raise RuntimeError("Testnet account must use single-asset margin mode")
        balance = exchange.fetch_balance()
        available = float(balance.get("USDT", {}).get("free") or 0)
        if available < 25:
            raise RuntimeError("Testnet account needs at least 25 available USDT")
        positions = exchange.fetch_positions()
        nonzero = [position["symbol"] for position in positions if abs(float(position.get("contracts") or 0)) > 0]
        open_orders = exchange.fapiPrivateGetOpenOrders()
        open_algo_orders = exchange.fapiPrivateGetOpenAlgoOrders()
        if nonzero or open_orders or open_algo_orders:
            raise RuntimeError("Testnet account must be dedicated and start with no positions or open orders")
        return {"available_usdt": available, "testnet_markets": len(exchange.markets), "flat": True}
    finally:
        exchange.close()
