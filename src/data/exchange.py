from __future__ import annotations

from typing import Any

import ccxt

from src.config import Settings


def create_exchange(settings: Settings, sandbox: bool | None = None) -> ccxt.Exchange:
    exchange_class = getattr(ccxt, settings.exchange_id)
    config: dict[str, Any] = {
        "enableRateLimit": settings.enable_rate_limit,
        "timeout": settings.ccxt_timeout_ms,
        "trust_env": settings.ccxt_trust_env,
        "requests_trust_env": settings.ccxt_trust_env,
        "aiohttp_trust_env": settings.ccxt_trust_env,
        "options": {"defaultType": settings.market_type},
    }
    if settings.binance_api_key and settings.binance_secret:
        config["apiKey"] = settings.binance_api_key
        config["secret"] = settings.binance_secret

    exchange = exchange_class(config)
    if settings.sandbox if sandbox is None else sandbox:
        exchange.set_sandbox_mode(True)
    return exchange

