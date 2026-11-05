"""Provider selection for existing CLI commands."""
import ccxt

from alpha_web3_research.config import Settings
from alpha_web3_research.data.ohlcv import OHLCVFetcher
from alpha_web3_research.data.providers.binance import BinanceProvider


def create_ohlcv_fetcher(exchange: ccxt.Exchange, settings: Settings) -> OHLCVFetcher:
    if settings.data_provider == "binance_official_ccxt":
        if exchange.id != "binanceusdm" or settings.market_type != "swap":
            raise ValueError("binance_official_ccxt requires binanceusdm swap")
        return BinanceProvider(exchange, settings)
    if settings.data_provider == "auto":
        if exchange.id == "binanceusdm" and settings.market_type == "swap":
            return BinanceProvider(exchange, settings)
        # Preserve previously supported CCXT commands without claiming a validated adapter.
        return OHLCVFetcher(exchange, settings)
    raise ValueError(f"unsupported data_provider: {settings.data_provider}")
