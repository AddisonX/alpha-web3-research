from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import ccxt
import polars as pl

from src.config import Settings
from src.data.time import parse_timestamp_ms
from src.universe.rules import (
    is_leveraged_base,
    is_non_crypto_base,
    is_stable_base,
)

OHLCV_COLUMNS = [
    "exchange",
    "market_type",
    "symbol",
    "timeframe",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "fetched_at",
    "source",
]


class OHLCVFetcher:
    def __init__(self, exchange: ccxt.Exchange, settings: Settings) -> None:
        self.exchange = exchange
        self.settings = settings

    def list_usdt_perpetual_symbols(
        self,
        quote: str = "USDT",
        limit: int | None = None,
        exclude_non_crypto: bool = True,
        exclude_stable: bool = True,
        exclude_leveraged: bool = True,
        sort_by_quote_volume: bool = False,
    ) -> list[str]:
        markets = self.exchange.load_markets()
        symbols: list[str] = []
        for symbol, market in markets.items():
            if not self._is_perpetual_quote_market(
                market=market,
                quote=quote,
                exclude_non_crypto=exclude_non_crypto,
                exclude_stable=exclude_stable,
                exclude_leveraged=exclude_leveraged,
            ):
                continue
            symbols.append(symbol)
        if sort_by_quote_volume:
            symbols = self._sort_symbols_by_quote_volume(symbols)
        else:
            symbols = sorted(symbols)
        return symbols[:limit] if limit else symbols

    def ranked_usdt_perpetual_symbols(
        self,
        quote: str = "USDT",
        limit: int | None = None,
        exclude_non_crypto: bool = True,
        exclude_stable: bool = True,
        exclude_leveraged: bool = True,
    ) -> list[dict[str, Any]]:
        symbols = self.list_usdt_perpetual_symbols(
            quote=quote,
            limit=None,
            exclude_non_crypto=exclude_non_crypto,
            exclude_stable=exclude_stable,
            exclude_leveraged=exclude_leveraged,
            sort_by_quote_volume=True,
        )
        tickers = self._fetch_tickers(symbols)
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            ticker = tickers.get(symbol, {})
            rows.append(
                {
                    "symbol": symbol,
                    "quote_volume": ticker.get("quoteVolume"),
                    "base_volume": ticker.get("baseVolume"),
                    "last": ticker.get("last"),
                    "percentage": ticker.get("percentage"),
                }
            )
        return rows[:limit] if limit else rows

    def fetch_ohlcv_frame(
        self,
        symbol: str,
        timeframe: str,
        since: str,
        until: str | None = None,
        limit_per_request: int = 1000,
        include_unclosed: bool = False,
    ) -> pl.DataFrame:
        since_ms = parse_timestamp_ms(since)
        until_ms = parse_timestamp_ms(until)
        if since_ms is None:
            raise ValueError("since is required")

        timeframe_ms = self.exchange.parse_timeframe(timeframe) * 1000
        current_since = since_ms
        rows: list[list[Any]] = []

        while True:
            batch = self.exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit_per_request)
            if not batch:
                break

            for candle in batch:
                timestamp = int(candle[0])
                if timestamp < since_ms:
                    continue
                if until_ms is not None and timestamp >= until_ms:
                    continue
                rows.append(candle)

            last_timestamp = int(batch[-1][0])
            next_since = last_timestamp + timeframe_ms
            if next_since <= current_since:
                break
            current_since = next_since

            if until_ms is not None and current_since >= until_ms:
                break
            if len(batch) < limit_per_request:
                break

        fetched_at = datetime.now(timezone.utc).isoformat()
        frame = pl.DataFrame(
            rows,
            schema=["timestamp", "open", "high", "low", "close", "volume"],
            orient="row",
        ) if rows else pl.DataFrame(schema={column: pl.Null for column in ["timestamp", "open", "high", "low", "close", "volume"]})

        if frame.is_empty():
            return self._empty_frame()

        if not include_unclosed:
            last_closed_timestamp = self.exchange.milliseconds() - timeframe_ms
            frame = frame.filter(pl.col("timestamp") <= last_closed_timestamp)

        frame = (
            frame.with_columns(
                pl.lit(self.exchange.id).alias("exchange"),
                pl.lit(self.settings.market_type).alias("market_type"),
                pl.lit(symbol).alias("symbol"),
                pl.lit(timeframe).alias("timeframe"),
                pl.lit(fetched_at).alias("fetched_at"),
                pl.lit("ccxt.fetch_ohlcv").alias("source"),
            )
            .select(OHLCV_COLUMNS)
            .unique(subset=["exchange", "market_type", "symbol", "timeframe", "timestamp"], keep="last")
            .sort(["symbol", "timeframe", "timestamp"])
        )
        return frame

    def _empty_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "exchange": pl.String,
                "market_type": pl.String,
                "symbol": pl.String,
                "timeframe": pl.String,
                "timestamp": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "fetched_at": pl.String,
                "source": pl.String,
            }
        )

    def _sort_symbols_by_quote_volume(self, symbols: list[str]) -> list[str]:
        tickers = self._fetch_tickers(symbols)
        return sorted(symbols, key=lambda symbol: tickers.get(symbol, {}).get("quoteVolume") or 0.0, reverse=True)

    def _fetch_tickers(self, symbols: list[str]) -> dict[str, Any]:
        try:
            return self.exchange.fetch_tickers(symbols)
        except Exception:
            return {}

    @staticmethod
    def _is_perpetual_quote_market(
        market: dict[str, Any],
        quote: str,
        exclude_non_crypto: bool,
        exclude_stable: bool,
        exclude_leveraged: bool,
    ) -> bool:
        if not market.get("swap"):
            return False
        if market.get("quote") != quote:
            return False
        if market.get("active") is False:
            return False
        base = market.get("base") or ""
        info = market.get("info") or {}
        underlying_type = info.get("underlyingType")
        if exclude_non_crypto and (underlying_type not in {None, "COIN"} or is_non_crypto_base(base)):
            return False
        if exclude_stable and is_stable_base(base):
            return False
        if exclude_leveraged and is_leveraged_base(base):
            return False
        return True
