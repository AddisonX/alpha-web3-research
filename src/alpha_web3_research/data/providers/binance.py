"""Binance USD-M historical adapter, reusing the existing CCXT collector."""
from __future__ import annotations

import polars as pl

from alpha_web3_research.data.contracts import (
    BarBatch, BarRequest, Instrument, InstrumentKey, ProviderCapabilities,
)
from alpha_web3_research.data.ohlcv import OHLCVFetcher
from alpha_web3_research.data.time import format_timestamp_ms, parse_timestamp_ms


class BinanceProvider(OHLCVFetcher):
    provider_id = "binance_official_ccxt"

    def capabilities(self) -> ProviderCapabilities:
        self._validate_market()
        return ProviderCapabilities(
            provider_id=self.provider_id,
            venue="binanceusdm",
            datasets=("bars",),
            timeframes=tuple(sorted(self.exchange.timeframes or {})),
        )

    def _validate_market(self) -> None:
        if self.exchange.id != "binanceusdm" or self.settings.market_type != "swap":
            raise ValueError("BinanceProvider supports binanceusdm swap only")

    def list_instruments(self) -> list[Instrument]:
        self._validate_market()
        return [
            Instrument(
                key=InstrumentKey("binanceusdm", "swap", symbol),
                native_symbol=market["id"],
                base=market.get("base"), quote=market.get("quote"),
                settle=market.get("settle"), contract_size=market.get("contractSize"),
                active=market.get("active"),
            )
            for symbol, market in sorted(self.exchange.load_markets().items())
            if market.get("swap") and market.get("linear")
        ]

    def fetch_bars(self, request: BarRequest, *, limit_per_request: int = 1000) -> BarBatch:
        capabilities = self.capabilities()
        if request.instrument.venue != capabilities.venue or request.instrument.market_type != "swap":
            raise ValueError("request instrument belongs to a different market")
        if request.timeframe not in capabilities.timeframes:
            raise ValueError(f"unsupported timeframe: {request.timeframe}")
        if not 1 <= limit_per_request <= 1500:
            raise ValueError("limit_per_request must be between 1 and 1500")
        markets = self.exchange.load_markets()
        market = markets.get(request.instrument.symbol)
        if not market or not market.get("swap") or not market.get("linear"):
            raise ValueError(f"unknown USD-M perpetual: {request.instrument.symbol}")
        frame = super().fetch_ohlcv_frame(
            symbol=request.instrument.symbol, timeframe=request.timeframe,
            since=format_timestamp_ms(request.start_ms), until=format_timestamp_ms(request.end_ms),
            limit_per_request=limit_per_request, include_unclosed=request.include_unclosed,
        )
        return BarBatch(request=request, frame=frame, provider_id=self.provider_id)

    def fetch_ohlcv_frame(
        self, symbol: str, timeframe: str, since: str, until: str | None = None,
        limit_per_request: int = 1000, include_unclosed: bool = False,
    ) -> pl.DataFrame:
        """Legacy CLI compatibility: retain column names, source and numeric values."""
        start_ms = parse_timestamp_ms(since)
        if start_ms is None:
            raise ValueError("since is required")
        return self.fetch_bars(
            BarRequest(InstrumentKey(self.exchange.id, self.settings.market_type, symbol),
                       timeframe, start_ms, parse_timestamp_ms(until), include_unclosed),
            limit_per_request=limit_per_request,
        ).frame
