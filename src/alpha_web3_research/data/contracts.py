"""Version-one historical bar contracts; times are UTC epoch milliseconds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

import polars as pl


@dataclass(frozen=True)
class InstrumentKey:
    venue: str
    market_type: str
    symbol: str

    def __post_init__(self) -> None:
        if not all((self.venue, self.market_type, self.symbol)):
            raise ValueError("instrument identity fields must be non-empty")

    @property
    def instrument_id(self) -> str:
        return "v1:" + "|".join(quote(part, safe="") for part in (self.venue, self.market_type, self.symbol))


@dataclass(frozen=True)
class Instrument:
    key: InstrumentKey
    native_symbol: str
    base: str | None
    quote: str | None
    settle: str | None
    contract_size: float | None
    active: bool | None


@dataclass(frozen=True)
class ProviderCapabilities:
    provider_id: str
    venue: str
    datasets: tuple[str, ...]
    timeframes: tuple[str, ...]
    supports_live: bool = False
    # Unknown coverage must not be represented as unlimited coverage.
    history_start_ms: int | None = None


@dataclass(frozen=True)
class BarRequest:
    instrument: InstrumentKey
    timeframe: str
    start_ms: int
    end_ms: int | None = None
    include_unclosed: bool = False

    def __post_init__(self) -> None:
        if not self.timeframe:
            raise ValueError("timeframe is required")
        if self.start_ms < 0 or (self.end_ms is not None and self.end_ms <= self.start_ms):
            raise ValueError("bar range must be nonnegative and nonempty: [start, end)")


@dataclass(frozen=True)
class BarBatch:
    """Envelope provenance is explicit; frame retains the legacy OHLCV schema.

    A nonempty response is not a completeness guarantee. Coverage checks and
    persistent multi-provider snapshots belong to the next implementation stage.
    """
    request: BarRequest
    frame: pl.DataFrame
    provider_id: str
    schema_version: int = 1


class HistoricalBarProvider(Protocol):
    def capabilities(self) -> ProviderCapabilities: ...
    def list_instruments(self) -> list[Instrument]: ...
    def fetch_bars(self, request: BarRequest, *, limit_per_request: int = 1000) -> BarBatch: ...


@dataclass(frozen=True)
class BarQuery:
    venue: str
    market_type: str
    timeframe: str
    # None means all instruments in the selected market; () explicitly means none.
    symbols: tuple[str, ...] | None = None
    start_ms: int | None = None
    end_ms: int | None = None

    def __post_init__(self) -> None:
        if not all((self.venue, self.market_type, self.timeframe)):
            raise ValueError("venue, market_type and timeframe are required")
        if any(value is not None and value < 0 for value in (self.start_ms, self.end_ms)):
            raise ValueError("query times must be nonnegative")
        if self.start_ms is not None and self.end_ms is not None and self.end_ms <= self.start_ms:
            raise ValueError("query requires start < end")
