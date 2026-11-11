from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
import ccxt
import polars as pl
from src.config import Settings
from src.data.ohlcv import OHLCVFetcher
from src.factors.builder import FactorBuilder
from src.strategies.base import DataRequirements

@dataclass(frozen=True)
class PreparedMarket:
    features: pl.DataFrame
    requested_pair_count: int
    errors: dict[str, str]


class MarketDataError(RuntimeError):
    def __init__(self, message: str, errors: dict[str, str]) -> None:
        super().__init__(message)
        self.errors = errors


class MarketDataSource(Protocol):
    def prepare(self, pairs: list[str], decision_timestamp: int,
                requirements: DataRequirements) -> PreparedMarket: ...


class MarketDataProvider:
    def __init__(self, settings: Settings, exchange: ccxt.Exchange) -> None:
        self.settings = settings
        self.exchange = exchange

    def prepare(
        self, pairs: list[str], decision_timestamp: int, requirements: DataRequirements,
    ) -> PreparedMarket:
        timeframe_ms = int(self.exchange.parse_timeframe(requirements.timeframe) * 1000)
        from src.strategies.xgb15 import FEATURES
        caps = None
        if requirements.features == FEATURES:
            import pandas as pd
            cap_path = self.settings.resolved_data_path() / "imports/xgb15/caps.parquet"
            caps = pd.read_parquet(cap_path)
            from src.data.xgb15_caps import required_cap_date
            required_date = required_cap_date(decision_timestamp)
            if not pd.to_datetime(caps.date).eq(required_date).any():
                raise MarketDataError(f"market cap source date {required_date.date()} is missing; OHLCV fetch not started", {})
        since_ms = decision_timestamp - (requirements.lookback_bars - 1) * timeframe_ms
        until_ms = decision_timestamp + timeframe_ms
        fetcher = OHLCVFetcher(exchange=self.exchange, settings=self.settings)
        frames, errors = [], {}
        for pair in pairs:
            try:
                frame = fetcher.fetch_ohlcv_frame(
                    symbol=pair, timeframe=requirements.timeframe,
                    since=datetime.fromtimestamp(since_ms / 1000, timezone.utc).isoformat(),
                    until=datetime.fromtimestamp(until_ms / 1000, timezone.utc).isoformat(),
                    limit_per_request=1500, include_unclosed=False,
                )
                if frame.filter(pl.col("timestamp") == decision_timestamp).is_empty():
                    raise ValueError("decision candle is missing")
                if frame.height < requirements.min_closed_bars:
                    raise ValueError(f"only {frame.height} closed candles available")
                frames.append(frame)
            except Exception as exc:
                errors[pair] = f"{type(exc).__name__}: {exc}"
        if not frames:
            raise MarketDataError("no OHLCV frames were fetched", errors)
        if requirements.features == FEATURES:
            from src.factors.xgb15 import build_xgb15_features
            raw = pl.concat(frames, how="diagonal_relaxed").to_pandas()
            matrix = build_xgb15_features(raw, caps, decision_timestamp, pairs)
            matrix = matrix.dropna(subset=list(FEATURES))
            if matrix.empty:
                raise MarketDataError("no complete XGB15 features; check point-in-time cap coverage", errors)
            return PreparedMarket(pl.from_pandas(matrix), len(pairs), errors)
        factors = FactorBuilder(self.settings)._build_factor_long_frame(
            pl.concat(frames, how="diagonal_relaxed")
        )
        matrix = build_decision_feature_matrix(factors, decision_timestamp, list(requirements.features))
        return PreparedMarket(matrix, len(pairs), errors)


def build_decision_feature_matrix(
    factors: pl.DataFrame,
    decision_timestamp: int,
    feature_names: list[str],
) -> pl.DataFrame:
    selected = factors.filter(
        (pl.col("timestamp") == decision_timestamp)
        & pl.col("factor_name").is_in(feature_names)
    )
    if selected.is_empty():
        return pl.DataFrame(schema={"symbol": pl.String, **{name: pl.Float64 for name in feature_names}})
    wide = selected.pivot(index="symbol", on="factor_name", values="factor_value")
    for name in feature_names:
        if name not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    return (
        wide.filter(pl.any_horizontal([pl.col(name).is_not_null() for name in feature_names]))
        .select(["symbol", *feature_names])
        .sort("symbol")
    )

