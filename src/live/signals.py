from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import ccxt

from src.config import Settings
from src.live.paths import instance_directory
# Existing helper imports remain available while their implementations move out.
from src.live.market import MarketDataError, MarketDataProvider, MarketDataSource, build_decision_feature_matrix
from src.live.signal_store import HOUR_MS, SignalStore, build_signal_timeline
from src.strategies.base import DEFAULT_STRATEGY_ID, LongOnlyStrategy, StrategyContext, StrategyMetadata
from src.strategies.registry import create_strategy


@dataclass(frozen=True)
class LiveSignalResult:
    decision_timestamp: int
    signal_path: Path
    status_path: Path
    cross_section_count: int
    selected_count: int
    selected_pairs: list[str]
    skipped: bool


class LiveLongOnlySignalProducer:
    """Schedule data preparation, decision and publication; never select coins here."""

    def __init__(
        self, settings: Settings, exchange: ccxt.Exchange, pairs_path: Path,
        signal_path: Path, status_path: Path, production_root: Path | None = None,
        min_cross_section_count: int = 60, lookback_bars: int | None = None,
        rebalance_interval_hours: int = 24, rebalance_hour_utc: int = 0,
        strategy_id: str = DEFAULT_STRATEGY_ID, strategy: LongOnlyStrategy | None = None,
        data_provider: MarketDataSource | None = None,
    ) -> None:
        self.settings = settings
        self.exchange = exchange
        self.pairs_path = pairs_path
        self.signal_path = signal_path
        self.status_path = status_path
        self.production_root = production_root or settings.resolved_data_path() / "models" / "production"
        self.min_cross_section_count = min_cross_section_count
        self.lookback_bars = lookback_bars
        self.rebalance_interval_hours = rebalance_interval_hours
        self.rebalance_hour_utc = rebalance_hour_utc
        self.strategy_id = strategy.strategy_id if strategy is not None else strategy_id
        self.strategy = strategy
        self.data_provider = data_provider or MarketDataProvider(settings, exchange)
        self.store = SignalStore(signal_path, status_path)
        if self.strategy_id != DEFAULT_STRATEGY_ID:
            default_dir = instance_directory(settings.project_root) / "signals"
            if signal_path.resolve() == (default_dir / "alpha_live_long_only_signals.parquet").resolve() or status_path.resolve() == (default_dir / "alpha_live_long_only_status.json").resolve():
                raise ValueError("alternative strategies must use separate signal and status paths")

    def run(self, force: bool = False, now_ms: int | None = None) -> LiveSignalResult:
        previous = self.store.read_status()
        owner = previous.get("strategy_id", DEFAULT_STRATEGY_ID)
        if previous and owner != self.strategy_id:
            raise ValueError("signal directory belongs to a different strategy")
        strategy = None
        metadata = None
        errors: dict[str, str] = {}
        decision_timestamp = 0
        try:
            strategy = self.strategy or create_strategy(self.strategy_id, self.production_root)
            metadata = strategy.metadata
            if not isinstance(metadata, StrategyMetadata):
                raise ValueError("strategy must provide StrategyMetadata")
            requirements = strategy.requirements
            if requirements.timeframe != "4h":
                raise ValueError("the current Freqtrade adapter supports only 4h signals")
            if self.lookback_bars is not None:
                requirements = replace(requirements, lookback_bars=self.lookback_bars)
            timeframe_ms = int(self.exchange.parse_timeframe(requirements.timeframe) * 1000)
            decision_timestamp = latest_closed_decision_timestamp(
                int(now_ms if now_ms is not None else self.exchange.milliseconds()),
                timeframe_ms, self.rebalance_interval_hours, self.rebalance_hour_utc,
            )
            if (not force and previous.get("state") == "healthy"
                    and int(previous.get("decision_timestamp", -1)) == decision_timestamp
                    and self.signal_path.exists()):
                return LiveSignalResult(decision_timestamp, self.signal_path, self.status_path,
                                        int(previous["cross_section_count"]), int(previous["selected_count"]),
                                        list(previous["selected_pairs"]), True)
            pairs = self._read_pairs()
            market = self.data_provider.prepare(pairs, decision_timestamp, requirements)
            errors = market.errors
            if market.features.height < self.min_cross_section_count:
                raise RuntimeError(f"cross section {market.features.height} < minimum {self.min_cross_section_count}")
            context = StrategyContext(decision_timestamp, market.features)
            decision = strategy.decide(context)
            decision.validate(context)
            selected = self.store.publish(
                self.strategy_id, decision, requirements, metadata,
                "binance_futures_testnet" if "testnet.binancefuture.com" in json.dumps(self.exchange.urls.get("api", {})) else "binance_futures",
                market.requested_pair_count, errors, self.rebalance_interval_hours,
            )
            return LiveSignalResult(decision_timestamp, self.signal_path, self.status_path,
                                    market.features.height, len(selected), selected, False)
        except Exception as exc:
            if isinstance(exc, MarketDataError):
                errors = exc.errors
            # Never publish an empty target on failure; freeze via unhealthy status.
            self.store.write_failure(self.strategy_id, decision_timestamp,
                                     metadata if isinstance(metadata, StrategyMetadata) else None,
                                     errors, f"{type(exc).__name__}: {exc}")
            raise

    def _read_pairs(self) -> list[str]:
        pairs = json.loads(self.pairs_path.read_text(encoding="utf-8"))
        if not isinstance(pairs, list) or not pairs:
            raise ValueError(f"pair list is empty or invalid: {self.pairs_path}")
        return sorted({str(pair) for pair in pairs})


def latest_closed_decision_timestamp(now_ms: int, timeframe_ms: int,
                                    rebalance_interval_hours: int, rebalance_hour_utc: int) -> int:
    latest_closed_open = (now_ms // timeframe_ms) * timeframe_ms - timeframe_ms
    interval_ms = rebalance_interval_hours * HOUR_MS
    offset_ms = rebalance_hour_utc * HOUR_MS
    return latest_closed_open - ((latest_closed_open - offset_ms) % interval_ms)
