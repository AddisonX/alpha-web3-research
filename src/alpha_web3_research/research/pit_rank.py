from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

import polars as pl
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from alpha_web3_research.config import Settings
from alpha_web3_research.data.binance_archive import BinanceArchiveClient, BAR_MS, native_to_symbol
from alpha_web3_research.data.time import parse_timestamp_ms
from alpha_web3_research.strategies.pit_rank_excess import STRATEGY_ID, STRATEGY_KEY
from alpha_web3_research.universe.point_in_time import (
    PointInTimeUniverseRules, audit_universe_survival, build_point_in_time_universe,
)

DAY_MS = 86_400_000


class PITRankResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    strategy_key: str = STRATEGY_KEY
    inception: str = "2025-01-01T00:00:00Z"
    history_start: str = "2024-07-01T00:00:00Z"
    train_start: str = Field(default="2025-01-01T00:00:00Z",
                             validation_alias=AliasChoices("train_start", "development_start"))
    validation_start: str = "2026-04-01T00:00:00Z"
    test_start: str = "2026-05-01T00:00:00Z"
    requested_end: str = "2026-11-01T00:00:00Z"
    data_namespace: str = "pit_rank64_202501_202610"
    features_file: str = "strategies/features/pit_rank64.v1.json"
    min_history_days: int = Field(default=180, ge=30)
    liquidity_lookback_days: int = Field(default=30, ge=1)
    min_median_daily_quote_volume: float = Field(default=5_000_000, ge=0)
    min_history_coverage: float = Field(default=0.98, gt=0, le=1)
    max_symbols: int = Field(default=120, ge=2)
    min_cross_section: int = Field(default=20, ge=2)
    training_protocol: Literal["fixed_train_valid_test"] = "fixed_train_valid_test"
    early_stopping_rounds: int = Field(default=40, ge=1)
    decision_hour_utc: int = Field(default=4, ge=0, le=23)
    fraction: float = Field(default=0.10, gt=0, le=1)
    stop_loss: float = Field(default=0.10, gt=0, lt=1)
    one_way_fee_rate: float = Field(default=0.0005, ge=0)
    one_way_slippage_rate: float = Field(default=0.0002, ge=0)
    seed: int = 42
    n_estimators: int = Field(default=500, ge=1,
                              validation_alias=AliasChoices("n_estimators", "max_estimators"))
    num_leaves: int = Field(default=15, ge=2)
    min_child_samples: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=0.03, gt=0, le=1)
    reg_alpha: float = Field(default=0.1, ge=0)
    reg_lambda: float = Field(default=10.0, ge=0)

    @model_validator(mode="after")
    def validate_dates(self) -> PITRankResearchConfig:
        dates = [parse_timestamp_ms(value) for value in (
            self.history_start, self.inception, self.train_start, self.validation_start, self.test_start, self.requested_end)]
        if not dates[0] < dates[1] <= dates[2] < dates[3] < dates[4] < dates[5]:
            raise ValueError("require history_start < inception <= train_start < validation_start < test_start < requested_end")
        if dates[1] - dates[0] < self.min_history_days * DAY_MS:
            raise ValueError("history_start must allow the complete inception history window")
        if self.strategy_key != STRATEGY_KEY or Path(self.data_namespace).name != self.data_namespace:
            raise ValueError("strategy key or isolated data namespace is invalid")
        if self.decision_hour_utc % 4:
            raise ValueError("decision_hour_utc must lie on a native 4h bar boundary")
        return self

    def data_contract(self) -> dict[str, object]:
        """Only acquisition/universe choices determine whether raw bars can be reused."""
        return self.model_dump(include={
            "inception", "history_start", "requested_end", "data_namespace", "min_history_days",
            "liquidity_lookback_days", "min_median_daily_quote_volume", "min_history_coverage", "max_symbols",
        })

    @classmethod
    def load(cls, path: Path) -> PITRankResearchConfig:
        return cls.model_validate_json(path.read_text())


def research_root(settings: Settings, config: PITRankResearchConfig) -> Path:
    return settings.resolved_data_path() / "research" / config.data_namespace


def read_bars(path: Path) -> pl.DataFrame:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise ValueError(f"no archive bars found at {path}")
    return pl.concat([pl.read_parquet(file) for file in files], how="vertical_relaxed").sort(["symbol", "timestamp"])


def prepare_pit_rank_data(
    settings: Settings, config: PITRankResearchConfig, workers: int = 12,
    progress: Callable[[str], None] = print,
) -> Path:
    root = research_root(settings, config)
    root.mkdir(parents=True, exist_ok=True)
    client = BinanceArchiveClient(settings.resolved_data_path() / "raw/binance_archive", workers)
    index = client.historical_index(root / "archive_index.json", progress)
    inception = parse_timestamp_ms(config.inception)
    history_start = parse_timestamp_ms(config.history_start)
    requested_end = parse_timestamp_ms(config.requested_end)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    effective_end = min(requested_end, now_ms - now_ms % BAR_MS)
    first_month = datetime.fromtimestamp(history_start / 1000, timezone.utc).strftime("%Y-%m")
    inception_month = datetime.fromtimestamp(inception / 1000, timezone.utc).strftime("%Y-%m")
    last_month = datetime.fromtimestamp((effective_end - 1) / 1000, timezone.utc).strftime("%Y-%m")
    inception_requests = {
        native: [key for key in keys if first_month <= key.rsplit("-", 2)[-2] + "-" + key.rsplit("-", 1)[-1][:2] < inception_month]
        for native, keys in index.items()
    }
    inception_requests = {native: keys for native, keys in inception_requests.items() if keys}
    # Use filename dates, never today's exchange active flag, to discover inception candidates.
    inception_requests = {
        native: keys for native, keys in inception_requests.items()
        if keys[0].split("-")[-2] + "-" + keys[0].split("-")[-1][:2] <= first_month
    }
    progress(f"Downloading pre-inception history for {len(inception_requests)} archived instruments")
    client.fetch_many(inception_requests, history_start, inception, root / "inception_bars", progress)
    universe = build_point_in_time_universe(
        read_bars(root / "inception_bars"), inception,
        PointInTimeUniverseRules(
            config.min_history_days, config.liquidity_lookback_days,
            config.min_median_daily_quote_volume, config.min_history_coverage, config.max_symbols,
        ),
    )
    universe.write_parquet(root / "universe.parquet")
    included = universe.filter(pl.col("included"))
    if included.height < config.min_cross_section:
        raise ValueError(f"inception universe too small: {included.height}")
    included_symbols = set(included["symbol"].to_list())
    natives = sorted(native for native in index if native_to_symbol(native) in included_symbols)
    requests = {
        native: [key for key in index[native]
                 if first_month <= key.split("-")[-2] + "-" + key.split("-")[-1][:2] <= last_month]
        for native in natives
    }
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(client.list_objects,
                        f"data/futures/um/daily/klines/{native}/4h/{native}-4h-{last_month}-"): native
            for native in natives
            if not any(key.endswith(f"-{last_month}.zip") for key in requests[native])
        }
        for future in as_completed(futures):
            native = futures[future]
            requests[native].extend(key for key in future.result() if key.endswith(".zip"))
    progress(f"Inception universe fixed at {included.height}; extending its bars through {last_month}")
    sources = client.fetch_many(requests, history_start, effective_end, root / "bars", progress)
    bars = read_bars(root / "bars")
    actual_end = bars["timestamp"].max() + BAR_MS
    audit = audit_universe_survival(universe, bars, actual_end)
    audit.write_parquet(root / "survival_audit.parquet")
    report = {
        "strategy_key": config.strategy_key, "config": config.model_dump(), "data_contract": config.data_contract(),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": universe.height, "universe_count": included.height,
        "requested_end_ms": requested_end, "closed_bar_cutoff_ms": effective_end,
        "actual_last_bar_close_ms": actual_end, "inception_as_of_ms": inception,
        "symbols_with_data_at_archive_end": audit.filter(pl.col("data_reaches_audit_end")).height,
        "future_survival_used_for_selection": False, "sources": sources,
    }
    (root / "data_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    progress(f"Prepared {bars.height:,} bars; audit data reaches latest archive for "
             f"{report['symbols_with_data_at_archive_end']}/{included.height} inception members")
    return root
