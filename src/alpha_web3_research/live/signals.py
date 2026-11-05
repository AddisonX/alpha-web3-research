from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ccxt
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from alpha_web3_research.config import Settings
from alpha_web3_research.data.ohlcv import OHLCVFetcher
from alpha_web3_research.factors.builder import FactorBuilder

HOUR_MS = 3_600_000


@dataclass(frozen=True)
class LiveSignalResult:
    decision_timestamp: int
    signal_path: Path
    status_path: Path
    cross_section_count: int
    selected_count: int
    selected_pairs: list[str]
    skipped: bool


class ProductionEnsemble:
    def __init__(self, production_root: Path) -> None:
        latest_path = production_root / "latest.json"
        if not latest_path.exists():
            raise FileNotFoundError(f"production model pointer not found: {latest_path}")
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        self.bundle_dir = production_root / str(latest["bundle"])
        self.manifest_path = self.bundle_dir / "manifest.json"
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        source_config = manifest["source_config"]
        self.features = [str(name) for name in source_config["features"]]
        self.timeframe = str(source_config["timeframe"])
        self.horizon_bars = int(source_config["horizon_bars"])
        self.label_column = str(source_config["label_column"])
        self.top_quantile_fraction = float(source_config["top_quantile_fraction"])
        self.source_experiment = str(manifest["source_experiment"])
        self.boosters = [
            lgb.Booster(model_file=str(self.bundle_dir / str(item["file"])))
            for item in manifest["models"]
        ]
        if not self.boosters:
            raise ValueError(f"production bundle contains no models: {self.manifest_path}")
        for booster in self.boosters:
            if booster.feature_name() != self.features:
                raise ValueError("production model feature order does not match the manifest")

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        matrix = features.select(self.features).to_pandas()
        predictions = [booster.predict(matrix) for booster in self.boosters]
        return np.mean(np.vstack(predictions), axis=0)


class LiveLongOnlySignalProducer:
    def __init__(
        self,
        settings: Settings,
        exchange: ccxt.Exchange,
        pairs_path: Path,
        signal_path: Path,
        status_path: Path,
        production_root: Path | None = None,
        min_cross_section_count: int = 60,
        lookback_bars: int = 40,
        rebalance_interval_hours: int = 24,
        rebalance_hour_utc: int = 0,
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

    def run(self, force: bool = False, now_ms: int | None = None) -> LiveSignalResult:
        model = ProductionEnsemble(self.production_root)
        if model.timeframe != "4h":
            raise ValueError(f"live producer currently requires a 4h production model, got {model.timeframe}")
        timeframe_ms = int(self.exchange.parse_timeframe(model.timeframe) * 1000)
        current_ms = int(now_ms if now_ms is not None else self.exchange.milliseconds())
        decision_timestamp = latest_closed_decision_timestamp(
            now_ms=current_ms,
            timeframe_ms=timeframe_ms,
            rebalance_interval_hours=self.rebalance_interval_hours,
            rebalance_hour_utc=self.rebalance_hour_utc,
        )
        previous_status = self._read_status()
        if (
            not force
            and previous_status.get("state") == "healthy"
            and int(previous_status.get("decision_timestamp", -1)) == decision_timestamp
            and self.signal_path.exists()
        ):
            return LiveSignalResult(
                decision_timestamp=decision_timestamp,
                signal_path=self.signal_path,
                status_path=self.status_path,
                cross_section_count=int(previous_status["cross_section_count"]),
                selected_count=int(previous_status["selected_count"]),
                selected_pairs=[str(pair) for pair in previous_status["selected_pairs"]],
                skipped=True,
            )

        pairs = self._read_pairs()
        fetcher = OHLCVFetcher(exchange=self.exchange, settings=self.settings)
        since_ms = decision_timestamp - (self.lookback_bars - 1) * timeframe_ms
        until_ms = decision_timestamp + timeframe_ms
        frames: list[pl.DataFrame] = []
        errors: dict[str, str] = {}
        for pair in pairs:
            try:
                frame = fetcher.fetch_ohlcv_frame(
                    symbol=pair,
                    timeframe=model.timeframe,
                    since=_iso_timestamp(since_ms),
                    until=_iso_timestamp(until_ms),
                    limit_per_request=1500,
                    include_unclosed=False,
                )
                if frame.filter(pl.col("timestamp") == decision_timestamp).is_empty():
                    raise ValueError("decision candle is missing")
                if frame.height < 25:
                    raise ValueError(f"only {frame.height} closed candles available")
                frames.append(frame)
            except Exception as exc:
                errors[pair] = f"{type(exc).__name__}: {exc}"

        if not frames:
            self._write_failure_status(
                decision_timestamp=decision_timestamp,
                model=model,
                errors=errors,
                reason="no OHLCV frames were fetched",
            )
            raise RuntimeError("live signal generation failed: no OHLCV frames were fetched")

        ohlcv = pl.concat(frames, how="diagonal_relaxed")
        factors = FactorBuilder(settings=self.settings)._build_factor_long_frame(ohlcv)
        matrix = build_decision_feature_matrix(
            factors=factors,
            decision_timestamp=decision_timestamp,
            feature_names=model.features,
        )
        if matrix.height < self.min_cross_section_count:
            self._write_failure_status(
                decision_timestamp=decision_timestamp,
                model=model,
                errors=errors,
                reason=(
                    f"cross section has {matrix.height} symbols, "
                    f"below minimum {self.min_cross_section_count}"
                ),
            )
            raise RuntimeError(
                f"live signal generation failed: cross section {matrix.height} "
                f"< minimum {self.min_cross_section_count}"
            )

        scores = model.predict(matrix)
        ranked = rank_long_only_scores(
            pairs=matrix.get_column("symbol").to_list(),
            scores=scores,
            top_quantile_fraction=model.top_quantile_fraction,
        )
        generated_at = datetime.now(timezone.utc).isoformat()
        timeline = build_signal_timeline(
            ranked=ranked,
            decision_timestamp=decision_timestamp,
            generated_at=generated_at,
            timeframe_ms=timeframe_ms,
            rebalance_interval_hours=self.rebalance_interval_hours,
            source_experiment=model.source_experiment,
        )
        history = self._merge_history(timeline)
        _atomic_write_parquet(self.signal_path, history)

        selected_pairs = (
            ranked.filter(pl.col("target_position") == 1)
            .sort("score_position")
            .get_column("pair")
            .to_list()
        )
        status = {
            "state": "healthy",
            "generated_at": generated_at,
            "market_data_source": (
                "binance_futures_testnet"
                if "testnet.binancefuture.com"
                in json.dumps(self.exchange.urls.get("api", {}))
                else "binance_futures"
            ),
            "decision_timestamp": decision_timestamp,
            "decision_time": _iso_timestamp(decision_timestamp),
            "next_rebalance_timestamp": decision_timestamp + self.rebalance_interval_hours * HOUR_MS,
            "next_rebalance_time": _iso_timestamp(
                decision_timestamp + self.rebalance_interval_hours * HOUR_MS
            ),
            "grace_valid_until": (
                decision_timestamp
                + self.rebalance_interval_hours * HOUR_MS
                + timeframe_ms
            ),
            "grace_valid_until_time": _iso_timestamp(
                decision_timestamp
                + self.rebalance_interval_hours * HOUR_MS
                + timeframe_ms
            ),
            "source_experiment": model.source_experiment,
            "model_count": len(model.boosters),
            "timeframe": model.timeframe,
            "horizon_bars": model.horizon_bars,
            "label_column": model.label_column,
            "requested_pair_count": len(pairs),
            "cross_section_count": ranked.height,
            "selected_count": len(selected_pairs),
            "selected_pairs": selected_pairs,
            "fetch_error_count": len(errors),
            "fetch_errors": errors,
        }
        _atomic_write_json(self.status_path, status)
        return LiveSignalResult(
            decision_timestamp=decision_timestamp,
            signal_path=self.signal_path,
            status_path=self.status_path,
            cross_section_count=ranked.height,
            selected_count=len(selected_pairs),
            selected_pairs=selected_pairs,
            skipped=False,
        )

    def _read_pairs(self) -> list[str]:
        pairs = json.loads(self.pairs_path.read_text(encoding="utf-8"))
        if not isinstance(pairs, list) or not pairs:
            raise ValueError(f"pair list is empty or invalid: {self.pairs_path}")
        return sorted({str(pair) for pair in pairs})

    def _read_status(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return {}
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _merge_history(self, current: pl.DataFrame) -> pl.DataFrame:
        if not self.signal_path.exists():
            return current.sort(["date", "pair"])
        history = pl.read_parquet(self.signal_path)
        cutoff = current.get_column("date").min() - pd.Timedelta(days=45)
        return (
            pl.concat([history.filter(pl.col("date") >= cutoff), current], how="diagonal_relaxed")
            .unique(subset=["date", "pair"], keep="last")
            .sort(["date", "pair"])
        )

    def _write_failure_status(
        self,
        decision_timestamp: int,
        model: ProductionEnsemble,
        errors: dict[str, str],
        reason: str,
    ) -> None:
        _atomic_write_json(
            self.status_path,
            {
                "state": "failed",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "decision_timestamp": decision_timestamp,
                "decision_time": _iso_timestamp(decision_timestamp),
                "source_experiment": model.source_experiment,
                "reason": reason,
                "fetch_error_count": len(errors),
                "fetch_errors": errors,
            },
        )


def latest_closed_decision_timestamp(
    now_ms: int,
    timeframe_ms: int,
    rebalance_interval_hours: int,
    rebalance_hour_utc: int,
) -> int:
    latest_closed_open = (now_ms // timeframe_ms) * timeframe_ms - timeframe_ms
    interval_ms = rebalance_interval_hours * HOUR_MS
    offset_ms = rebalance_hour_utc * HOUR_MS
    return latest_closed_open - ((latest_closed_open - offset_ms) % interval_ms)


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


def rank_long_only_scores(
    pairs: list[str],
    scores: np.ndarray,
    top_quantile_fraction: float,
) -> pl.DataFrame:
    if len(pairs) != len(scores):
        raise ValueError("pair and score counts do not match")
    selected_count = max(1, math.ceil(len(pairs) * top_quantile_fraction))
    return (
        pl.DataFrame({"pair": pairs, "model_score": scores})
        .sort(["model_score", "pair"], descending=[True, False])
        .with_row_index("score_position", offset=1)
        .with_columns(
            pl.lit(len(pairs)).alias("cross_section_count"),
            pl.lit(selected_count).alias("selected_count"),
            (
                1.0
                - (pl.col("score_position") - 1)
                / max(1, len(pairs) - 1)
            ).alias("model_score_rank_pct"),
            (pl.col("score_position") <= selected_count).cast(pl.Int8).alias("target_position"),
        )
    )


def build_signal_timeline(
    ranked: pl.DataFrame,
    decision_timestamp: int,
    generated_at: str,
    timeframe_ms: int,
    rebalance_interval_hours: int,
    source_experiment: str,
) -> pl.DataFrame:
    next_rebalance = decision_timestamp + rebalance_interval_hours * HOUR_MS
    grace_valid_until = next_rebalance + timeframe_ms
    timestamps = list(range(decision_timestamp, grace_valid_until, timeframe_ms))
    timeline = pl.DataFrame({"timestamp": timestamps})
    return (
        ranked.join(timeline, how="cross")
        .with_columns(
            pl.from_epoch("timestamp", time_unit="ms").dt.replace_time_zone("UTC").alias("date"),
            pl.lit(generated_at).alias("generated_at"),
            pl.lit(decision_timestamp).alias("decision_timestamp"),
            pl.lit(grace_valid_until).alias("signal_valid_until"),
            pl.lit(True).alias("signal_valid"),
            (pl.col("timestamp") == decision_timestamp).alias("is_rebalance"),
            pl.lit(source_experiment).alias("source_experiment"),
        )
        .select(
            "date",
            "timestamp",
            "pair",
            "generated_at",
            "source_experiment",
            "decision_timestamp",
            "signal_valid_until",
            "signal_valid",
            "is_rebalance",
            "target_position",
            "model_score",
            "model_score_rank_pct",
            "score_position",
            "cross_section_count",
            "selected_count",
        )
        .sort(["date", "pair"])
    )


def _iso_timestamp(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat()


def _atomic_write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
