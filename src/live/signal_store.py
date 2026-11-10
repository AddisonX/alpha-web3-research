from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import polars as pl
from src.strategies.base import StrategyDecision, StrategyMetadata, DataRequirements

HOUR_MS = 3_600_000

class SignalStore:
    """Translate complete binary targets into the existing Freqtrade file contract."""

    def __init__(self, signal_path: Path, status_path: Path) -> None:
        self.signal_path = signal_path
        self.status_path = status_path

    def read_status(self) -> dict[str, Any]:
        try:
            status = json.loads(self.status_path.read_text(encoding="utf-8"))
            return status if isinstance(status, dict) else {}
        except (OSError, ValueError):
            return {}

    def write_failure(self, strategy_id: str, decision_timestamp: int, metadata: StrategyMetadata | None,
                      errors: dict[str, str], reason: str) -> None:
        _atomic_write_json(self.status_path, {
            "state": "failed", "strategy_id": strategy_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "decision_timestamp": decision_timestamp, "decision_time": _iso_timestamp(decision_timestamp),
            "source_experiment": metadata.source_experiment if metadata else "",
            "reason": reason, "fetch_error_count": len(errors), "fetch_errors": errors,
        })

    def publish(self, strategy_id: str, decision: StrategyDecision, requirements: DataRequirements,
                metadata: StrategyMetadata, market_data_source: str, requested_pair_count: int,
                errors: dict[str, str], rebalance_interval_hours: int) -> list[str]:
        ranked = self._execution_columns(decision.targets)
        generated_at = datetime.now(timezone.utc).isoformat()
        stamp = decision.decision_timestamp
        timeframe_ms = 4 * HOUR_MS
        timeline = build_signal_timeline(ranked, stamp, generated_at, timeframe_ms,
                                         rebalance_interval_hours, metadata.source_experiment)
        _atomic_write_parquet(self.signal_path, self._merge_history(timeline))
        selected = ranked.filter(pl.col("target_position") == 1).sort("score_position")["pair"].to_list()
        next_rebalance = stamp + rebalance_interval_hours * HOUR_MS
        valid_until = next_rebalance + timeframe_ms
        _atomic_write_json(self.status_path, {
            "state": "healthy", "strategy_id": strategy_id, "generated_at": generated_at,
            "market_data_source": market_data_source,
            "decision_timestamp": stamp, "decision_time": _iso_timestamp(stamp),
            "next_rebalance_timestamp": next_rebalance, "next_rebalance_time": _iso_timestamp(next_rebalance),
            "grace_valid_until": valid_until, "grace_valid_until_time": _iso_timestamp(valid_until),
            "source_experiment": metadata.source_experiment, "model_count": metadata.model_count,
            "timeframe": requirements.timeframe, "horizon_bars": metadata.horizon_bars,
            "label_column": metadata.label_column, "requested_pair_count": requested_pair_count,
            "cross_section_count": ranked.height, "selected_count": len(selected), "selected_pairs": selected,
            "fetch_error_count": len(errors), "fetch_errors": errors,
        })
        return selected

    @staticmethod
    def _execution_columns(targets: pl.DataFrame) -> pl.DataFrame:
        # Model diagnostics are optional; keep the legacy execution columns available.
        frame = targets
        if "score_position" not in frame.columns:
            frame = frame.sort(["target_position", "pair"], descending=[True, False]).with_row_index("score_position", offset=1)
        if "model_score" not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("model_score"))
        if "model_score_rank_pct" not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("model_score_rank_pct"))
        return frame.with_columns(
            pl.lit(frame.height).alias("cross_section_count"),
            pl.lit(frame.filter(pl.col("target_position") == 1).height).alias("selected_count"),
        )

    def _merge_history(self, current: pl.DataFrame) -> pl.DataFrame:
        if not self.signal_path.exists():
            return current.sort(["date", "pair"])
        history = pl.read_parquet(self.signal_path)
        cutoff = current["date"].min() - pd.Timedelta(days=45)
        return (pl.concat([history.filter(pl.col("date") >= cutoff), current], how="diagonal_relaxed")
                .unique(subset=["date", "pair"], keep="last").sort(["date", "pair"]))


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

