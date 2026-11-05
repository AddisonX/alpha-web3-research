from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import numpy as np

from alpha_web3_research.config import Settings
from alpha_web3_research.data.storage import ParquetOHLCVStore


@dataclass(frozen=True)
class FactorDecorrelationResult:
    selected_path: Path
    decisions_path: Path
    report_path: Path
    selected_count: int
    candidate_count: int


class FactorDecorrelator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_root = settings.resolved_data_path() / "evaluation"
        self.report_root = settings.resolved_reports_path()
        self.factors_path = settings.resolved_data_path() / "factors" / "latest_factors.parquet"
        self.summary_path = self.data_root / "single_factor_summary.parquet"
        self.timeseries_path = self.data_root / "single_factor_timeseries.parquet"

    def run(
        self,
        max_value_corr: float = 0.85,
        max_ic_corr: float = 0.85,
        min_abs_rank_ic: float = 0.0,
        min_samples: int = 1,
        max_factors_per_group: int | None = None,
    ) -> FactorDecorrelationResult:
        summary = self._read_required(self.summary_path, "single factor summary")
        timeseries = self._read_required(self.timeseries_path, "single factor timeseries")
        if not self.factors_path.exists():
            raise FileNotFoundError(f"factors parquet not found: {self.factors_path}")

        candidates = (
            summary.filter(
                pl.col("mean_rank_ic").is_not_null()
                & (pl.col("sample_count") >= min_samples)
                & (pl.col("abs_mean_rank_ic") >= min_abs_rank_ic)
            )
            .with_columns(
                (
                    pl.col("abs_mean_rank_ic")
                    * pl.max_horizontal(
                        pl.col("positive_rank_ic_rate").fill_null(0.0),
                        1.0 - pl.col("positive_rank_ic_rate").fill_null(1.0),
                    )
                ).alias("selection_score")
            )
            .sort(["timeframe", "horizon_bars", "selection_score", "abs_mean_rank_ic"], descending=[False, False, True, True])
        )

        decisions: list[dict[str, object]] = []
        selected_rows: list[dict[str, object]] = []
        generated_at = datetime.now(timezone.utc).isoformat()

        for group in candidates.select(["timeframe", "horizon_bars"]).unique().sort(["timeframe", "horizon_bars"]).iter_rows(named=True):
            timeframe = group["timeframe"]
            horizon_bars = group["horizon_bars"]
            group_candidates = candidates.filter((pl.col("timeframe") == timeframe) & (pl.col("horizon_bars") == horizon_bars))
            factor_names = group_candidates.get_column("factor_name").to_list()
            value_corr = self._factor_value_corr(timeframe=timeframe, factor_names=factor_names)
            ic_corr = self._ic_corr(timeseries=timeseries, timeframe=timeframe, horizon_bars=horizon_bars, factor_names=factor_names)

            selected: list[str] = []
            for row in group_candidates.iter_rows(named=True):
                factor_name = row["factor_name"]
                max_value, value_peer = max_abs_corr_against_selected(value_corr, factor_name, selected)
                max_ic, ic_peer = max_abs_corr_against_selected(ic_corr, factor_name, selected)
                reason = ""
                if max_value is not None and max_value > max_value_corr:
                    reason = f"value_corr>{max_value_corr} with {value_peer}"
                elif max_ic is not None and max_ic > max_ic_corr:
                    reason = f"ic_corr>{max_ic_corr} with {ic_peer}"
                elif max_factors_per_group is not None and len(selected) >= max_factors_per_group:
                    reason = f"max_factors_per_group={max_factors_per_group}"

                is_selected = reason == ""
                if is_selected:
                    selected.append(str(factor_name))
                    selected_rows.append({**row, "generated_at": generated_at})

                decisions.append(
                    {
                        **row,
                        "generated_at": generated_at,
                        "selected": is_selected,
                        "selection_order": len(selected) if is_selected else None,
                        "exclusion_reason": reason,
                        "max_abs_value_corr_to_selected": max_value,
                        "max_value_corr_peer": value_peer,
                        "max_abs_ic_corr_to_selected": max_ic,
                        "max_ic_corr_peer": ic_peer,
                    }
                )

        decision_frame = pl.DataFrame(decisions) if decisions else pl.DataFrame()
        selected_frame = pl.DataFrame(selected_rows) if selected_rows else pl.DataFrame()

        self.data_root.mkdir(parents=True, exist_ok=True)
        self.report_root.mkdir(parents=True, exist_ok=True)
        decisions_path = self.data_root / "decorrelation_decisions.parquet"
        selected_path = self.data_root / "decorrelated_factors.parquet"
        report_path = self.report_root / "decorrelated_factors_report.md"
        decision_frame.write_parquet(decisions_path)
        selected_frame.write_parquet(selected_path)
        self._write_report(
            decisions=decision_frame,
            selected=selected_frame,
            report_path=report_path,
            max_value_corr=max_value_corr,
            max_ic_corr=max_ic_corr,
            min_abs_rank_ic=min_abs_rank_ic,
            min_samples=min_samples,
        )
        ParquetOHLCVStore(settings=self.settings).create_duckdb_database()
        return FactorDecorrelationResult(
            selected_path=selected_path,
            decisions_path=decisions_path,
            report_path=report_path,
            selected_count=selected_frame.height,
            candidate_count=decision_frame.height,
        )

    def _factor_value_corr(self, timeframe: str, factor_names: list[str]) -> dict[tuple[str, str], float]:
        if not factor_names:
            return {}
        frame = (
            pl.scan_parquet(self.factors_path)
            .filter((pl.col("timeframe") == timeframe) & pl.col("factor_name").is_in(factor_names))
            .select(["symbol", "timestamp", "factor_name", "factor_value"])
            .collect()
        )
        if frame.is_empty():
            return {}
        wide = frame.pivot(index=["symbol", "timestamp"], on="factor_name", values="factor_value")
        return wide_corr_to_dict(wide.select([name for name in factor_names if name in wide.columns]), min_periods=100)

    def _ic_corr(self, timeseries: pl.DataFrame, timeframe: str, horizon_bars: int, factor_names: list[str]) -> dict[tuple[str, str], float]:
        frame = timeseries.filter(
            (pl.col("timeframe") == timeframe)
            & (pl.col("horizon_bars") == horizon_bars)
            & pl.col("factor_name").is_in(factor_names)
        ).select(["timestamp", "factor_name", "rank_ic"])
        if frame.is_empty():
            return {}
        wide = frame.pivot(index="timestamp", on="factor_name", values="rank_ic")
        return wide_corr_to_dict(wide.select([name for name in factor_names if name in wide.columns]), min_periods=20)

    def _write_report(
        self,
        decisions: pl.DataFrame,
        selected: pl.DataFrame,
        report_path: Path,
        max_value_corr: float,
        max_ic_corr: float,
        min_abs_rank_ic: float,
        min_samples: int,
    ) -> None:
        lines = [
            "# Factor Decorrelation Report",
            "",
            f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
            f"- Max factor value correlation: `{max_value_corr}`",
            f"- Max IC correlation: `{max_ic_corr}`",
            f"- Minimum |RankIC|: `{min_abs_rank_ic}`",
            f"- Minimum samples: `{min_samples}`",
            "",
        ]
        if decisions.is_empty():
            lines.append("No candidates passed the pre-filters.")
            report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return

        counts = decisions.group_by(["timeframe", "horizon_bars"]).agg(
            pl.len().alias("candidates"),
            pl.col("selected").sum().alias("selected"),
        ).sort(["timeframe", "horizon_bars"])
        lines.extend(["## Selection Counts", "", "| timeframe | horizon | candidates | selected |", "|---|---:|---:|---:|"])
        for row in counts.iter_rows(named=True):
            lines.append(f"| {row['timeframe']} | {row['horizon_bars']} | {row['candidates']} | {row['selected']} |")

        lines.extend(["", "## Selected Factors", "", "| factor | family | timeframe | horizon | score | mean_rank_ic | sample_count |", "|---|---|---|---:|---:|---:|---:|"])
        if not selected.is_empty():
            for row in selected.sort(["timeframe", "horizon_bars", "selection_score"], descending=[False, False, True]).head(50).iter_rows(named=True):
                lines.append(
                    f"| {row['factor_name']} | {row['factor_family']} | {row['timeframe']} | {row['horizon_bars']} | "
                    f"{fmt(row['selection_score'])} | {fmt(row['mean_rank_ic'])} | {row['sample_count']} |"
                )

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _read_required(path: Path, name: str) -> pl.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"{name} parquet not found: {path}")
        return pl.read_parquet(path)


def wide_corr_to_dict(frame: pl.DataFrame, min_periods: int) -> dict[tuple[str, str], float]:
    pandas_frame = frame.to_pandas().replace([np.inf, -np.inf], np.nan)
    corr_frame = pandas_frame.corr(min_periods=min_periods)
    columns = list(corr_frame.columns)
    result: dict[tuple[str, str], float] = {}
    for left in columns:
        for right in columns:
            value = corr_frame.at[left, right]
            if value is None or float(value) != float(value):
                continue
            result[(left, right)] = float(value)
    return result


def max_abs_corr_against_selected(
    corr: dict[tuple[str, str], float],
    factor_name: str,
    selected: list[str],
) -> tuple[float | None, str | None]:
    best_value: float | None = None
    best_peer: str | None = None
    for peer in selected:
        value = corr.get((factor_name, peer), corr.get((peer, factor_name)))
        if value is None:
            continue
        abs_value = abs(value)
        if best_value is None or abs_value > best_value:
            best_value = abs_value
            best_peer = peer
    return best_value, best_peer


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:.6f}"
    return str(value)
