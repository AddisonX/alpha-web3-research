from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from alpha_web3_research.config import Settings
from alpha_web3_research.data.storage import ParquetOHLCVStore

JOIN_COLUMNS = ["exchange", "market_type", "symbol", "timeframe", "timestamp"]
EVAL_COLUMNS = ["factor_name", "factor_family", "timeframe"]
EVAL_HORIZON_COLUMNS = EVAL_COLUMNS + ["horizon_bars"]


@dataclass(frozen=True)
class SingleFactorEvaluationResult:
    summary_path: Path
    timeseries_path: Path
    group_returns_path: Path
    report_path: Path
    evaluated_factors: int
    evaluated_rows: int


class SingleFactorEvaluator:
    def __init__(self, settings: Settings, store: ParquetOHLCVStore | None = None) -> None:
        self.settings = settings
        self.store = store or ParquetOHLCVStore(settings=settings)
        self.data_root = settings.resolved_data_path() / "evaluation"
        self.report_root = settings.resolved_reports_path()

    def evaluate(
        self,
        label_column: str = "forward_return",
        quantiles: int = 5,
        min_pairs: int = 2,
    ) -> SingleFactorEvaluationResult:
        factors_path = self.settings.resolved_data_path() / "factors" / "latest_factors.parquet"
        labels_path = self.settings.resolved_data_path() / "labels" / "latest_labels.parquet"
        if not factors_path.exists():
            raise FileNotFoundError(f"factors parquet not found: {factors_path}")
        labels = self._read_required(labels_path, "labels")
        if label_column not in labels.columns:
            raise ValueError(f"label column not found: {label_column}")

        factor_meta = (
            pl.scan_parquet(factors_path)
            .select(["factor_name", "factor_family", "timeframe"])
            .unique()
            .sort(["timeframe", "factor_name"])
            .collect()
        )
        label_meta = labels.select(["timeframe", "horizon_bars"]).unique().sort(["timeframe", "horizon_bars"])
        summary_frames: list[pl.DataFrame] = []
        timeseries_frames: list[pl.DataFrame] = []
        group_return_frames: list[pl.DataFrame] = []
        evaluated_rows = 0

        for row in factor_meta.iter_rows(named=True):
            factor_name = row["factor_name"]
            timeframe = row["timeframe"]
            factor_frame = (
                pl.scan_parquet(factors_path)
                .filter((pl.col("factor_name") == factor_name) & (pl.col("timeframe") == timeframe))
                .select(["generated_at", *JOIN_COLUMNS, "factor_family", "factor_name", "factor_value"])
                .collect()
            )
            horizons_for_timeframe = label_meta.filter(pl.col("timeframe") == timeframe).get_column("horizon_bars").to_list()
            if factor_frame.is_empty() or not horizons_for_timeframe:
                continue
            for horizon_bars in horizons_for_timeframe:
                labels_for_batch = labels.filter((pl.col("timeframe") == timeframe) & (pl.col("horizon_bars") == horizon_bars))
                if labels_for_batch.is_empty():
                    continue
                joined = self._joined_frame(factors=factor_frame, labels=labels_for_batch, label_column=label_column)
                if joined.is_empty():
                    continue
                evaluated_rows += joined.height
                timeseries = self._build_timeseries(joined=joined, label_column=label_column, quantiles=quantiles, min_pairs=min_pairs)
                summary = self._build_summary(joined=joined, timeseries=timeseries, label_column=label_column)
                group_returns = self._build_group_returns(timeseries=timeseries, quantiles=quantiles)
                if not summary.is_empty():
                    summary_frames.append(summary)
                if not timeseries.is_empty():
                    timeseries_frames.append(timeseries)
                if not group_returns.is_empty():
                    group_return_frames.append(group_returns)

        summary = pl.concat(summary_frames, how="diagonal_relaxed") if summary_frames else self._empty_summary()
        timeseries = pl.concat(timeseries_frames, how="diagonal_relaxed") if timeseries_frames else self._empty_timeseries()
        group_returns = pl.concat(group_return_frames, how="diagonal_relaxed") if group_return_frames else self._empty_group_returns()
        if not summary.is_empty():
            summary = summary.sort(["abs_mean_rank_ic", "abs_mean_ic", "sample_count"], descending=[True, True, True], nulls_last=True)

        self.data_root.mkdir(parents=True, exist_ok=True)
        self.report_root.mkdir(parents=True, exist_ok=True)
        summary_path = self.data_root / "single_factor_summary.parquet"
        timeseries_path = self.data_root / "single_factor_timeseries.parquet"
        group_returns_path = self.data_root / "single_factor_group_returns.parquet"
        report_path = self.report_root / "single_factor_report.md"
        summary.write_parquet(summary_path)
        timeseries.write_parquet(timeseries_path)
        group_returns.write_parquet(group_returns_path)
        self._write_report(summary=summary, group_returns=group_returns, report_path=report_path, label_column=label_column, quantiles=quantiles)

        ParquetOHLCVStore(settings=self.settings).create_duckdb_database()
        return SingleFactorEvaluationResult(
            summary_path=summary_path,
            timeseries_path=timeseries_path,
            group_returns_path=group_returns_path,
            report_path=report_path,
            evaluated_factors=summary.select(pl.col("factor_name").n_unique()).item() if not summary.is_empty() else 0,
            evaluated_rows=evaluated_rows,
        )

    def _joined_frame(self, factors: pl.DataFrame, labels: pl.DataFrame, label_column: str) -> pl.DataFrame:
        if label_column not in labels.columns:
            raise ValueError(f"label column not found: {label_column}")
        label_columns = JOIN_COLUMNS + ["horizon_bars", label_column]
        if label_column != "forward_return":
            label_columns.append("forward_return")
        return (
            factors.join(
                labels.select(label_columns),
                on=JOIN_COLUMNS,
                how="inner",
            )
            .filter(pl.col("factor_value").is_not_null() & pl.col(label_column).is_not_null())
            .with_columns(pl.col("factor_value").cast(pl.Float64), pl.col(label_column).cast(pl.Float64))
        )

    def _build_timeseries(
        self,
        joined: pl.DataFrame,
        label_column: str,
        quantiles: int,
        min_pairs: int,
    ) -> pl.DataFrame:
        if joined.is_empty():
            return self._empty_timeseries()
        grouped = (
            joined.with_columns(
                pl.col("factor_value").rank(method="average").over(EVAL_HORIZON_COLUMNS + ["timestamp"]).alias("factor_rank"),
                pl.col(label_column).rank(method="average").over(EVAL_HORIZON_COLUMNS + ["timestamp"]).alias("label_rank"),
                pl.len().over(EVAL_HORIZON_COLUMNS + ["timestamp"]).alias("cross_section_count"),
                quantile_expr("factor_value", quantiles).over(EVAL_HORIZON_COLUMNS + ["timestamp"]).alias("factor_quantile"),
            )
            .with_columns((pl.col("factor_value") * pl.col(label_column)).alias("factor_label_product"))
            .group_by(EVAL_HORIZON_COLUMNS + ["timestamp"])
            .agg(
                pl.first("cross_section_count").alias("cross_section_count"),
                pl.corr("factor_value", label_column).alias("ic"),
                pl.corr("factor_rank", "label_rank").alias("rank_ic"),
                pl.mean(label_column).alias("mean_forward_return"),
                pl.mean("factor_value").alias("mean_factor_value"),
                pl.col("factor_quantile").alias("factor_quantiles"),
                pl.col(label_column).alias("forward_returns"),
            )
            .with_columns(nan_to_null("ic"), nan_to_null("rank_ic"))
            .filter(pl.col("cross_section_count") >= min_pairs)
            .sort(EVAL_HORIZON_COLUMNS + ["timestamp"])
        )
        return grouped

    def _build_summary(self, joined: pl.DataFrame, timeseries: pl.DataFrame, label_column: str) -> pl.DataFrame:
        if joined.is_empty() or timeseries.is_empty():
            return self._empty_summary()

        coverage = (
            joined.group_by(EVAL_HORIZON_COLUMNS)
            .agg(
                pl.len().alias("sample_count"),
                pl.n_unique("timestamp").alias("timestamp_count"),
                pl.n_unique("symbol").alias("symbol_count"),
                (pl.col("factor_value").is_not_null().sum() / pl.len()).alias("coverage"),
                pl.mean("factor_value").alias("mean_factor_value"),
                pl.std("factor_value").alias("std_factor_value"),
            )
        )
        ts_summary = (
            timeseries.group_by(EVAL_HORIZON_COLUMNS)
            .agg(
                pl.mean("ic").alias("mean_ic"),
                pl.std("ic").alias("std_ic"),
                safe_ir("ic").alias("icir"),
                pl.mean("rank_ic").alias("mean_rank_ic"),
                pl.std("rank_ic").alias("std_rank_ic"),
                safe_ir("rank_ic").alias("rank_icir"),
                (pl.col("ic") > 0).mean().alias("positive_ic_rate"),
                (pl.col("rank_ic") > 0).mean().alias("positive_rank_ic_rate"),
                pl.mean("mean_forward_return").alias("mean_forward_return"),
            )
        )
        return (
            coverage.join(ts_summary, on=EVAL_HORIZON_COLUMNS, how="left")
            .with_columns(
                nan_to_null("mean_ic"),
                nan_to_null("std_ic"),
                nan_to_null("icir"),
                nan_to_null("mean_rank_ic"),
                nan_to_null("std_rank_ic"),
                nan_to_null("rank_icir"),
                pl.col("mean_ic").abs().alias("abs_mean_ic"),
                pl.col("mean_rank_ic").abs().alias("abs_mean_rank_ic"),
                pl.lit(datetime.now(timezone.utc).isoformat()).alias("generated_at"),
                pl.lit(label_column).alias("label_column"),
            )
            .sort(["abs_mean_rank_ic", "abs_mean_ic", "sample_count"], descending=[True, True, True], nulls_last=True)
        )

    def _build_group_returns(self, timeseries: pl.DataFrame, quantiles: int) -> pl.DataFrame:
        if timeseries.is_empty():
            return self._empty_group_returns()

        exploded = timeseries.select(EVAL_HORIZON_COLUMNS + ["timestamp", "factor_quantiles", "forward_returns"]).explode(
            ["factor_quantiles", "forward_returns"]
        )
        grouped = (
            exploded.group_by(EVAL_HORIZON_COLUMNS + ["factor_quantiles"])
            .agg(
                pl.mean("forward_returns").alias("mean_forward_return"),
                pl.len().alias("sample_count"),
            )
            .rename({"factor_quantiles": "factor_quantile"})
            .sort(EVAL_HORIZON_COLUMNS + ["factor_quantile"])
        )
        spread = (
            grouped.filter(pl.col("factor_quantile").is_in([1, quantiles]))
            .pivot(index=EVAL_HORIZON_COLUMNS, columns="factor_quantile", values="mean_forward_return")
        )
        low_col = "1"
        high_col = str(quantiles)
        if low_col in spread.columns and high_col in spread.columns:
            spread = spread.with_columns((pl.col(high_col) - pl.col(low_col)).alias("top_bottom_spread")).select(
                EVAL_HORIZON_COLUMNS + ["top_bottom_spread"]
            )
            grouped = grouped.join(spread, on=EVAL_HORIZON_COLUMNS, how="left")
        else:
            grouped = grouped.with_columns(pl.lit(None, dtype=pl.Float64).alias("top_bottom_spread"))
        return grouped

    def _write_report(self, summary: pl.DataFrame, group_returns: pl.DataFrame, report_path: Path, label_column: str, quantiles: int) -> None:
        lines = [
            "# Single Factor Evaluation Report",
            "",
            f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
            f"- Label column: `{label_column}`",
            f"- Quantiles: `{quantiles}`",
            "",
        ]
        if summary.is_empty():
            lines.append("No factor evaluation rows were produced.")
            report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return

        lines.extend(
            [
                "## Top Factors By |RankIC|",
                "",
                "| factor | family | timeframe | horizon | samples | mean_ic | mean_rank_ic | rank_icir | positive_rank_ic_rate |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary.head(20).iter_rows(named=True):
            lines.append(
                f"| {row['factor_name']} | {row['factor_family']} | {row['timeframe']} | {row['horizon_bars']} | "
                f"{row['sample_count']} | {fmt(row['mean_ic'])} | {fmt(row['mean_rank_ic'])} | "
                f"{fmt(row['rank_icir'])} | {fmt(row['positive_rank_ic_rate'])} |"
            )

        lines.extend(["", "## Top-Bottom Spread", ""])
        if group_returns.is_empty():
            lines.append("No group return rows were produced.")
        else:
            spread = (
                group_returns.select(EVAL_HORIZON_COLUMNS + ["top_bottom_spread"])
                .unique()
                .sort("top_bottom_spread", descending=True, nulls_last=True)
                .head(20)
            )
            lines.extend(
                [
                    "| factor | family | timeframe | horizon | top_bottom_spread |",
                    "|---|---|---|---:|---:|",
                ]
            )
            for row in spread.iter_rows(named=True):
                lines.append(
                    f"| {row['factor_name']} | {row['factor_family']} | {row['timeframe']} | "
                    f"{row['horizon_bars']} | {fmt(row['top_bottom_spread'])} |"
                )

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _read_required(path: Path, name: str) -> pl.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"{name} parquet not found: {path}")
        return pl.read_parquet(path)

    @staticmethod
    def _empty_timeseries() -> pl.DataFrame:
        return pl.DataFrame()

    @staticmethod
    def _empty_summary() -> pl.DataFrame:
        return pl.DataFrame()

    @staticmethod
    def _empty_group_returns() -> pl.DataFrame:
        return pl.DataFrame()


def quantile_expr(column: str, quantiles: int) -> pl.Expr:
    rank_pct = (pl.col(column).rank(method="average") - 1.0) / (pl.len() - 1.0)
    return pl.when(pl.len() > 1).then((rank_pct * quantiles).floor().clip(0, quantiles - 1).cast(pl.Int64) + 1).otherwise(1)


def safe_ir(column: str) -> pl.Expr:
    return pl.when(pl.std(column) > 0).then(pl.mean(column) / pl.std(column)).otherwise(None)


def nan_to_null(column: str) -> pl.Expr:
    return pl.when(pl.col(column).is_nan()).then(None).otherwise(pl.col(column)).alias(column)


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:.6f}"
    return str(value)
