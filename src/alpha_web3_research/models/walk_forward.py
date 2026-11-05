from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from alpha_web3_research.config import Settings
from alpha_web3_research.data.storage import ParquetOHLCVStore
from alpha_web3_research.data.time import format_timestamp_ms
from alpha_web3_research.models.training import DEFAULT_PARAMS, LightGBMCrossSectionalTrainer, fmt

DAY_MS = 86_400_000


@dataclass(frozen=True)
class WalkForwardResult:
    experiment_id: str
    folds_path: Path
    predictions_path: Path
    feature_importance_path: Path
    summary_path: Path
    config_path: Path
    report_path: Path
    fold_count: int


@dataclass(frozen=True)
class WalkForwardGridResult:
    experiment_id: str
    summary_path: Path
    report_path: Path
    run_count: int


class RollingWalkForwardRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_root = settings.resolved_data_path()
        self.models_root = self.data_root / "models"
        self.walk_root = self.models_root / "walk_forward"
        self.reports_root = settings.resolved_reports_path() / "models"

    def run(
        self,
        timeframe: str,
        horizon_bars: int,
        label_column: str,
        train_days: int = 90,
        valid_days: int = 14,
        test_days: int = 14,
        step_days: int = 14,
        min_cross_section_count: int = 20,
        top_quantile_fraction: float = 0.10,
        embargo_bars: int | None = None,
        seed: int = 42,
        params: dict[str, Any] | None = None,
    ) -> WalkForwardResult:
        validate_walk_forward_inputs(
            train_days=train_days,
            valid_days=valid_days,
            test_days=test_days,
            step_days=step_days,
            top_quantile_fraction=top_quantile_fraction,
        )
        trainer = LightGBMCrossSectionalTrainer(settings=self.settings)
        feature_names = trainer._selected_feature_names(timeframe=timeframe, horizon_bars=horizon_bars)
        dataset = trainer._build_dataset(
            timeframe=timeframe,
            horizon_bars=horizon_bars,
            label_column=label_column,
            feature_names=feature_names,
            min_cross_section_count=min_cross_section_count,
        )
        timestamp_min = int(dataset.get_column("timestamp").min())
        timestamp_max = int(dataset.get_column("timestamp").max())
        bar_ms = timeframe_to_ms(timeframe)
        effective_embargo_bars = horizon_bars if embargo_bars is None else embargo_bars
        embargo_ms = effective_embargo_bars * bar_ms
        train_ms = train_days * DAY_MS
        valid_ms = valid_days * DAY_MS
        test_ms = test_days * DAY_MS
        step_ms = step_days * DAY_MS
        model_params = {**DEFAULT_PARAMS, **(params or {}), "random_state": seed}
        experiment_id = f"walk_forward_{timeframe}_h{horizon_bars}_{safe_slug(label_column)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        experiment_dir = self.walk_root / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)

        fold_rows: list[dict[str, object]] = []
        prediction_frames: list[pl.DataFrame] = []
        importance_frames: list[pl.DataFrame] = []
        fold_index = 0
        train_start = timestamp_min
        while True:
            train_end = train_start + train_ms
            valid_start = train_end + embargo_ms
            valid_end = valid_start + valid_ms
            test_start = valid_end + embargo_ms
            test_end = test_start + test_ms
            if test_end > timestamp_max + bar_ms:
                break

            split_dataset = assign_fold_split(
                dataset=dataset,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=test_end,
            )
            train = split_dataset.filter(pl.col("split") == "train")
            valid = split_dataset.filter(pl.col("split") == "valid")
            test = split_dataset.filter(pl.col("split") == "test")
            if train.is_empty() or valid.is_empty() or test.is_empty():
                train_start += step_ms
                continue

            fold_index += 1
            model = lgb.LGBMRegressor(**model_params)
            model.fit(
                train.select(feature_names).to_pandas(),
                train.get_column(label_column).to_numpy(),
                eval_set=[(valid.select(feature_names).to_pandas(), valid.get_column(label_column).to_numpy())],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)],
            )
            fold_predictions = trainer._predict(
                model=model,
                dataset=split_dataset,
                feature_names=feature_names,
                label_column=label_column,
            ).with_columns(pl.lit(fold_index).alias("fold"))
            split_meta = fold_split_meta(
                fold_index=fold_index,
                train=train,
                valid=valid,
                test=test,
                embargo_bars=effective_embargo_bars,
            )
            metrics = trainer._metrics(
                predictions=fold_predictions,
                label_column=label_column,
                feature_names=feature_names,
                split_meta=split_meta,
                params=model_params,
                top_quantile_fraction=top_quantile_fraction,
            )
            fold_importance = trainer._feature_importance(model=model, feature_names=feature_names).with_columns(
                pl.lit(fold_index).alias("fold")
            )
            model.booster_.save_model(str(experiment_dir / f"fold_{fold_index:02d}_model.txt"))
            prediction_frames.append(fold_predictions)
            importance_frames.append(fold_importance)
            fold_rows.append(fold_summary_row(experiment_id=experiment_id, fold=fold_index, metrics=metrics, split_meta=split_meta))
            train_start += step_ms

        if not fold_rows:
            raise ValueError("walk-forward produced no folds; reduce window sizes or check available data range")

        folds = pl.DataFrame(fold_rows)
        predictions = pl.concat(prediction_frames, how="diagonal_relaxed")
        feature_importance = pl.concat(importance_frames, how="diagonal_relaxed")
        summary = aggregate_summary(
            folds=folds,
            experiment_id=experiment_id,
            timeframe=timeframe,
            horizon_bars=horizon_bars,
            label_column=label_column,
            feature_count=len(feature_names),
            top_quantile_fraction=top_quantile_fraction,
        )
        config = {
            "experiment_id": experiment_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "timeframe": timeframe,
            "horizon_bars": horizon_bars,
            "label_column": label_column,
            "train_days": train_days,
            "valid_days": valid_days,
            "test_days": test_days,
            "step_days": step_days,
            "min_cross_section_count": min_cross_section_count,
            "top_quantile_fraction": top_quantile_fraction,
            "embargo_bars": effective_embargo_bars,
            "seed": seed,
            "params": model_params,
            "features": feature_names,
            "data_start": format_timestamp_ms(timestamp_min),
            "data_end": format_timestamp_ms(timestamp_max),
        }

        folds_path = experiment_dir / "folds.parquet"
        predictions_path = experiment_dir / "predictions.parquet"
        feature_importance_path = experiment_dir / "feature_importance.parquet"
        summary_path = experiment_dir / "summary.json"
        config_path = experiment_dir / "config.json"
        report_path = self.reports_root / f"{experiment_id}.md"
        folds.write_parquet(folds_path)
        predictions.write_parquet(predictions_path)
        feature_importance.write_parquet(feature_importance_path)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_report(report_path=report_path, folds=folds, summary=summary, config=config, feature_importance=feature_importance)

        latest_dir = self.walk_root
        folds.write_parquet(latest_dir / "latest_folds.parquet")
        predictions.write_parquet(latest_dir / "latest_predictions.parquet")
        feature_importance.write_parquet(latest_dir / "latest_feature_importance.parquet")
        (latest_dir / "latest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (latest_dir / "latest_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (self.reports_root / "latest_walk_forward_report.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
        ParquetOHLCVStore(settings=self.settings).create_duckdb_database()
        return WalkForwardResult(
            experiment_id=experiment_id,
            folds_path=folds_path,
            predictions_path=predictions_path,
            feature_importance_path=feature_importance_path,
            summary_path=summary_path,
            config_path=config_path,
            report_path=report_path,
            fold_count=folds.height,
        )


class WalkForwardGridRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_root = settings.resolved_data_path()
        self.models_root = self.data_root / "models"
        self.walk_root = self.models_root / "walk_forward"
        self.grid_root = self.walk_root / "grids"
        self.reports_root = settings.resolved_reports_path() / "models"
        self.decorrelated_path = self.data_root / "evaluation" / "decorrelated_factors.parquet"

    def run(
        self,
        labels: list[str],
        top_quantile_fractions: list[float],
        timeframes: list[str] | None = None,
        train_days: int = 90,
        valid_days: int = 14,
        test_days: int = 14,
        step_days: int = 14,
        min_cross_section_count: int = 20,
        seed: int = 42,
    ) -> WalkForwardGridResult:
        experiment_id = f"walk_forward_grid_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        combinations = self._available_combinations(timeframes=timeframes)
        runner = RollingWalkForwardRunner(settings=self.settings)
        rows: list[dict[str, object]] = []

        for combination in combinations.iter_rows(named=True):
            timeframe = str(combination["timeframe"])
            horizon_bars = int(combination["horizon_bars"])
            for label_column in labels:
                for top_quantile_fraction in top_quantile_fractions:
                    result = runner.run(
                        timeframe=timeframe,
                        horizon_bars=horizon_bars,
                        label_column=label_column,
                        train_days=train_days,
                        valid_days=valid_days,
                        test_days=test_days,
                        step_days=step_days,
                        min_cross_section_count=min_cross_section_count,
                        top_quantile_fraction=top_quantile_fraction,
                        seed=seed,
                    )
                    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "grid_experiment_id": experiment_id,
                            "walk_forward_experiment_id": result.experiment_id,
                            "timeframe": timeframe,
                            "horizon_bars": horizon_bars,
                            "label_column": label_column,
                            "top_quantile_fraction": top_quantile_fraction,
                            "fold_count": result.fold_count,
                            "feature_count": summary["feature_count"],
                            "test_spread_mean": summary["test_spread_mean"],
                            "test_spread_median": summary["test_spread_median"],
                            "test_spread_min": summary["test_spread_min"],
                            "test_spread_max": summary["test_spread_max"],
                            "test_spread_win_rate": summary["test_spread_win_rate"],
                            "test_spread_t_stat": summary["test_spread_t_stat"],
                            "valid_spread_mean": summary["valid_spread_mean"],
                            "valid_spread_win_rate": summary["valid_spread_win_rate"],
                            "test_rank_ic_mean": summary["test_rank_ic_mean"],
                            "test_rank_ic_median": summary["test_rank_ic_median"],
                            "folds_path": str(result.folds_path),
                            "summary_path": str(result.summary_path),
                            "report_path": str(result.report_path),
                        }
                    )

        summary_frame = pl.DataFrame(rows)
        grid_dir = self.grid_root / experiment_id
        grid_dir.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)
        summary_path = grid_dir / "summary.parquet"
        latest_summary_path = self.grid_root / "latest_summary.parquet"
        report_path = self.reports_root / f"{experiment_id}.md"
        latest_report_path = self.reports_root / "latest_walk_forward_grid_report.md"
        summary_frame.write_parquet(summary_path)
        summary_frame.write_parquet(latest_summary_path)
        write_grid_report(
            report_path=report_path,
            summary=summary_frame,
            experiment_id=experiment_id,
            labels=labels,
            top_quantile_fractions=top_quantile_fractions,
            train_days=train_days,
            valid_days=valid_days,
            test_days=test_days,
            step_days=step_days,
        )
        latest_report_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
        ParquetOHLCVStore(settings=self.settings).create_duckdb_database()
        return WalkForwardGridResult(
            experiment_id=experiment_id,
            summary_path=summary_path,
            report_path=report_path,
            run_count=summary_frame.height,
        )

    def _available_combinations(self, timeframes: list[str] | None) -> pl.DataFrame:
        if not self.decorrelated_path.exists():
            raise FileNotFoundError(f"decorrelated factors not found: {self.decorrelated_path}")
        frame = (
            pl.read_parquet(self.decorrelated_path)
            .group_by(["timeframe", "horizon_bars"])
            .agg(pl.len().alias("feature_count"))
            .sort(["timeframe", "horizon_bars"])
        )
        if timeframes:
            frame = frame.filter(pl.col("timeframe").is_in(timeframes))
        if frame.is_empty():
            raise ValueError("no timeframe/horizon combinations found for walk-forward grid")
        return frame


def assign_fold_split(
    dataset: pl.DataFrame,
    train_start: int,
    train_end: int,
    valid_start: int,
    valid_end: int,
    test_start: int,
    test_end: int,
) -> pl.DataFrame:
    return (
        dataset.with_columns(
            pl.when((pl.col("timestamp") >= train_start) & (pl.col("timestamp") < train_end))
            .then(pl.lit("train"))
            .when((pl.col("timestamp") >= valid_start) & (pl.col("timestamp") < valid_end))
            .then(pl.lit("valid"))
            .when((pl.col("timestamp") >= test_start) & (pl.col("timestamp") < test_end))
            .then(pl.lit("test"))
            .otherwise(None)
            .alias("split")
        )
        .filter(pl.col("split").is_not_null())
        .sort(["split", "timestamp", "symbol"])
    )


def fold_split_meta(fold_index: int, train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame, embargo_bars: int) -> dict[str, Any]:
    return {
        "fold": fold_index,
        "embargo_bars": embargo_bars,
        "train": frame_range(train),
        "valid": frame_range(valid),
        "test": frame_range(test),
    }


def frame_range(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "rows": frame.height,
        "timestamp_count": frame.get_column("timestamp").n_unique(),
        "symbol_count": frame.get_column("symbol").n_unique(),
        "start": format_timestamp_ms(int(frame.get_column("timestamp").min())),
        "end": format_timestamp_ms(int(frame.get_column("timestamp").max())),
    }


def fold_summary_row(experiment_id: str, fold: int, metrics: dict[str, Any], split_meta: dict[str, Any]) -> dict[str, object]:
    test_metrics = metrics["splits"]["test"]
    valid_metrics = metrics["splits"]["valid"]
    train_metrics = metrics["splits"]["train"]
    return {
        "experiment_id": experiment_id,
        "fold": fold,
        "train_start": split_meta["train"]["start"],
        "train_end": split_meta["train"]["end"],
        "valid_start": split_meta["valid"]["start"],
        "valid_end": split_meta["valid"]["end"],
        "test_start": split_meta["test"]["start"],
        "test_end": split_meta["test"]["end"],
        "train_rows": train_metrics["rows"],
        "valid_rows": valid_metrics["rows"],
        "test_rows": test_metrics["rows"],
        "valid_rank_ic": valid_metrics["mean_target_rank_ic"],
        "valid_return_ic": valid_metrics["mean_forward_return_ic"],
        "valid_spread": valid_metrics["mean_top_bottom_spread"],
        "valid_inverse_spread": valid_metrics["mean_inverse_top_bottom_spread"],
        "test_rank_ic": test_metrics["mean_target_rank_ic"],
        "test_return_ic": test_metrics["mean_forward_return_ic"],
        "test_spread": test_metrics["mean_top_bottom_spread"],
        "test_inverse_spread": test_metrics["mean_inverse_top_bottom_spread"],
        "test_top_return": test_metrics["mean_top_quantile_forward_return"],
        "test_bottom_return": test_metrics["mean_bottom_quantile_forward_return"],
    }


def aggregate_summary(
    folds: pl.DataFrame,
    experiment_id: str,
    timeframe: str,
    horizon_bars: int,
    label_column: str,
    feature_count: int,
    top_quantile_fraction: float,
) -> dict[str, Any]:
    test_spread = folds.get_column("test_spread").drop_nulls().to_numpy()
    valid_spread = folds.get_column("valid_spread").drop_nulls().to_numpy()
    test_rank_ic = folds.get_column("test_rank_ic").drop_nulls().to_numpy()
    return {
        "experiment_id": experiment_id,
        "timeframe": timeframe,
        "horizon_bars": horizon_bars,
        "label_column": label_column,
        "feature_count": feature_count,
        "top_quantile_fraction": top_quantile_fraction,
        "fold_count": folds.height,
        "test_spread_mean": mean_or_none(test_spread),
        "test_spread_median": median_or_none(test_spread),
        "test_spread_min": min_or_none(test_spread),
        "test_spread_max": max_or_none(test_spread),
        "test_spread_win_rate": win_rate(test_spread),
        "test_spread_t_stat": t_stat(test_spread),
        "valid_spread_mean": mean_or_none(valid_spread),
        "valid_spread_win_rate": win_rate(valid_spread),
        "test_rank_ic_mean": mean_or_none(test_rank_ic),
        "test_rank_ic_median": median_or_none(test_rank_ic),
    }


def write_report(report_path: Path, folds: pl.DataFrame, summary: dict[str, Any], config: dict[str, Any], feature_importance: pl.DataFrame) -> None:
    lines = [
        "# Rolling Walk-Forward Report",
        "",
        f"- Experiment ID: `{summary['experiment_id']}`",
        f"- Timeframe: `{summary['timeframe']}`",
        f"- Horizon bars: `{summary['horizon_bars']}`",
        f"- Label: `{summary['label_column']}`",
        f"- Windows: train `{config['train_days']}`d / valid `{config['valid_days']}`d / test `{config['test_days']}`d / step `{config['step_days']}`d",
        f"- Embargo bars: `{config['embargo_bars']}`",
        f"- Top/bottom fraction: `{summary['top_quantile_fraction']:.0%}`",
        f"- Fold count: `{summary['fold_count']}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| test spread mean | {fmt(summary['test_spread_mean'])} |",
        f"| test spread median | {fmt(summary['test_spread_median'])} |",
        f"| test spread win rate | {fmt(summary['test_spread_win_rate'])} |",
        f"| test spread t-stat | {fmt(summary['test_spread_t_stat'])} |",
        f"| test spread min | {fmt(summary['test_spread_min'])} |",
        f"| test spread max | {fmt(summary['test_spread_max'])} |",
        f"| test RankIC mean | {fmt(summary['test_rank_ic_mean'])} |",
        "",
        "## Folds",
        "",
        "| fold | train | valid | test | valid spread | test RankIC | test spread | inverse spread |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in folds.iter_rows(named=True):
        lines.append(
            f"| {row['fold']} | {row['train_start']} to {row['train_end']} | "
            f"{row['valid_start']} to {row['valid_end']} | {row['test_start']} to {row['test_end']} | "
            f"{fmt(row['valid_spread'])} | {fmt(row['test_rank_ic'])} | {fmt(row['test_spread'])} | {fmt(row['test_inverse_spread'])} |"
        )
    lines.extend(["", "## Mean Feature Importance", "", "| feature | mean gain pct | mean split |", "|---|---:|---:|"])
    importance_summary = (
        feature_importance.group_by("feature_name")
        .agg(
            pl.mean("importance_gain_pct").alias("mean_importance_gain_pct"),
            pl.mean("importance_split").alias("mean_importance_split"),
        )
        .sort("mean_importance_gain_pct", descending=True)
    )
    for row in importance_summary.head(30).iter_rows(named=True):
        lines.append(f"| {row['feature_name']} | {fmt(row['mean_importance_gain_pct'])} | {fmt(row['mean_importance_split'])} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_grid_report(
    report_path: Path,
    summary: pl.DataFrame,
    experiment_id: str,
    labels: list[str],
    top_quantile_fractions: list[float],
    train_days: int,
    valid_days: int,
    test_days: int,
    step_days: int,
) -> None:
    lines = [
        "# Rolling Walk-Forward Grid Report",
        "",
        f"- Experiment ID: `{experiment_id}`",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Labels: `{', '.join(labels)}`",
        f"- Top/bottom fractions: `{', '.join(f'{item:.0%}' for item in top_quantile_fractions)}`",
        f"- Windows: train `{train_days}`d / valid `{valid_days}`d / test `{test_days}`d / step `{step_days}`d",
        f"- Runs: `{summary.height}`",
        "",
        "## Best By Mean Test Spread",
        "",
        "| rank | timeframe | horizon | label | top % | folds | mean spread | median spread | win rate | t-stat | mean RankIC |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ranked = summary.sort(["test_spread_mean", "test_spread_win_rate", "test_rank_ic_mean"], descending=[True, True, True])
    for index, row in enumerate(ranked.head(25).iter_rows(named=True), start=1):
        lines.append(grid_row(index, row))

    lines.extend(
        [
            "",
            "## Best By Win Rate",
            "",
            "| rank | timeframe | horizon | label | top % | folds | mean spread | median spread | win rate | t-stat | mean RankIC |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ranked_win = summary.sort(["test_spread_win_rate", "test_spread_mean", "test_rank_ic_mean"], descending=[True, True, True])
    for index, row in enumerate(ranked_win.head(25).iter_rows(named=True), start=1):
        lines.append(grid_row(index, row))

    lines.extend(
        [
            "",
            "## Label Summary",
            "",
            "| label | avg mean spread | best mean spread | avg win rate |",
            "|---|---:|---:|---:|",
        ]
    )
    label_summary = (
        summary.group_by("label_column")
        .agg(
            pl.mean("test_spread_mean").alias("avg_test_spread_mean"),
            pl.max("test_spread_mean").alias("best_test_spread_mean"),
            pl.mean("test_spread_win_rate").alias("avg_win_rate"),
        )
        .sort("avg_test_spread_mean", descending=True)
    )
    for row in label_summary.iter_rows(named=True):
        lines.append(
            f"| {row['label_column']} | {fmt(row['avg_test_spread_mean'])} | "
            f"{fmt(row['best_test_spread_mean'])} | {fmt(row['avg_win_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## All Runs",
            "",
            "| timeframe | horizon | label | top % | folds | mean spread | median spread | win rate | t-stat | mean RankIC |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.sort(["timeframe", "horizon_bars", "label_column", "top_quantile_fraction"]).iter_rows(named=True):
        lines.append(grid_row(None, row))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def grid_row(index: int | None, row: dict[str, object]) -> str:
    first = str(index) if index is not None else str(row["timeframe"])
    if index is not None:
        return (
            f"| {first} | {row['timeframe']} | {row['horizon_bars']} | {row['label_column']} | "
            f"{float(row['top_quantile_fraction']):.0%} | {row['fold_count']} | {fmt(row['test_spread_mean'])} | "
            f"{fmt(row['test_spread_median'])} | {fmt(row['test_spread_win_rate'])} | "
            f"{fmt(row['test_spread_t_stat'])} | {fmt(row['test_rank_ic_mean'])} |"
        )
    return (
        f"| {first} | {row['horizon_bars']} | {row['label_column']} | "
        f"{float(row['top_quantile_fraction']):.0%} | {row['fold_count']} | {fmt(row['test_spread_mean'])} | "
        f"{fmt(row['test_spread_median'])} | {fmt(row['test_spread_win_rate'])} | "
        f"{fmt(row['test_spread_t_stat'])} | {fmt(row['test_rank_ic_mean'])} |"
    )


def timeframe_to_ms(timeframe: str) -> int:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 3_600_000
    if unit == "d":
        return value * DAY_MS
    raise ValueError(f"unsupported timeframe: {timeframe}")


def validate_walk_forward_inputs(train_days: int, valid_days: int, test_days: int, step_days: int, top_quantile_fraction: float) -> None:
    if min(train_days, valid_days, test_days, step_days) <= 0:
        raise ValueError("walk-forward day windows must be positive")
    if top_quantile_fraction <= 0 or top_quantile_fraction >= 0.5:
        raise ValueError("top_quantile_fraction must be greater than 0 and less than 0.5")


def mean_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def median_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.median(values))


def min_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.min(values))


def max_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.max(values))


def win_rate(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values > 0))


def t_stat(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    std = float(np.std(values, ddof=1))
    if std == 0:
        return None
    return float(np.mean(values) / (std / np.sqrt(values.size)))


def safe_slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")
