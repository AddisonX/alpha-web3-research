from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error

from alpha_web3_research.config import Settings
from alpha_web3_research.data.storage import ParquetOHLCVStore
from alpha_web3_research.data.time import format_timestamp_ms

ID_COLUMNS = ["exchange", "market_type", "symbol", "timeframe", "timestamp"]
LABEL_COLUMNS = [
    "close",
    "horizon_bars",
    "horizon_name",
    "forward_return",
    "forward_return_rank",
    "forward_return_rank_pct",
    "volatility_adjusted_return",
    "cross_section_count",
]
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbosity": -1,
}


@dataclass(frozen=True)
class ModelTrainingResult:
    run_id: str
    model_path: Path
    predictions_path: Path
    feature_importance_path: Path
    metrics_path: Path
    config_path: Path
    report_path: Path
    row_count: int
    feature_count: int
    split_counts: dict[str, int]


class LightGBMCrossSectionalTrainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_root = settings.resolved_data_path()
        self.models_root = self.data_root / "models"
        self.reports_root = settings.resolved_reports_path() / "models"
        self.factors_path = self.data_root / "factors" / "latest_factors.parquet"
        self.labels_path = self.data_root / "labels" / "latest_labels.parquet"
        self.decorrelated_path = self.data_root / "evaluation" / "decorrelated_factors.parquet"

    def train(
        self,
        timeframe: str = "1h",
        horizon_bars: int = 8,
        label_column: str = "forward_return_rank_pct",
        train_fraction: float = 0.70,
        valid_fraction: float = 0.15,
        min_cross_section_count: int = 20,
        top_quantile_fraction: float = 0.20,
        embargo_bars: int | None = None,
        seed: int = 42,
        params: dict[str, Any] | None = None,
    ) -> ModelTrainingResult:
        self._validate_inputs(train_fraction=train_fraction, valid_fraction=valid_fraction, top_quantile_fraction=top_quantile_fraction)
        feature_names = self._selected_feature_names(timeframe=timeframe, horizon_bars=horizon_bars)
        dataset = self._build_dataset(
            timeframe=timeframe,
            horizon_bars=horizon_bars,
            label_column=label_column,
            feature_names=feature_names,
            min_cross_section_count=min_cross_section_count,
        )
        split_dataset, split_meta = self._with_time_splits(
            dataset=dataset,
            train_fraction=train_fraction,
            valid_fraction=valid_fraction,
            embargo_bars=horizon_bars if embargo_bars is None else embargo_bars,
        )

        train = split_dataset.filter(pl.col("split") == "train")
        valid = split_dataset.filter(pl.col("split") == "valid")
        test = split_dataset.filter(pl.col("split") == "test")
        if train.is_empty() or valid.is_empty() or test.is_empty():
            raise ValueError("time split produced an empty train/valid/test partition")

        model_params = {**DEFAULT_PARAMS, **(params or {}), "random_state": seed}
        model = lgb.LGBMRegressor(**model_params)
        x_train = train.select(feature_names).to_pandas()
        y_train = train.get_column(label_column).to_numpy()
        x_valid = valid.select(feature_names).to_pandas()
        y_valid = valid.get_column(label_column).to_numpy()
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)],
        )

        predictions = self._predict(model=model, dataset=split_dataset, feature_names=feature_names, label_column=label_column)
        metrics = self._metrics(
            predictions=predictions,
            label_column=label_column,
            feature_names=feature_names,
            split_meta=split_meta,
            params=model_params,
            top_quantile_fraction=top_quantile_fraction,
        )
        feature_importance = self._feature_importance(model=model, feature_names=feature_names)

        run_id = f"lgbm_{timeframe}_h{horizon_bars}_{safe_slug(label_column)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        run_dir = self.models_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)

        model_path = run_dir / "model.txt"
        predictions_path = run_dir / "predictions.parquet"
        feature_importance_path = run_dir / "feature_importance.parquet"
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        report_path = self.reports_root / f"{run_id}.md"

        model.booster_.save_model(str(model_path))
        predictions.write_parquet(predictions_path)
        feature_importance.write_parquet(feature_importance_path)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        config = self._config_snapshot(
            run_id=run_id,
            timeframe=timeframe,
            horizon_bars=horizon_bars,
            label_column=label_column,
            feature_names=feature_names,
            train_fraction=train_fraction,
            valid_fraction=valid_fraction,
            min_cross_section_count=min_cross_section_count,
            top_quantile_fraction=top_quantile_fraction,
            embargo_bars=horizon_bars if embargo_bars is None else embargo_bars,
            seed=seed,
            params=model_params,
            split_meta=split_meta,
        )
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._write_report(report_path=report_path, config=config, metrics=metrics, feature_importance=feature_importance)

        self._write_latest_artifacts(
            model_path=model_path,
            predictions=predictions,
            feature_importance=feature_importance,
            metrics=metrics,
            config=config,
            report_path=report_path,
        )
        ParquetOHLCVStore(settings=self.settings).create_duckdb_database()
        split_counts = {row["split"]: row["rows"] for row in predictions.group_by("split").agg(pl.len().alias("rows")).iter_rows(named=True)}
        return ModelTrainingResult(
            run_id=run_id,
            model_path=model_path,
            predictions_path=predictions_path,
            feature_importance_path=feature_importance_path,
            metrics_path=metrics_path,
            config_path=config_path,
            report_path=report_path,
            row_count=predictions.height,
            feature_count=len(feature_names),
            split_counts=split_counts,
        )

    def _selected_feature_names(self, timeframe: str, horizon_bars: int) -> list[str]:
        if not self.decorrelated_path.exists():
            raise FileNotFoundError(f"decorrelated factors not found: {self.decorrelated_path}")
        selected = (
            pl.read_parquet(self.decorrelated_path)
            .filter((pl.col("timeframe") == timeframe) & (pl.col("horizon_bars") == horizon_bars))
            .sort("selection_score", descending=True)
        )
        if selected.is_empty():
            raise ValueError(f"no decorrelated features for timeframe={timeframe}, horizon_bars={horizon_bars}")
        return selected.get_column("factor_name").unique(maintain_order=True).to_list()

    def _build_dataset(
        self,
        timeframe: str,
        horizon_bars: int,
        label_column: str,
        feature_names: list[str],
        min_cross_section_count: int,
    ) -> pl.DataFrame:
        if not self.factors_path.exists():
            raise FileNotFoundError(f"factors parquet not found: {self.factors_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"labels parquet not found: {self.labels_path}")

        labels = pl.read_parquet(self.labels_path)
        if label_column not in labels.columns:
            raise ValueError(f"label column not found: {label_column}")
        labels_for_target = labels.filter((pl.col("timeframe") == timeframe) & (pl.col("horizon_bars") == horizon_bars)).select(
            ID_COLUMNS + [column for column in LABEL_COLUMNS if column in labels.columns]
        )
        factors = (
            pl.scan_parquet(self.factors_path)
            .filter((pl.col("timeframe") == timeframe) & pl.col("factor_name").is_in(feature_names))
            .select(ID_COLUMNS + ["factor_name", "factor_value"])
            .collect()
        )
        if factors.is_empty() or labels_for_target.is_empty():
            raise ValueError(f"empty model dataset for timeframe={timeframe}, horizon_bars={horizon_bars}")
        wide = factors.pivot(index=ID_COLUMNS, on="factor_name", values="factor_value")
        present_features = [name for name in feature_names if name in wide.columns]
        if not present_features:
            raise ValueError("none of the selected features were found in factor data")

        dataset = (
            wide.join(labels_for_target, on=ID_COLUMNS, how="inner")
            .filter(pl.col(label_column).is_not_null())
            .filter(pl.col("cross_section_count") >= min_cross_section_count)
            .filter(pl.any_horizontal([pl.col(name).is_not_null() for name in present_features]))
            .sort(["timestamp", "symbol"])
        )
        if dataset.is_empty():
            raise ValueError("model dataset is empty after joining features and labels")
        return dataset.select(ID_COLUMNS + present_features + [column for column in LABEL_COLUMNS if column in dataset.columns])

    def _with_time_splits(
        self,
        dataset: pl.DataFrame,
        train_fraction: float,
        valid_fraction: float,
        embargo_bars: int,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        timestamps = dataset.get_column("timestamp").unique().sort().to_list()
        if len(timestamps) < 10:
            raise ValueError("not enough timestamps for time-based train/valid/test split")
        train_cut = int(len(timestamps) * train_fraction)
        valid_cut = int(len(timestamps) * (train_fraction + valid_fraction))
        train_ts = timestamps[: max(0, train_cut - embargo_bars)]
        valid_ts = timestamps[min(len(timestamps), train_cut + embargo_bars) : max(0, valid_cut - embargo_bars)]
        test_ts = timestamps[min(len(timestamps), valid_cut + embargo_bars) :]

        split_dataset = dataset.with_columns(
            pl.when(pl.col("timestamp").is_in(train_ts))
            .then(pl.lit("train"))
            .when(pl.col("timestamp").is_in(valid_ts))
            .then(pl.lit("valid"))
            .when(pl.col("timestamp").is_in(test_ts))
            .then(pl.lit("test"))
            .otherwise(None)
            .alias("split")
        ).filter(pl.col("split").is_not_null())
        split_meta = {
            "timestamp_count": len(timestamps),
            "embargo_bars": embargo_bars,
            "train": time_range(train_ts),
            "valid": time_range(valid_ts),
            "test": time_range(test_ts),
        }
        return split_dataset, split_meta

    def _predict(self, model: lgb.LGBMRegressor, dataset: pl.DataFrame, feature_names: list[str], label_column: str) -> pl.DataFrame:
        scores = model.predict(dataset.select(feature_names).to_pandas(), num_iteration=model.best_iteration_)
        output_columns = unique_preserve_order(
            ID_COLUMNS
            + [
                "horizon_bars",
                "horizon_name",
                "split",
                label_column,
                "forward_return",
                "forward_return_rank_pct",
                "volatility_adjusted_return",
            ]
        )
        predictions = dataset.select(output_columns).with_columns(pl.Series("model_score", scores))
        return (
            predictions.with_columns(
                pl.len().over(["timeframe", "horizon_bars", "timestamp"]).alias("cross_section_count"),
                pl.col("model_score").rank(method="average").over(["timeframe", "horizon_bars", "timestamp"]).alias("model_score_rank"),
            )
            .with_columns(
                pl.when(pl.col("cross_section_count") > 1)
                .then((pl.col("model_score_rank") - 1.0) / (pl.col("cross_section_count") - 1.0))
                .otherwise(0.5)
                .alias("model_score_rank_pct")
            )
            .sort(["timestamp", "model_score"], descending=[False, True])
        )

    def _metrics(
        self,
        predictions: pl.DataFrame,
        label_column: str,
        feature_names: list[str],
        split_meta: dict[str, Any],
        params: dict[str, Any],
        top_quantile_fraction: float,
    ) -> dict[str, Any]:
        top_threshold = 1.0 - top_quantile_fraction
        result: dict[str, Any] = {
            "label_column": label_column,
            "feature_count": len(feature_names),
            "features": feature_names,
            "params": params,
            "top_quantile_fraction": top_quantile_fraction,
            "split_meta": split_meta,
            "splits": {},
        }
        for split in ["train", "valid", "test"]:
            frame = predictions.filter(pl.col("split") == split)
            if frame.is_empty():
                result["splits"][split] = {"rows": 0}
                continue
            y_true = frame.get_column(label_column).to_numpy()
            y_pred = frame.get_column("model_score").to_numpy()
            ts_metrics = (
                frame.group_by("timestamp")
                .agg(
                    pl.corr("model_score", label_column).alias("target_rank_ic"),
                    pl.corr("model_score", "forward_return").alias("forward_return_ic"),
                    pl.col("forward_return").filter(pl.col("model_score_rank_pct") >= top_threshold).mean().alias("top_quantile_forward_return"),
                    pl.col("forward_return").filter(pl.col("model_score_rank_pct") <= top_quantile_fraction).mean().alias("bottom_quantile_forward_return"),
                )
                .with_columns(
                    (pl.col("top_quantile_forward_return") - pl.col("bottom_quantile_forward_return")).alias("top_bottom_spread"),
                    (pl.col("bottom_quantile_forward_return") - pl.col("top_quantile_forward_return")).alias("inverse_top_bottom_spread"),
                )
            )
            result["splits"][split] = {
                "rows": frame.height,
                "timestamp_count": frame.get_column("timestamp").n_unique(),
                "symbol_count": frame.get_column("symbol").n_unique(),
                "start": format_timestamp_ms(frame.get_column("timestamp").min()),
                "end": format_timestamp_ms(frame.get_column("timestamp").max()),
                "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "mean_target_rank_ic": safe_mean(ts_metrics.get_column("target_rank_ic")),
                "mean_forward_return_ic": safe_mean(ts_metrics.get_column("forward_return_ic")),
                "mean_top_quantile_forward_return": safe_mean(ts_metrics.get_column("top_quantile_forward_return")),
                "mean_bottom_quantile_forward_return": safe_mean(ts_metrics.get_column("bottom_quantile_forward_return")),
                "mean_top_bottom_spread": safe_mean(ts_metrics.get_column("top_bottom_spread")),
                "mean_inverse_top_bottom_spread": safe_mean(ts_metrics.get_column("inverse_top_bottom_spread")),
            }
        return result

    def _feature_importance(self, model: lgb.LGBMRegressor, feature_names: list[str]) -> pl.DataFrame:
        booster = model.booster_
        return (
            pl.DataFrame(
                {
                    "feature_name": feature_names,
                    "importance_gain": booster.feature_importance(importance_type="gain"),
                    "importance_split": booster.feature_importance(importance_type="split"),
                }
            )
            .with_columns(
                (
                    pl.col("importance_gain")
                    / pl.when(pl.col("importance_gain").sum() > 0).then(pl.col("importance_gain").sum()).otherwise(1.0)
                ).alias("importance_gain_pct")
            )
            .sort("importance_gain", descending=True)
        )

    def _config_snapshot(
        self,
        run_id: str,
        timeframe: str,
        horizon_bars: int,
        label_column: str,
        feature_names: list[str],
        train_fraction: float,
        valid_fraction: float,
        min_cross_section_count: int,
        top_quantile_fraction: float,
        embargo_bars: int,
        seed: int,
        params: dict[str, Any],
        split_meta: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "timeframe": timeframe,
            "horizon_bars": horizon_bars,
            "label_column": label_column,
            "train_fraction": train_fraction,
            "valid_fraction": valid_fraction,
            "test_fraction": 1.0 - train_fraction - valid_fraction,
            "min_cross_section_count": min_cross_section_count,
            "top_quantile_fraction": top_quantile_fraction,
            "embargo_bars": embargo_bars,
            "seed": seed,
            "params": params,
            "features": feature_names,
            "paths": {
                "factors": str(self.factors_path),
                "labels": str(self.labels_path),
                "decorrelated_factors": str(self.decorrelated_path),
            },
            "split_meta": split_meta,
        }

    def _write_latest_artifacts(
        self,
        model_path: Path,
        predictions: pl.DataFrame,
        feature_importance: pl.DataFrame,
        metrics: dict[str, Any],
        config: dict[str, Any],
        report_path: Path,
    ) -> None:
        latest_model_path = self.models_root / "latest_model.txt"
        latest_model_path.write_text(model_path.read_text(encoding="utf-8"), encoding="utf-8")
        predictions.write_parquet(self.models_root / "latest_predictions.parquet")
        feature_importance.write_parquet(self.models_root / "latest_feature_importance.parquet")
        (self.models_root / "latest_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (self.models_root / "latest_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        latest_report = self.reports_root / "latest_lgbm_report.md"
        latest_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    def _write_report(self, report_path: Path, config: dict[str, Any], metrics: dict[str, Any], feature_importance: pl.DataFrame) -> None:
        lines = [
            "# LightGBM Cross-Sectional Model Report",
            "",
            f"- Run ID: `{config['run_id']}`",
            f"- Generated at: {config['generated_at']}",
            f"- Timeframe: `{config['timeframe']}`",
            f"- Horizon bars: `{config['horizon_bars']}`",
            f"- Label: `{config['label_column']}`",
            f"- Feature count: `{len(config['features'])}`",
            f"- Top/bottom quantile fraction: `{config['top_quantile_fraction']}`",
            f"- Seed: `{config['seed']}`",
            "",
            "## Split Metrics",
            "",
            "| split | rows | timestamps | RMSE | MAE | target RankIC | return IC | top-bottom spread | inverse spread |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for split in ["train", "valid", "test"]:
            item = metrics["splits"].get(split, {})
            lines.append(
                f"| {split} | {item.get('rows', 0)} | {item.get('timestamp_count', 0)} | "
                f"{fmt(item.get('rmse'))} | {fmt(item.get('mae'))} | {fmt(item.get('mean_target_rank_ic'))} | "
                f"{fmt(item.get('mean_forward_return_ic'))} | {fmt(item.get('mean_top_bottom_spread'))} | "
                f"{fmt(item.get('mean_inverse_top_bottom_spread'))} |"
            )
        lines.extend(["", "## Top Feature Importance", "", "| feature | gain | gain % | split |", "|---|---:|---:|---:|"])
        for row in feature_importance.head(30).iter_rows(named=True):
            lines.append(
                f"| {row['feature_name']} | {fmt(row['importance_gain'])} | {fmt(row['importance_gain_pct'])} | {row['importance_split']} |"
            )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _validate_inputs(train_fraction: float, valid_fraction: float, top_quantile_fraction: float) -> None:
        if train_fraction <= 0 or valid_fraction <= 0 or train_fraction + valid_fraction >= 1:
            raise ValueError("train_fraction and valid_fraction must be positive and leave a non-empty test fraction")
        if top_quantile_fraction <= 0 or top_quantile_fraction >= 0.5:
            raise ValueError("top_quantile_fraction must be greater than 0 and less than 0.5")


def time_range(timestamps: list[int]) -> dict[str, Any]:
    if not timestamps:
        return {"timestamp_count": 0, "start": None, "end": None}
    return {
        "timestamp_count": len(timestamps),
        "start": format_timestamp_ms(min(timestamps)),
        "end": format_timestamp_ms(max(timestamps)),
    }


def safe_mean(series: pl.Series) -> float | None:
    cleaned = series.drop_nulls().filter(series.drop_nulls().is_finite())
    if cleaned.is_empty():
        return None
    return float(cleaned.mean())


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def safe_slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")
