from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from alpha_web3_research.backtest.hourly_rolling import (
    HourlyPortfolioRules,
    HourlyRollingPortfolioBacktester,
)
from alpha_web3_research.config import Settings
from alpha_web3_research.data.time import format_timestamp_ms
from alpha_web3_research.factors.hourly import HourlyMultiTimeframeFactorBuilder
from alpha_web3_research.labels.hourly import DEFAULT_HORIZONS, HourlyMultiHorizonLabelBuilder
from alpha_web3_research.strategies.hourly_multihorizon import build_composite_score

MODEL_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "min_child_samples": 100,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbosity": -1,
}


@dataclass(frozen=True)
class HourlyRollingResearchConfig:
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    train_fraction: float = 0.70
    valid_fraction: float = 0.15
    embargo_hours: int = 24
    round_trip_cost_rate: float = 0.0014
    disagreement_penalty: float = 0.25
    horizon_weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    threshold_grid: tuple[float, ...] = (
        0.00,
        0.005,
        0.01,
        0.015,
        0.02,
        0.025,
        0.03,
        0.04,
        0.05,
        0.06,
        0.075,
        0.09,
        0.10,
        0.12,
        0.15,
    )
    seed: int = 42


@dataclass(frozen=True)
class HourlyRollingResearchResult:
    run_id: str
    run_dir: Path
    predictions_path: Path
    hourly_backtest_path: Path
    positions_path: Path
    metrics_path: Path
    report_path: Path
    chosen_entry_threshold: float
    test_metrics: dict[str, Any]


class HourlyRollingResearchRunner:
    """Train four horizon bundles once, then run a strictly out-of-sample hourly portfolio simulation."""

    def __init__(
        self,
        settings: Settings,
        config: HourlyRollingResearchConfig | None = None,
        portfolio_rules: HourlyPortfolioRules | None = None,
    ) -> None:
        self.settings = settings
        self.config = config or HourlyRollingResearchConfig()
        self.portfolio_rules = portfolio_rules or HourlyPortfolioRules()
        self.models_root = settings.resolved_data_path() / "models" / "hourly_rolling"
        self.backtests_root = settings.resolved_data_path() / "backtests" / "hourly_rolling"
        self.reports_root = settings.resolved_reports_path() / "hourly_rolling"
        self._validate_config()

    def run(self, rebuild_dataset: bool = True) -> HourlyRollingResearchResult:
        features, feature_path = self._features(rebuild_dataset)
        labels, label_path = self._labels(rebuild_dataset)
        dataset, feature_names = self._dataset(features, labels)
        split_dataset, split_meta = split_hourly_dataset(
            dataset,
            train_fraction=self.config.train_fraction,
            valid_fraction=self.config.valid_fraction,
            embargo_hours=self.config.embargo_hours,
        )
        train = split_dataset.filter(pl.col("split") == "train")
        valid = split_dataset.filter(pl.col("split") == "valid")
        test = split_dataset.filter(pl.col("split") == "test")
        if train.is_empty() or valid.is_empty() or test.is_empty():
            raise ValueError("hourly time split produced an empty train, validation, or test partition")

        run_id = f"hourly_rolling_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        run_dir = self.models_root / run_id
        backtest_dir = self.backtests_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        backtest_dir.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)

        x_train = train.select(feature_names).to_pandas().astype(np.float32)
        x_valid = valid.select(feature_names).to_pandas().astype(np.float32)
        x_test = test.select(feature_names).to_pandas().astype(np.float32)
        prediction_frames: dict[str, pl.DataFrame] = {
            "valid": valid.select(["decision_timestamp", "symbol", "next_hour_return"]),
            "test": test.select(["decision_timestamp", "symbol", "next_hour_return"]),
        }
        model_metrics: dict[str, dict[str, Any]] = {}
        importance_rows: list[dict[str, Any]] = []

        for horizon in self.config.horizons:
            horizon_dir = run_dir / f"horizon={horizon}h"
            horizon_dir.mkdir(parents=True, exist_ok=True)
            return_target = f"forward_return_h{horizon}"
            volatility_target = f"forward_volatility_h{horizon}"
            return_model = self._fit_model(
                x_train,
                train.get_column(return_target).to_numpy(),
                x_valid,
                valid.get_column(return_target).to_numpy(),
                seed=self.config.seed + horizon,
            )
            volatility_model = self._fit_model(
                x_train,
                np.log1p(train.get_column(volatility_target).to_numpy()),
                x_valid,
                np.log1p(valid.get_column(volatility_target).to_numpy()),
                seed=self.config.seed + 100 + horizon,
            )
            return_model.booster_.save_model(str(horizon_dir / "return_model.txt"))
            volatility_model.booster_.save_model(str(horizon_dir / "volatility_model.txt"))

            valid_return_raw = return_model.predict(x_valid, num_iteration=return_model.best_iteration_)
            valid_volatility_raw = np.expm1(volatility_model.predict(x_valid, num_iteration=volatility_model.best_iteration_))
            calibration = fit_prediction_calibration(
                predicted_return=valid_return_raw,
                actual_return=valid.get_column(return_target).to_numpy(),
                predicted_volatility=valid_volatility_raw,
                actual_volatility=valid.get_column(volatility_target).to_numpy(),
            )
            (horizon_dir / "calibration.json").write_text(
                json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            horizon_metrics: dict[str, Any] = {"calibration": calibration}
            for split_name, x_split, split_frame in (
                ("valid", x_valid, valid),
                ("test", x_test, test),
            ):
                return_raw = return_model.predict(x_split, num_iteration=return_model.best_iteration_)
                volatility_raw = np.expm1(volatility_model.predict(x_split, num_iteration=volatility_model.best_iteration_))
                calibrated_return = calibration["return_intercept"] + calibration["return_slope"] * return_raw
                calibrated_volatility = np.maximum(
                    volatility_raw * calibration["volatility_scale"],
                    calibration["volatility_floor"],
                )
                actual_return = split_frame.get_column(return_target).to_numpy()
                actual_volatility = split_frame.get_column(volatility_target).to_numpy()
                horizon_metrics[split_name] = prediction_metrics(
                    calibrated_return,
                    actual_return,
                    calibrated_volatility,
                    actual_volatility,
                )
                prediction_frames[split_name] = prediction_frames[split_name].with_columns(
                    pl.Series(f"predicted_return_h{horizon}", calibrated_return),
                    pl.Series(f"predicted_volatility_h{horizon}", calibrated_volatility),
                    pl.Series(return_target, actual_return),
                    pl.Series(volatility_target, actual_volatility),
                )
            model_metrics[f"h{horizon}"] = horizon_metrics
            for feature_name, gain in zip(
                feature_names,
                return_model.booster_.feature_importance(importance_type="gain"),
                strict=True,
            ):
                importance_rows.append({"horizon": horizon, "feature_name": feature_name, "importance_gain": float(gain)})

        scored_predictions = pl.concat(
            [
                score_predictions(prediction_frames[split_name], self.config).with_columns(pl.lit(split_name).alias("split"))
                for split_name in ("valid", "test")
            ],
            how="vertical_relaxed",
        ).sort(["split", "decision_timestamp", "symbol"])
        predictions_path = run_dir / "predictions.parquet"
        scored_predictions.write_parquet(predictions_path)

        backtester = HourlyRollingPortfolioBacktester(self.portfolio_rules)
        valid_predictions = scored_predictions.filter(pl.col("split") == "valid")
        chosen_threshold, threshold_summary = backtester.tune_thresholds(
            valid_predictions,
            list(self.config.threshold_grid),
        )
        threshold_summary_path = backtest_dir / "validation_thresholds.parquet"
        threshold_summary.write_parquet(threshold_summary_path)
        test_result = backtester.run(
            scored_predictions.filter(pl.col("split") == "test"),
            entry_threshold=chosen_threshold,
        )
        hourly_backtest_path = backtest_dir / "hourly_portfolio.parquet"
        positions_path = backtest_dir / "positions.parquet"
        test_result.hourly.write_parquet(hourly_backtest_path)
        test_result.positions.write_parquet(positions_path)

        importance = (
            pl.DataFrame(importance_rows)
            .group_by("feature_name")
            .agg(pl.col("importance_gain").mean().alias("mean_importance_gain"))
            .sort("mean_importance_gain", descending=True)
        )
        importance.write_parquet(run_dir / "feature_importance.parquet")
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "strategy_key": "hourly_multihorizon_absolute_score_long_only@v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feature_source": str(feature_path),
            "label_source": str(label_path),
            "feature_count": len(feature_names),
            "features": feature_names,
            "config": asdict(self.config),
            "portfolio_rules": asdict(self.portfolio_rules),
            "split": split_meta,
            "model_metrics": model_metrics,
            "chosen_entry_threshold": chosen_threshold,
            "chosen_exit_threshold": chosen_threshold * self.portfolio_rules.exit_threshold_ratio,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metrics = {
            "run_id": run_id,
            "chosen_entry_threshold": chosen_threshold,
            "model_metrics": model_metrics,
            "test_portfolio": test_result.metrics,
        }
        metrics_path = backtest_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path = self.reports_root / f"{run_id}.md"
        write_report(
            report_path,
            manifest=manifest,
            threshold_summary=threshold_summary,
            test_metrics=test_result.metrics,
            importance=importance,
        )
        latest = self.models_root / "latest.json"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "report": str(report_path)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return HourlyRollingResearchResult(
            run_id=run_id,
            run_dir=run_dir,
            predictions_path=predictions_path,
            hourly_backtest_path=hourly_backtest_path,
            positions_path=positions_path,
            metrics_path=metrics_path,
            report_path=report_path,
            chosen_entry_threshold=chosen_threshold,
            test_metrics=test_result.metrics,
        )

    def _features(self, rebuild: bool) -> tuple[pl.DataFrame, Path]:
        builder = HourlyMultiTimeframeFactorBuilder(self.settings)
        if not rebuild and builder.output_path.exists():
            return pl.read_parquet(builder.output_path), builder.output_path
        features, result = builder.build(write=True)
        return features, result.path

    def _labels(self, rebuild: bool) -> tuple[pl.DataFrame, Path]:
        builder = HourlyMultiHorizonLabelBuilder(self.settings)
        if not rebuild and builder.output_path.exists():
            return pl.read_parquet(builder.output_path), builder.output_path
        labels, result = builder.build(horizons=self.config.horizons, write=True)
        return labels, result.path

    def _dataset(self, features: pl.DataFrame, labels: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
        feature_names = [column for column in features.columns if column.startswith(("1h__", "4h__"))]
        label_names = [
            name
            for horizon in self.config.horizons
            for name in (f"forward_return_h{horizon}", f"forward_volatility_h{horizon}")
        ]
        dataset = (
            features.drop("generated_at")
            .join(
                labels.select([*(["exchange", "market_type", "symbol", "decision_timestamp"]), "next_hour_return", *label_names]),
                on=["exchange", "market_type", "symbol", "decision_timestamp"],
                how="inner",
                validate="1:1",
            )
            .filter(pl.all_horizontal([pl.col(name).is_not_null() & pl.col(name).is_finite() for name in label_names]))
            .filter(pl.any_horizontal([pl.col(name).is_not_null() & pl.col(name).is_finite() for name in feature_names]))
            .with_columns(
                [
                    pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).cast(pl.Float32).alias(name)
                    for name in feature_names
                ]
            )
            .sort(["decision_timestamp", "symbol"])
        )
        usable_features = [
            name
            for name in feature_names
            if dataset.select(pl.col(name).is_not_null().sum()).item() > 0
            and dataset.select(pl.col(name).n_unique()).item() > 1
        ]
        if not usable_features:
            raise ValueError("no usable hourly features were generated")
        return dataset.select(["exchange", "market_type", "symbol", "decision_timestamp", *usable_features, "next_hour_return", *label_names]), usable_features

    @staticmethod
    def _fit_model(
        x_train,
        y_train: np.ndarray,
        x_valid,
        y_valid: np.ndarray,
        seed: int,
    ) -> lgb.LGBMRegressor:
        model = lgb.LGBMRegressor(**MODEL_PARAMS, random_state=seed)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)],
        )
        return model

    def _validate_config(self) -> None:
        if self.config.horizons != DEFAULT_HORIZONS:
            raise ValueError(f"hourly strategy horizons must be {DEFAULT_HORIZONS}")
        if len(self.config.horizon_weights) != len(self.config.horizons):
            raise ValueError("horizon_weights must match horizons")
        if any(weight < 0 for weight in self.config.horizon_weights) or sum(self.config.horizon_weights) <= 0:
            raise ValueError("horizon_weights must be non-negative and have positive sum")
        if not 0 < self.config.train_fraction < 1 or not 0 < self.config.valid_fraction < 1:
            raise ValueError("train and validation fractions must be in (0, 1)")
        if self.config.train_fraction + self.config.valid_fraction >= 1:
            raise ValueError("train and validation fractions must leave a test partition")


def split_hourly_dataset(
    dataset: pl.DataFrame,
    train_fraction: float,
    valid_fraction: float,
    embargo_hours: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    timestamps = dataset.get_column("decision_timestamp").unique().sort().to_list()
    if len(timestamps) < 100:
        raise ValueError("not enough hourly timestamps for train/validation/test split")
    train_boundary = int(len(timestamps) * train_fraction)
    valid_boundary = int(len(timestamps) * (train_fraction + valid_fraction))
    train_timestamps = timestamps[: max(0, train_boundary - embargo_hours)]
    valid_timestamps = timestamps[train_boundary : max(train_boundary, valid_boundary - embargo_hours)]
    test_timestamps = timestamps[valid_boundary:]
    split = (
        dataset.with_columns(
            pl.when(pl.col("decision_timestamp").is_in(train_timestamps))
            .then(pl.lit("train"))
            .when(pl.col("decision_timestamp").is_in(valid_timestamps))
            .then(pl.lit("valid"))
            .when(pl.col("decision_timestamp").is_in(test_timestamps))
            .then(pl.lit("test"))
            .otherwise(None)
            .alias("split")
        )
        .filter(pl.col("split").is_not_null())
    )
    metadata = {
        "embargo_hours": embargo_hours,
        "train": timestamp_range(train_timestamps),
        "valid": timestamp_range(valid_timestamps),
        "test": timestamp_range(test_timestamps),
    }
    return split, metadata


def fit_prediction_calibration(
    predicted_return: np.ndarray,
    actual_return: np.ndarray,
    predicted_volatility: np.ndarray,
    actual_volatility: np.ndarray,
) -> dict[str, float]:
    slope, intercept = np.polyfit(predicted_return, actual_return, deg=1)
    slope = float(np.clip(slope, 0.0, 3.0))
    intercept = float(np.clip(intercept, -0.05, 0.05))
    positive_prediction = np.maximum(predicted_volatility, 1e-8)
    volatility_scale = float(np.clip(np.median(actual_volatility / positive_prediction), 0.25, 4.0))
    volatility_floor = float(max(np.quantile(actual_volatility, 0.05), 1e-4))
    return {
        "return_slope": slope,
        "return_intercept": intercept,
        "volatility_scale": volatility_scale,
        "volatility_floor": volatility_floor,
    }


def score_predictions(frame: pl.DataFrame, config: HourlyRollingResearchConfig) -> pl.DataFrame:
    """Compatibility wrapper; score construction belongs to the strategy policy."""

    return build_composite_score(
        frame=frame,
        horizons=config.horizons,
        horizon_weights=config.horizon_weights,
        round_trip_cost_rate=config.round_trip_cost_rate,
        disagreement_penalty=config.disagreement_penalty,
    )


def prediction_metrics(
    predicted_return: np.ndarray,
    actual_return: np.ndarray,
    predicted_volatility: np.ndarray,
    actual_volatility: np.ndarray,
) -> dict[str, float | None]:
    correlation: float | None = None
    if np.std(predicted_return) > 1e-12:
        candidate = float(pearsonr(predicted_return, actual_return).statistic)
        correlation = candidate if np.isfinite(candidate) else None
    return {
        "return_rmse": float(np.sqrt(mean_squared_error(actual_return, predicted_return))),
        "return_mae": float(mean_absolute_error(actual_return, predicted_return)),
        "return_pearson": correlation,
        "volatility_rmse": float(np.sqrt(mean_squared_error(actual_volatility, predicted_volatility))),
        "volatility_mae": float(mean_absolute_error(actual_volatility, predicted_volatility)),
    }


def timestamp_range(timestamps: list[int]) -> dict[str, Any]:
    if not timestamps:
        return {"count": 0, "start": None, "end": None}
    return {
        "count": len(timestamps),
        "start": format_timestamp_ms(min(timestamps)),
        "end": format_timestamp_ms(max(timestamps)),
    }


def write_report(
    path: Path,
    manifest: dict[str, Any],
    threshold_summary: pl.DataFrame,
    test_metrics: dict[str, Any],
    importance: pl.DataFrame,
) -> None:
    lines = [
        "# 小时滚动多期限策略研究报告",
        "",
        "## 实验口径",
        "",
        "- 决策频率：每小时。",
        "- 数据：1h 原始 K 线 + 最近四根 1h K 线滚动重构 4h K 线。",
        "- 预测期限：1h、4h、12h、24h 累计收益和未来路径波动。",
        "- 模型：每个期限一组 LightGBM 收益/风险模型；单次 train/validation/test 时间切分，不使用 fold。",
        "- 选币：绝对综合 Score 超过阈值；不使用横截面排名，不限制持仓数量。",
        f"- 特征数：`{manifest['feature_count']}`。",
        f"- Score 分歧惩罚：`{manifest['config']['disagreement_penalty']}`。",
        f"- 预测内含往返成本：`{manifest['config']['round_trip_cost_rate']:.4%}`。",
        "",
        "## 时间切分",
        "",
        f"- Train：`{manifest['split']['train']['start']}` 至 `{manifest['split']['train']['end']}`。",
        f"- Validation：`{manifest['split']['valid']['start']}` 至 `{manifest['split']['valid']['end']}`。",
        f"- Test：`{manifest['split']['test']['start']}` 至 `{manifest['split']['test']['end']}`。",
        f"- 边界 purge：`{manifest['split']['embargo_hours']}` 小时。",
        "",
        "## 阈值选择",
        "",
        f"Validation 选择的进入阈值：`{manifest['chosen_entry_threshold']:.4f}`；退出阈值：`{manifest['chosen_exit_threshold']:.4f}`。",
        "",
        "| entry threshold | return | Sharpe | max drawdown | avg exposure | avg positions | turnover |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in threshold_summary.iter_rows(named=True):
        lines.append(
            f"| {row['entry_threshold']:.4f} | {row['total_return']:.2%} | {fmt(row['annualized_sharpe'])} | "
            f"{row['max_drawdown']:.2%} | {row['average_gross_exposure']:.2%} | "
            f"{row['average_active_positions']:.2f} | {row['total_turnover']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Test 样本外组合结果",
            "",
            f"- 总收益：`{test_metrics['total_return']:.2%}`。",
            f"- 年化 Sharpe：`{fmt(test_metrics['annualized_sharpe'])}`。",
            f"- 最大回撤：`{test_metrics['max_drawdown']:.2%}`。",
            f"- 平均总仓位：`{test_metrics['average_gross_exposure']:.2%}`。",
            f"- 平均持仓数：`{test_metrics['average_active_positions']:.2f}`；最大持仓数：`{test_metrics['max_active_positions']}`。",
            f"- 平均每小时换手：`{test_metrics['average_hourly_turnover']:.2%}`。",
            f"- 累计单边换手：`{test_metrics['total_turnover']:.2f}`。",
            f"- 最大收益贡献币种：`{test_metrics.get('top_contributor', '')}`，毛收益贡献 "
            f"`{float(test_metrics.get('top_contributor_gross_return', 0.0)):.2%}`。",
            "",
            "## 组合约束",
            "",
        ]
    )
    for key, value in manifest["portfolio_rules"].items():
        lines.append(f"- `{key}`：`{value}`。")
    lines.extend(["", "## 主要特征", "", "| feature | mean gain |", "|---|---:|"])
    for row in importance.head(30).iter_rows(named=True):
        lines.append(f"| {row['feature_name']} | {row['mean_importance_gain']:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"
