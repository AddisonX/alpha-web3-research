"""Fixed 2025 training and 2026 inference with exactly 300 XGBoost trees."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from zoneinfo import ZoneInfo

import numpy as np
import plotly.graph_objects as go
import polars as pl
import xgboost as xgb
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alpha_web3_research.backtest.daily_equal_weight import run_daily_equal_weight
from alpha_web3_research.config import Settings
from alpha_web3_research.data.time import format_timestamp_ms, parse_timestamp_ms
from alpha_web3_research.models.pit_rank import excess_prediction_metrics, purged_training_data
from alpha_web3_research.models.pit_rank_dataset import build_pit_rank_dataset
from alpha_web3_research.research.pit_rank import read_bars
from alpha_web3_research.strategies.pit_rank_xgboost import (
    PITRankXGBoostLongOnlyStrategy, STRATEGY_ID, STRATEGY_KEY,
)


class PITXGBoostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    strategy_key: Literal[STRATEGY_KEY] = STRATEGY_KEY
    source_dataset_namespace: str = "pit_rank64_202501_202610"
    features_file: str = "strategies/features/pit_rank64.v1.json"
    train_start: str = "2025-01-01T00:00:00Z"
    test_start: str = "2026-01-01T00:00:00Z"
    requested_test_end: str = "2027-01-01T00:00:00Z"
    n_estimators: Literal[300] = 300
    learning_rate: float = Field(default=0.03, gt=0, le=1)
    max_depth: int = Field(default=4, ge=1)
    min_child_weight: float = Field(default=100.0, ge=0)
    reg_alpha: float = Field(default=0.1, ge=0)
    reg_lambda: float = Field(default=10.0, ge=0)
    seed: int = 42
    min_cross_section: int = Field(default=20, ge=2)
    decision_hour_utc: int = Field(default=4, ge=0, le=23)
    fraction: float = Field(default=0.10, gt=0, le=1)
    stop_loss: None = None
    one_way_fee_rate: float = Field(default=0.0005, ge=0)
    one_way_slippage_rate: float = Field(default=0.0002, ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> PITXGBoostConfig:
        if not parse_timestamp_ms(self.train_start) < parse_timestamp_ms(self.test_start) < parse_timestamp_ms(self.requested_test_end):
            raise ValueError("require train_start < test_start < requested_test_end")
        if Path(self.source_dataset_namespace).name != self.source_dataset_namespace:
            raise ValueError("source dataset must be a namespace name")
        if self.decision_hour_utc % 4:
            raise ValueError("decision time must be a native 4h close boundary")
        return self

    @classmethod
    def load(cls, path: Path) -> PITXGBoostConfig:
        return cls.model_validate_json(path.read_text())


def fixed_year_split(
    dataset: pl.DataFrame, config: PITXGBoostConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    start, cutoff, end = [parse_timestamp_ms(value) for value in (
        config.train_start, config.test_start, config.requested_test_end,
    )]
    train = purged_training_data(dataset, start, cutoff)
    test = dataset.filter((pl.col("decision_timestamp") >= cutoff) & (pl.col("decision_timestamp") < end))
    if train.is_empty() or test.is_empty():
        raise ValueError("2025 training and 2026 test features must both be nonempty")
    purged = dataset.filter(
        (pl.col("decision_timestamp") >= start) & (pl.col("decision_timestamp") < cutoff)
        & (pl.col("label_end_timestamp") >= cutoff)
    ).height
    return train, test, {
        "training_protocol": "fixed_year_train_test", "fit_count": 1,
        "validation_set_count": 0, "early_stopping": False, "parameter_search": False,
        "train_start": config.train_start, "train_end_exclusive": config.test_start,
        "test_start": config.test_start, "requested_test_end_exclusive": config.requested_test_end,
        "train_rows": train.height, "test_feature_rows": test.height,
        "train_decision_min": int(train["decision_timestamp"].min()),
        "train_decision_max": int(train["decision_timestamp"].max()),
        "train_label_end_max": int(train["label_end_timestamp"].max()),
        "test_decision_min": int(test["decision_timestamp"].min()),
        "test_decision_max": int(test["decision_timestamp"].max()),
        "boundary_label_purged_rows": purged,
    }


def monthly_performance(daily: pl.DataFrame) -> pl.DataFrame:
    return daily.with_columns(
        pl.from_epoch("decision_timestamp", time_unit="ms").dt.strftime("%Y-%m").alias("month"),
    ).group_by("month").agg(
        pl.len().alias("days"),
        ((1 + pl.col("net_return")).product() - 1).alias("strategy_return"),
        ((1 + pl.col("universe_return")).product() - 1).alias("universe_return"),
        pl.col("excess_return").sum().alias("excess_simple_return"),
    ).sort("month")


def beijing_time(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, timezone.utc).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")


class PITXGBoostResearchRunner:
    def __init__(self, settings: Settings, config: PITXGBoostConfig) -> None:
        self.settings = settings
        self.config = config

    def run(self, progress: Callable[[str], None] = print) -> Path:
        config = self.config
        source = self.settings.resolved_data_path() / "research" / config.source_dataset_namespace
        source_manifest_path = source / "data_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text())
        if source_manifest["inception_as_of_ms"] > parse_timestamp_ms(config.train_start):
            raise ValueError("the source inception universe must precede the training range")
        if source_manifest.get("future_survival_used_for_selection") is not False:
            raise ValueError("source universe must not filter historical members by future survival")
        feature_path = self.settings.project_root / config.features_file
        feature_manifest = json.loads(feature_path.read_text())
        names = feature_manifest["features"]
        if len(names) != 64 or len(set(names)) != 64:
            raise ValueError("expected the fixed 64 distinct factor names")
        universe = pl.read_parquet(source / "universe.parquet").filter(pl.col("included"))
        bars = read_bars(source / "bars").filter(pl.col("symbol").is_in(universe["symbol"].to_list()))
        run_id = f"{STRATEGY_ID}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        run = self.settings.resolved_data_path() / "models" / STRATEGY_ID / run_id
        run.mkdir(parents=True, exist_ok=False)
        progress(f"Building 64 factor ranks from the fixed {universe.height}-member inception universe")
        dataset, labels = build_pit_rank_dataset(bars, names, self.settings,
                                                config.min_cross_section, config.decision_hour_utc)
        train, test, split = fixed_year_split(dataset, config)
        dataset.write_parquet(run / "dataset.parquet")
        labels.write_parquet(run / "labels.parquet")
        (run / "split.json").write_text(json.dumps(split, indent=2) + "\n")
        params = {
            "objective": "reg:squarederror", "booster": "gbtree", "tree_method": "hist",
            "device": "cpu", "n_estimators": config.n_estimators, "num_parallel_tree": 1,
            "max_depth": config.max_depth, "learning_rate": config.learning_rate,
            "min_child_weight": config.min_child_weight, "reg_alpha": config.reg_alpha,
            "reg_lambda": config.reg_lambda, "subsample": 0.8, "colsample_bytree": 0.8,
            "random_state": config.seed, "n_jobs": 4, "verbosity": 0,
        }
        progress(f"One XGBoost fit: {train.height:,} training rows from 2025; "
                 f"{test.height:,} test feature rows from 2026; exactly 300 trees, no early stopping")
        model = xgb.XGBRegressor(**params)
        model.fit(train.select(names).to_pandas(), train["excess_forward_return"].to_numpy())
        if model.get_booster().num_boosted_rounds() != 300 or len(model.get_booster().get_dump()) != 300:
            raise ValueError("XGBoost did not produce exactly 300 trees")
        model_path = run / "model.ubj"
        model.save_model(model_path)
        frozen = xgb.XGBRegressor()
        frozen.load_model(model_path)
        predictions = test.with_columns(
            pl.Series("prediction", frozen.predict(test.select(names).to_pandas())),
            pl.lit("test").alias("split"), pl.lit(config.strategy_key).alias("strategy_key"),
        )
        predictions.write_parquet(run / "predictions.parquet")
        training_predictions = train.with_columns(
            pl.Series("prediction", frozen.predict(train.select(names).to_pandas())),
        )
        gain = frozen.get_booster().get_score(importance_type="total_gain")
        importance = pl.DataFrame({"factor_name": names,
                                   "importance_gain": [float(gain.get(name, 0.0)) for name in names]})
        importance.sort("importance_gain", descending=True).write_parquet(run / "feature_importance.parquet")
        result = run_daily_equal_weight(
            predictions, bars, PITRankXGBoostLongOnlyStrategy(config.fraction),
            config.one_way_fee_rate, config.one_way_slippage_rate,
        )
        backtest = self.settings.resolved_data_path() / "backtests" / STRATEGY_ID / run_id
        backtest.mkdir(parents=True, exist_ok=False)
        monthly = monthly_performance(result.daily)
        for name, frame in (("portfolio", result.portfolio), ("daily", result.daily),
                            ("trades", result.trades), ("data_issues", result.data_issues), ("monthly", monthly)):
            frame.write_parquet(backtest / f"{name}.parquet")
        portfolio_metrics = {
            **result.metrics,
            "universe_total_return": float(np.prod(1 + result.daily["universe_return"].to_numpy()) - 1),
            "strategy_cumulative_simple_return": float(result.daily["net_return"].sum()),
            "universe_cumulative_simple_return": float(result.daily["universe_return"].sum()),
            "daily_win_rate": float((result.daily["net_return"] > 0).mean()),
            "mean_gross_exposure": float(result.portfolio["gross_exposure"].mean()),
            "backtest_start_ms": int(result.daily["decision_timestamp"].min()),
            "backtest_end_ms": int(result.portfolio["timestamp"].max()),
        }
        manifest = {
            "schema_version": 1, "strategy_key": config.strategy_key, "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(), "config": config.model_dump(),
            "model_type": "xgboost", "xgboost_version": xgb.__version__, "model_count": 1,
            "training_protocol": "fixed_year_train_test", "fit_count": 1,
            "early_stopping": False, "validation_set_count": 0, "parameter_search": False,
            "model_params": params, "final_tree_count": len(frozen.get_booster().get_dump()),
            "boosting_rounds": frozen.get_booster().num_boosted_rounds(), "model_file": "model.ubj",
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "features": names, "feature_transform": "within_timestamp_average_rank_pct_0_1",
            "feature_manifest_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
            "feature_selection_holdout_contamination": feature_manifest["feature_selection_holdout_contamination"],
            "source_data_manifest": str(source_manifest_path),
            "source_data_manifest_sha256": hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
            "actual_source_bar_end_ms": int(bars["timestamp"].max()) + 14_400_000,
            "future_survival_used_for_selection": False, "universe_count": universe.height,
            "timeframe": "4h", "horizon_bars": 6, "label_column": "excess_forward_return",
            "split": split, "training_prediction_metrics": excess_prediction_metrics(training_predictions),
            "test_prediction_metrics": excess_prediction_metrics(predictions), "test_portfolio": portfolio_metrics,
            "test_labels_used_for_training_or_tuning": False, "backtest_root": str(backtest),
        }
        report, chart = self._write_report(manifest, result.daily, monthly)
        manifest["report_path"], manifest["chart_path"] = str(report), str(chart)
        (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        (backtest / "metrics.json").write_text(json.dumps(portfolio_metrics, indent=2) + "\n")
        (run.parent / "latest.json").write_text(json.dumps({
            "strategy_key": config.strategy_key, "run_id": run_id, "manifest": str(run / "manifest.json"),
            "model_type": "xgboost", "training_protocol": "fixed_year_train_test",
        }, indent=2) + "\n")
        progress(f"Saved exactly 300 XGBoost trees and 2026 holdout backtest: {run}")
        progress(f"Return={portfolio_metrics['total_return']:.2%}; drawdown={portfolio_metrics['max_drawdown']:.2%}; "
                 f"excess simple={portfolio_metrics['cumulative_excess_simple_return']:.2%}")
        return run

    def _write_report(self, manifest: dict[str, Any], daily: pl.DataFrame, monthly: pl.DataFrame) -> tuple[Path, Path]:
        directory = self.settings.resolved_reports_path() / "strategies" / STRATEGY_ID
        directory.mkdir(parents=True, exist_ok=True)
        report = directory / f"{manifest['run_id']}.md"
        chart = directory / f"{manifest['run_id']}.html"
        metrics, split, config = manifest["test_portfolio"], manifest["split"], self.config
        start = beijing_time(metrics["backtest_start_ms"])
        end = beijing_time(metrics["backtest_end_ms"])
        dates = [datetime.fromtimestamp((stamp + 86_400_000) / 1000, timezone.utc)
                 for stamp in daily["decision_timestamp"].to_list()]
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=dates, y=daily["equity"].to_list(), name="XGBoost 策略净权益"))
        figure.add_trace(go.Scatter(x=dates,
                                   y=(np.cumprod(1 + daily["universe_return"].to_numpy()) * metrics["initial_balance"]).tolist(),
                                   name="币池每日等权基准"))
        figure.update_layout(title="2026 年 XGBoost 300树策略与币池等权基准", template="plotly_white",
                             xaxis_title="UTC", yaxis_title="USDT", hovermode="x unified",
                             xaxis={"rangeslider": {"visible": True}}, height=650)
        figure.write_html(chart, include_plotlyjs=True)
        rows = [
            "# 2026年回测：2025年训练的 XGBoost 300树 Rank64 策略", "",
            f"- 策略：`{config.strategy_key}`；运行：`{manifest['run_id']}`。",
            f"- 模型：XGBoost {manifest['xgboost_version']}，一个模型，固定300棵树，不早停、不设验证集、不滚动、不搜索参数。",
            "- 训练：2025-01-01至2026-01-01，不含结束边界。标签结束时间必须早于2026年起点。",
            f"- 实际训练首末决策：{format_timestamp_ms(split['train_decision_min'])} 至 {format_timestamp_ms(split['train_decision_max'])}（UTC）。",
            f"- 训练样本{split['train_rows']:,}条，边界剔除{split['boundary_label_purged_rows']}条；2026年测试特征{split['test_feature_rows']:,}条。",
            f"- 计划预测2026年；实际完整回测区间为北京时间 **{start} 至 {end}**，共{metrics['daily_period_count']}个完整日窗口。未来月份及末端未兑现的24小时预测不计入回测。",
            f"- 原始行情截止北京时间{beijing_time(manifest['actual_source_bar_end_ms'])}；沿用2025起点固定币池{manifest['universe_count']}个原始成员，不以未来存续结果反筛。",
            "- 特征：固定64个因子，在每个决策截面独立做0～1平均秩排名。symbol不作为模型特征。",
            "- 标签：该币未来24小时简单收益减去同一固定universe可计算同期完整收益成员的等权平均收益。不是累计收益标签。",
            "- 交易：每天北京时间12:00，预测超额收益前10%等仓只多，目标总仓位100%，不设止损，仅在日调仓时调整仓位。",
            f"- 单边手续费{config.one_way_fee_rate:.2%}、单边滑点{config.one_way_slippage_rate:.2%}，资金费率暂按0。",
            "", "## 2026年整体结果", "",
            "| 指标 | 数值 |", "|---|---:|",
            f"| 策略总收益（净值复利） | {metrics['total_return']:.2%} |",
            f"| 币池等权基准收益（复利） | {metrics['universe_total_return']:.2%} |",
            f"| 策略最大回撤（4h权益） | {metrics['max_drawdown']:.2%} |",
            f"| 策略累计单利收益 | {metrics['strategy_cumulative_simple_return']:.2%} |",
            f"| 币池累计单利收益 | {metrics['universe_cumulative_simple_return']:.2%} |",
            f"| 累计超额单利收益 | {metrics['cumulative_excess_simple_return']:.2%} |",
            f"| 日收益年化Sharpe | {metrics['daily_sharpe']:.3f} |" if metrics['daily_sharpe'] is not None else "| 日收益年化Sharpe | 无法计算 |",
            f"| 日胜率 | {metrics['daily_win_rate']:.2%} |",
            f"| 期末权益（初始10,000 USDT） | {metrics['final_equity']:,.2f} USDT |",
            f"| 交易笔数 / 止损次数 | {metrics['trade_count']} / {metrics['stop_count']} |",
            f"| 交易手续费累计 | {metrics['total_fees']:,.2f} USDT |",
            f"| 执行数据异常 | {metrics['data_issue_count']} |",
            "", "## 月度结果", "",
            "按决策日归属月份，10月是不完整月份。月度超额单利为当月各日策略净收益减币池收益后相加，不等同于两个复利收益直接相减。",
            "", "| 月份 | 天数 | 策略净收益 | 币池等权收益 | 当月超额单利 |", "|---|---:|---:|---:|---:|",
        ]
        for month in monthly.iter_rows(named=True):
            rows.append(f"| {month['month']} | {month['days']} | {month['strategy_return']:.2%} | "
                        f"{month['universe_return']:.2%} | {month['excess_simple_return']:.2%} |")
        rows.extend([
            "", "## 模型诊断", "",
            "训练集指标：", "", f"```json\n{json.dumps(manifest['training_prediction_metrics'], indent=2)}\n```",
            "", "测试集指标：", "", f"```json\n{json.dumps(manifest['test_prediction_metrics'], indent=2)}\n```",
            "", "## 固定模型参数", "", f"```json\n{json.dumps(manifest['model_params'], indent=2)}\n```",
            "", "## 口径与限制", "",
            "- 64因子清单曾使用2026-07相关性数据筛选，所以这次虽只用2025标签拟合，2026仍不能称为完全未参与过特征选择的最终留出集。",
            "- 不执行跳空或盘中止损，4h收盘价用于权益估值；费用和滑点进入净值，未回放分钟或逐笔订单簿，未计真实历史资金费率。",
            "- 基准每天等权重置、不计交易成本。最后的持仓按市值计价，不虚构已清仓。",
            "", "## 文件", "",
            f"- 权益对比图：[{chart.name}]({chart.name})",
            f"- 模型与预测：`data/models/{STRATEGY_ID}/{manifest['run_id']}/`",
            f"- 回测账本与月度表：`data/backtests/{STRATEGY_ID}/{manifest['run_id']}/`",
            "- XGBoost参数参考：[官方文档](https://xgboost.readthedocs.io/en/stable/parameter.html)。",
        ])
        report.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return report, chart
