"""One fixed train/validation/test split, with early stopping on the validation tail."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import polars as pl

from alpha_web3_research.backtest.daily_equal_weight import run_daily_equal_weight
from alpha_web3_research.config import Settings
from alpha_web3_research.data.time import format_timestamp_ms, parse_timestamp_ms
from alpha_web3_research.models.pit_rank_dataset import build_pit_rank_dataset
from alpha_web3_research.research.pit_rank import PITRankResearchConfig, read_bars, research_root
from alpha_web3_research.strategies.pit_rank_excess import PITRankExcessLongOnlyStrategy, STRATEGY_ID


def purged_training_data(dataset: pl.DataFrame, start: int, cutoff: int) -> pl.DataFrame:
    return dataset.filter(
        (pl.col("decision_timestamp") >= start)
        & (pl.col("decision_timestamp") < cutoff)
        & (pl.col("label_end_timestamp") < cutoff)
        & pl.col("excess_forward_return").is_finite()
    )


def fixed_train_valid_test_split(
    dataset: pl.DataFrame, config: PITRankResearchConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    start = parse_timestamp_ms(config.train_start)
    validation_start = parse_timestamp_ms(config.validation_start)
    cutoff = parse_timestamp_ms(config.test_start)
    end = parse_timestamp_ms(config.requested_end)
    train = purged_training_data(dataset, start, validation_start)
    valid = purged_training_data(dataset, validation_start, cutoff)
    # Inference membership must not depend on whether the future test label exists.
    test = dataset.filter((pl.col("decision_timestamp") >= cutoff) & (pl.col("decision_timestamp") < end))
    if train.is_empty() or valid.is_empty() or test.is_empty():
        raise ValueError("fixed training, validation and test sets must all be nonempty")
    train_boundary_purged = dataset.filter(
        (pl.col("decision_timestamp") >= start) & (pl.col("decision_timestamp") < validation_start)
        & (pl.col("label_end_timestamp") >= validation_start)
    ).height
    valid_boundary_purged = dataset.filter(
        (pl.col("decision_timestamp") >= validation_start) & (pl.col("decision_timestamp") < cutoff)
        & (pl.col("label_end_timestamp") >= cutoff)
    ).height
    metadata = {
        "training_protocol": "fixed_train_valid_test", "fit_count": 1, "validation_set_count": 1,
        "parameter_search": False, "early_stopping": True,
        "early_stopping_rounds": config.early_stopping_rounds,
        "requested_train_start": config.train_start, "train_end_exclusive": config.validation_start,
        "validation_start": config.validation_start, "validation_end_exclusive": config.test_start,
        "requested_test_end_exclusive": config.requested_end,
        "train_rows": train.height, "validation_rows": valid.height, "test_rows": test.height,
        "train_decision_min": int(train["decision_timestamp"].min()),
        "train_decision_max": int(train["decision_timestamp"].max()),
        "train_label_end_max": int(train["label_end_timestamp"].max()),
        "validation_decision_min": int(valid["decision_timestamp"].min()),
        "validation_decision_max": int(valid["decision_timestamp"].max()),
        "validation_label_end_max": int(valid["label_end_timestamp"].max()),
        "test_decision_min": int(test["decision_timestamp"].min()),
        "test_decision_max": int(test["decision_timestamp"].max()),
        "train_boundary_label_purged_rows": train_boundary_purged,
        "validation_boundary_label_purged_rows": valid_boundary_purged,
    }
    return train, valid, test, metadata


def excess_prediction_metrics(predictions: pl.DataFrame) -> dict[str, Any]:
    frame = predictions.filter(pl.col("prediction").is_finite() & pl.col("excess_forward_return").is_finite())
    if frame.is_empty():
        return {"sample_count": 0, "rmse": None, "rank_ic": None, "rank_icir": None}
    error = frame["prediction"].to_numpy() - frame["excess_forward_return"].to_numpy()
    ranked = frame.with_columns(
        pl.col("prediction").rank(method="average").over("decision_timestamp").alias("prediction_rank"),
        pl.col("excess_forward_return").rank(method="average").over("decision_timestamp").alias("return_rank"),
    ).group_by("decision_timestamp").agg(pl.corr("prediction_rank", "return_rank").alias("rank_ic"))
    correlations = ranked["rank_ic"].to_numpy()
    correlations = correlations[np.isfinite(correlations)]
    mean_ic = float(np.mean(correlations)) if len(correlations) else None
    std_ic = float(np.std(correlations, ddof=1)) if len(correlations) > 1 else None
    return {
        "sample_count": frame.height, "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "zero_excess_baseline_rmse": float(np.sqrt(np.mean(frame["excess_forward_return"].to_numpy()**2))),
        "rank_ic": mean_ic, "rank_icir": mean_ic / std_ic if std_ic and mean_ic is not None else None,
        "valid_rank_ic_days": len(correlations),
    }


class PITRankResearchRunner:
    def __init__(self, settings: Settings, config: PITRankResearchConfig) -> None:
        self.settings = settings
        self.config = config

    def run(self, progress: Callable[[str], None] = print) -> Path:
        config = self.config
        source_root = research_root(self.settings, config)
        data_manifest_path = source_root / "data_manifest.json"
        if not data_manifest_path.exists():
            raise FileNotFoundError("prepare the independent PIT research dataset first")
        data_manifest = json.loads(data_manifest_path.read_text())
        prepared_contract = data_manifest.get("data_contract") or {
            key: data_manifest["config"].get(key) for key in config.data_contract()
        }
        if prepared_contract != config.data_contract():
            raise ValueError("prepared data acquisition/universe config differs; prepare data again")
        feature_path = self.settings.project_root / config.features_file
        feature_manifest = json.loads(feature_path.read_text())
        names = feature_manifest["features"]
        if len(names) != 64 or len(set(names)) != 64:
            raise ValueError("the fixed feature manifest must contain exactly 64 distinct factors")
        run_id = f"{STRATEGY_ID}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        run_root = self.settings.resolved_data_path() / "models" / STRATEGY_ID / run_id
        run_root.mkdir(parents=True, exist_ok=False)
        bars = read_bars(source_root / "bars")
        universe = pl.read_parquet(source_root / "universe.parquet").filter(pl.col("included"))
        bars = bars.filter(pl.col("symbol").is_in(universe["symbol"].to_list()))
        progress(f"Building daily ranks for {len(names)} factors and {universe.height} inception members")
        dataset, labels = build_pit_rank_dataset(
            bars, names, self.settings, config.min_cross_section, config.decision_hour_utc,
        )
        dataset.write_parquet(run_root / "dataset.parquet")
        labels.write_parquet(run_root / "labels.parquet")
        final_train, valid, test, split_metadata = fixed_train_valid_test_split(dataset, config)
        (run_root / "split.json").write_text(json.dumps(split_metadata, indent=2) + "\n")
        final_params = {
            "objective": "regression", "learning_rate": config.learning_rate, "n_estimators": config.n_estimators,
            "num_leaves": config.num_leaves, "min_child_samples": config.min_child_samples,
            "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
            "reg_alpha": config.reg_alpha, "reg_lambda": config.reg_lambda,
            "random_state": config.seed, "verbosity": -1, "n_jobs": 4,
            "deterministic": True, "force_col_wise": True,
        }
        progress(f"Fixed split: {final_train.height:,} train, {valid.height:,} validation, {test.height:,} test rows; "
                 f"one fit, at most {config.n_estimators} trees, early stopping on validation only")
        final_model = lgb.LGBMRegressor(**final_params)
        final_model.fit(final_train.select(names).to_numpy(), final_train["excess_forward_return"].to_numpy(),
                        feature_name=names,
                        eval_set=[(valid.select(names).to_numpy(), valid["excess_forward_return"].to_numpy())],
                        eval_metric="rmse", callbacks=[
                            lgb.early_stopping(config.early_stopping_rounds, verbose=False), lgb.log_evaluation(0),
                        ])
        best_iteration = int(final_model.best_iteration_ or config.n_estimators)
        model_path = run_root / "model.txt"
        final_model.booster_.save_model(str(model_path), num_iteration=best_iteration)
        frozen_model = lgb.Booster(model_file=str(model_path))
        valid_predictions = valid.with_columns(
            pl.Series("prediction", frozen_model.predict(valid.select(names).to_numpy())),
            pl.lit("validation").alias("split"),
        )
        valid_predictions.write_parquet(run_root / "validation_predictions.parquet")
        predictions = test.with_columns(
            pl.Series("prediction", frozen_model.predict(test.select(names).to_numpy())),
            pl.lit("test").alias("split"), pl.lit(config.strategy_key).alias("strategy_key"),
        )
        predictions.write_parquet(run_root / "predictions.parquet")
        importance = pl.DataFrame({"factor_name": names,
                                   "importance_gain": frozen_model.feature_importance(importance_type="gain")})
        importance.sort("importance_gain", descending=True).write_parquet(run_root / "feature_importance.parquet")
        result = run_daily_equal_weight(
            predictions, bars, PITRankExcessLongOnlyStrategy(config.fraction, config.stop_loss),
            config.one_way_fee_rate, config.one_way_slippage_rate,
        )
        backtest_root = self.settings.resolved_data_path() / "backtests" / STRATEGY_ID / run_id
        backtest_root.mkdir(parents=True, exist_ok=False)
        for filename, frame in (("portfolio", result.portfolio), ("daily", result.daily),
                                ("trades", result.trades), ("data_issues", result.data_issues)):
            frame.write_parquet(backtest_root / f"{filename}.parquet")
        test_metrics = excess_prediction_metrics(predictions)
        manifest = {
            "schema_version": 1, "run_id": run_id, "strategy_key": config.strategy_key,
            "created_at": datetime.now(timezone.utc).isoformat(), "config": config.model_dump(),
            "model_count": 1, "ensemble_method": None, "model_file": "model.txt",
            "final_tree_count": frozen_model.num_trees(), "best_iteration": best_iteration,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "features": names, "feature_transform": "within_timestamp_average_rank_pct_0_1",
            "label_column": "excess_forward_return", "timeframe": "4h", "horizon_bars": 6,
            "label_contract": "next_24h_open_to_open_return_minus_inception_universe_equal_weight_return",
            "feature_manifest_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
            "feature_selection_holdout_contamination": feature_manifest["feature_selection_holdout_contamination"],
            "data_manifest": str(data_manifest_path),
            "data_manifest_sha256": hashlib.sha256(data_manifest_path.read_bytes()).hexdigest(),
            "future_survival_used_for_selection": False,
            "training_protocol": "fixed_train_valid_test", "fit_count": 1,
            "validation_set_count": 1, "parameter_search": False, "early_stopping": True,
            "early_stopping_rounds": config.early_stopping_rounds,
            "validation_prediction_metrics": excess_prediction_metrics(valid_predictions),
            "model_params": final_params, "split": split_metadata,
            "final_training_start": format_timestamp_ms(int(final_train["decision_timestamp"].min())),
            "final_training_decision_max": format_timestamp_ms(int(final_train["decision_timestamp"].max())),
            "final_training_label_end_max": format_timestamp_ms(int(final_train["label_end_timestamp"].max())),
            "test_frozen_cutoff": config.test_start, "test_labels_used_for_training_or_tuning": False,
            "test_prediction_start": format_timestamp_ms(int(test["decision_timestamp"].min())),
            "test_prediction_end": format_timestamp_ms(int(test["decision_timestamp"].max())),
            "test_prediction_metrics": test_metrics, "test_portfolio": result.metrics,
            "backtest_root": str(backtest_root),
        }
        (run_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        (backtest_root / "metrics.json").write_text(json.dumps(result.metrics, indent=2) + "\n")
        self._write_report(manifest, run_root, source_root)
        (run_root.parent / "latest.json").write_text(json.dumps({
            "strategy_key": config.strategy_key, "run_id": run_id,
            "manifest": str(run_root / "manifest.json"), "model_count": 1,
            "training_protocol": "fixed_train_valid_test",
        }, indent=2) + "\n")
        progress(f"Saved one frozen model and holdout results: {run_root}")
        return run_root

    def _write_report(self, manifest: dict[str, Any], run_root: Path, source_root: Path) -> Path:
        report_root = self.settings.resolved_reports_path() / "strategies" / STRATEGY_ID
        report_root.mkdir(parents=True, exist_ok=True)
        path = report_root / f"{manifest['run_id']}.md"
        portfolio = manifest["test_portfolio"]
        config = self.config
        audit = pl.read_parquet(source_root / "survival_audit.parquet")
        missing = audit.filter(~pl.col("data_reaches_audit_end"))["symbol"].to_list()
        lines = [
            "# PIT Rank64 单模型只多策略研究结果", "",
            f"- 策略：`{config.strategy_key}`。",
            f"- 起点币池：在 `{config.inception}` 使用之前已收盘历史、180 天历史覆盖及 30 天成交额筛选，固定 {audit.height} 个币。",
            f"- 到最新归档仍有数据：{audit.height - len(missing)}/{audit.height}。未存续成员继续保留在历史样本中，不使用结局反筛。",
            f"- 原始数据：独立命名空间 `{source_root.relative_to(self.settings.project_root)}`。",
            "- 因子：固定64个，每个日决策截面将原始值转换为0～1平均秩排名，symbol只作标识。",
            "- 标签：决策后24小时价格简单收益，减去同一固定币池中有完整同期标签成员的等权平均收益。不使用累计收益作单日标签。",
            f"- 一次固定划分：训练集从{config.train_start}至{config.validation_start}，验证集至{config.test_start}，测试集至{config.requested_end}，均左闭右开。不滚动、不划fold、不搜索参数。",
            f"- 最终训练区间起点：{manifest['final_training_start']}，最后标签结束：{manifest['final_training_label_end_max']}。",
            "- 仅保存一个最终模型。测试期2026-05起固定该模型，不使用测试标签重训或选择参数。",
            f"- 树数上限{config.n_estimators}，验证RMSE连续{config.early_stopping_rounds}轮不改善则early stopping，最佳迭代{manifest['best_iteration']}，最终保存{manifest['final_tree_count']}棵树；叶子数{config.num_leaves}，学习率{config.learning_rate}。",
            "- 验证集只决定停止迭代，不将验证数据重新加入训练。测试集不用于early stopping。",
            f"- 训练样本：{manifest['split']['train_rows']:,}，验证样本：{manifest['split']['validation_rows']:,}，测试特征样本：{manifest['split']['test_rows']:,}；训练/验证边界分别剔除跨界标签样本{manifest['split']['train_boundary_label_purged_rows']}/{manifest['split']['validation_boundary_label_purged_rows']}条。",
            f"- 测试预测：{manifest['test_prediction_start']} 至 {manifest['test_prediction_end']}；10月只统计已获得行情。",
            "- 每日UTC04:00（北京时间12:00）选模型分数前10%，向上取整，等仓只多。总目标仓位100%，成本占用会留下少量现金。",
            "- 单币相对实际平均入场价止损10%，检查4h high/low，跳空跌破按更差的开盘价成交，止损后当日不重开。同一币持续持有时不每日重置止损基准。",
            "", "## 验证集模型指标", "",
            f"```json\n{json.dumps(manifest['validation_prediction_metrics'], ensure_ascii=False, indent=2)}\n```", "",
            "## 测试集模型指标", "",
            f"```json\n{json.dumps(manifest['test_prediction_metrics'], ensure_ascii=False, indent=2)}\n```", "",
            "## 执行回测", "",
            f"- 总收益：{portfolio['total_return']:.2%}",
            f"- 最大回撤：{portfolio['max_drawdown']:.2%}",
            f"- 累计超额单利收益：{portfolio['cumulative_excess_simple_return']:.2%}",
            f"- 完整日窗口：{portfolio['daily_period_count']}；交易笔数：{portfolio['trade_count']}；止损次数：{portfolio['stop_count']}。",
            f"- 执行数据异常：{portfolio['data_issue_count']}，完整性通过：{portfolio['execution_data_complete']}。缺失持仓行情不会被当作零收益或按预知价格提前平仓。",
            f"- 单边手续费 {config.one_way_fee_rate:.2%}，单边滑点 {config.one_way_slippage_rate:.2%}；资金费率暂按0，未回放订单簿。",
            "", "## 研究限制", "",
            "- 用户指定的64因子清单来自2026-07相关性筛选，已经参考了部分测试期数据。尽管本次拟合没有用测试标签，测试期仍不能声称完全未参与特征选择。",
            "- 归档目录可发现历史合约，存续审计只证明数据覆盖，缺失不等于已取得官方下架结论。符号迁移不自动拼接。",
            "- 原生4h止损只知道bar内是否触及阈值，不具有分钟或逐笔执行精度。",
            "", "## 未覆盖到末端的原始成员", "",
            ", ".join(missing) if missing else "全部原始成员均覆盖到审计末端。",
            "", "## 起点成员存续审计明细", "",
            "按2024年末流动性排名；时间采用UTC，末端覆盖指已获得的归档末端。",
            "", "| 起点排名 | 币种 | 最近30日日成交额中位数（USDT） | 最后可见bar收盘 | 覆盖归档末端 |",
            "|---:|---|---:|---|---|",
        ]
        for row in audit.sort("liquidity_rank").iter_rows(named=True):
            observed_close = format_timestamp_ms(row["observed_last_open_ms"] + 14_400_000) if row["observed_last_open_ms"] is not None else "缺失"
            lines.append(
                f"| {row['liquidity_rank']} | `{row['symbol']}` | {row['median_daily_quote_volume']:,.0f} | "
                f"{observed_close} | {'是' if row['data_reaches_audit_end'] else '否'} |"
            )
        lines.extend([
            "", "## 产物", "",
            f"- 模型及预测：`{run_root.relative_to(self.settings.project_root)}`",
            f"- 回测：`{Path(manifest['backtest_root']).relative_to(self.settings.project_root)}`",
        ])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
