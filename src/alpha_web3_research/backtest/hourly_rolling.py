from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from alpha_web3_research.strategies.base import TargetWeightStrategy
from alpha_web3_research.strategies.hourly_multihorizon import (
    HourlyMultiHorizonAbsoluteScoreLongOnlyStrategy,
    capped_normalized_weights as strategy_capped_normalized_weights,
)

HOURS_PER_YEAR = 24 * 365


@dataclass(frozen=True)
class HourlyPortfolioRules:
    gross_exposure: float = 0.90
    max_asset_weight: float = 0.12
    exit_threshold_ratio: float = 0.70
    minimum_trade_weight: float = 0.02
    max_weight_step: float = 0.04
    max_hourly_turnover: float = 0.25
    one_way_cost_rate: float = 0.0007


@dataclass(frozen=True)
class HourlyBacktestResult:
    entry_threshold: float
    metrics: dict[str, Any]
    hourly: pl.DataFrame
    positions: pl.DataFrame


class HourlyRollingPortfolioBacktester:
    """Recompute long-only target weights every hour and trade only the required delta."""

    def __init__(
        self,
        rules: HourlyPortfolioRules | None = None,
        strategy: TargetWeightStrategy | None = None,
    ) -> None:
        self.rules = rules or HourlyPortfolioRules()
        self.strategy = strategy or HourlyMultiHorizonAbsoluteScoreLongOnlyStrategy()
        self._validate_rules()

    def run(self, predictions: pl.DataFrame, entry_threshold: float) -> HourlyBacktestResult:
        required = {"decision_timestamp", "symbol", "score", "predicted_volatility_h24", "next_hour_return"}
        missing = sorted(required - set(predictions.columns))
        if missing:
            raise ValueError(f"hourly backtest predictions missing columns: {', '.join(missing)}")
        if entry_threshold < 0:
            raise ValueError("entry_threshold must be non-negative")

        frame = (
            predictions.select(sorted(required))
            .filter(pl.col("score").is_finite() & pl.col("predicted_volatility_h24").is_finite())
            .sort(["decision_timestamp", "symbol"])
        )
        if frame.is_empty():
            raise ValueError("hourly backtest predictions are empty")
        symbols = frame.get_column("symbol").unique().sort().to_list()
        symbol_to_index = {symbol: index for index, symbol in enumerate(symbols)}
        current = np.zeros(len(symbols), dtype=float)
        equity = 1.0
        equity_peak = 1.0
        exit_threshold = entry_threshold * self.rules.exit_threshold_ratio
        hourly_rows: list[dict[str, Any]] = []
        position_rows: list[dict[str, Any]] = []

        for (timestamp,), timestamp_frame in frame.group_by("decision_timestamp", maintain_order=True):
            indices = np.asarray([symbol_to_index[item] for item in timestamp_frame.get_column("symbol")], dtype=int)
            scores = np.full(len(symbols), np.nan)
            risks = np.full(len(symbols), np.nan)
            next_returns = np.zeros(len(symbols), dtype=float)
            scores[indices] = timestamp_frame.get_column("score").to_numpy()
            risks[indices] = timestamp_frame.get_column("predicted_volatility_h24").to_numpy()
            next_returns[indices] = timestamp_frame.get_column("next_hour_return").fill_null(0.0).to_numpy()

            desired = self.strategy.target_weights(
                scores=scores,
                risks=risks,
                current_weights=current,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                gross_exposure=self.rules.gross_exposure,
                max_asset_weight=self.rules.max_asset_weight,
            )
            eligible = desired > 0
            post_trade, turnover = apply_trade_controls(current, desired, self.rules)
            transaction_cost = turnover * self.rules.one_way_cost_rate
            gross_return = float(np.dot(post_trade, next_returns))
            net_return = gross_return - transaction_cost
            equity_before = equity
            equity *= 1.0 + net_return
            equity_peak = max(equity_peak, equity)
            drawdown = equity / equity_peak - 1.0

            hourly_rows.append(
                {
                    "decision_timestamp": int(timestamp),
                    "gross_return": gross_return,
                    "transaction_cost": transaction_cost,
                    "net_return": net_return,
                    "equity": equity,
                    "drawdown": drawdown,
                    "turnover": turnover,
                    "gross_exposure": float(post_trade.sum()),
                    "active_positions": int((post_trade >= 0.001).sum()),
                    "eligible_symbols": int(eligible.sum()),
                    "transaction_cost_value": equity_before * transaction_cost,
                }
            )
            changed = (np.abs(post_trade - current) > 1e-12) | (post_trade > 1e-8) | eligible
            for index in np.flatnonzero(changed):
                position_rows.append(
                    {
                        "decision_timestamp": int(timestamp),
                        "symbol": symbols[index],
                        "score": float(scores[index]) if np.isfinite(scores[index]) else None,
                        "predicted_volatility_h24": float(risks[index]) if np.isfinite(risks[index]) else None,
                        "eligible": bool(eligible[index]),
                        "weight_before": float(current[index]),
                        "target_weight": float(desired[index]),
                        "weight_after": float(post_trade[index]),
                        "traded_weight": float(post_trade[index] - current[index]),
                        "next_hour_return": float(next_returns[index]),
                        "gross_contribution": float(post_trade[index] * next_returns[index]),
                    }
                )

            denominator = 1.0 + net_return
            if denominator <= 0:
                current = np.zeros_like(current)
            else:
                current = post_trade * (1.0 + next_returns) / denominator
                current = np.maximum(current, 0.0)

        hourly = pl.DataFrame(hourly_rows)
        positions = pl.DataFrame(position_rows)
        metrics = calculate_metrics(hourly, entry_threshold, exit_threshold, self.rules)
        metrics["strategy_key"] = self.strategy.strategy_key
        if not positions.is_empty():
            contributions = positions.group_by("symbol").agg(pl.col("gross_contribution").sum()).sort(
                "gross_contribution", descending=True
            )
            top = contributions.row(0, named=True)
            positive_total = contributions.filter(pl.col("gross_contribution") > 0).get_column("gross_contribution").sum()
            metrics["top_contributor"] = str(top["symbol"])
            metrics["top_contributor_gross_return"] = float(top["gross_contribution"])
            metrics["top_contributor_share_of_positive_gross"] = (
                float(top["gross_contribution"] / positive_total) if positive_total and positive_total > 0 else None
            )
        return HourlyBacktestResult(
            entry_threshold=entry_threshold,
            metrics=metrics,
            hourly=hourly,
            positions=positions,
        )

    def tune_thresholds(self, predictions: pl.DataFrame, thresholds: list[float]) -> tuple[float, pl.DataFrame]:
        if not thresholds:
            raise ValueError("threshold grid cannot be empty")
        rows: list[dict[str, Any]] = []
        for threshold in thresholds:
            result = self.run(predictions, entry_threshold=threshold)
            rows.append({"entry_threshold": threshold, **result.metrics})
        summary = pl.DataFrame(rows).sort(
            ["annualized_sharpe", "total_return", "max_drawdown"],
            descending=[True, True, True],
            nulls_last=True,
        )
        viable = summary.filter((pl.col("active_hour_fraction") >= 0.10) & (pl.col("total_turnover") > 0))
        chosen = viable if not viable.is_empty() else summary
        return float(chosen.get_column("entry_threshold").item(0)), summary

    def _validate_rules(self) -> None:
        if not 0 < self.rules.gross_exposure <= 1:
            raise ValueError("gross_exposure must be in (0, 1]")
        if not 0 < self.rules.max_asset_weight <= self.rules.gross_exposure:
            raise ValueError("max_asset_weight must be positive and no larger than gross_exposure")
        if not 0 <= self.rules.exit_threshold_ratio <= 1:
            raise ValueError("exit_threshold_ratio must be in [0, 1]")
        if min(
            self.rules.minimum_trade_weight,
            self.rules.max_weight_step,
            self.rules.max_hourly_turnover,
            self.rules.one_way_cost_rate,
        ) < 0:
            raise ValueError("trade controls and costs must be non-negative")


def capped_normalized_weights(raw: np.ndarray, gross_exposure: float, max_asset_weight: float) -> np.ndarray:
    """Compatibility wrapper; allocation policy lives in the strategy module."""

    return strategy_capped_normalized_weights(raw, gross_exposure, max_asset_weight)


def apply_trade_controls(current: np.ndarray, desired: np.ndarray, rules: HourlyPortfolioRules) -> tuple[np.ndarray, float]:
    delta = desired - current
    forced_exit = (desired <= 1e-12) & (current > 1e-12)
    forced_cap_reduction = (current > rules.max_asset_weight) & (desired < current)
    forced_reduction = forced_exit | forced_cap_reduction
    delta = np.where(forced_reduction, desired - current, np.clip(delta, -rules.max_weight_step, rules.max_weight_step))
    delta = np.where((np.abs(delta) < rules.minimum_trade_weight) & ~forced_reduction, 0.0, delta)

    reductions = np.minimum(delta, 0.0)
    reduction_turnover = float(-reductions.sum())
    if reduction_turnover > rules.max_hourly_turnover and reduction_turnover > 0:
        reductions *= rules.max_hourly_turnover / reduction_turnover
    after_reductions = np.maximum(current + reductions, 0.0)
    used_turnover = float(-reductions.sum())

    additions = np.maximum(delta, 0.0)
    addition_budget = min(
        max(rules.max_hourly_turnover - used_turnover, 0.0),
        max(rules.gross_exposure - float(after_reductions.sum()), 0.0),
    )
    requested_additions = float(additions.sum())
    if requested_additions > addition_budget and requested_additions > 0:
        additions *= addition_budget / requested_additions
    post_trade = np.maximum(after_reductions + additions, 0.0)
    turnover = float(np.abs(post_trade - current).sum())
    return post_trade, turnover


def calculate_metrics(
    hourly: pl.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    rules: HourlyPortfolioRules,
) -> dict[str, Any]:
    returns = hourly.get_column("net_return").to_numpy()
    mean_return = float(np.mean(returns)) if returns.size else 0.0
    volatility = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    sharpe = mean_return / volatility * np.sqrt(HOURS_PER_YEAR) if volatility > 0 else None
    return {
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
        "start_timestamp": int(hourly.get_column("decision_timestamp").min()),
        "end_timestamp": int(hourly.get_column("decision_timestamp").max()),
        "hours": hourly.height,
        "total_return": float(hourly.get_column("equity").tail(1).item() - 1.0),
        "annualized_sharpe": sharpe,
        "annualized_volatility": volatility * np.sqrt(HOURS_PER_YEAR),
        "max_drawdown": float(hourly.get_column("drawdown").min()),
        "positive_hour_rate": float((hourly.get_column("net_return") > 0).mean()),
        "average_gross_exposure": float(hourly.get_column("gross_exposure").mean()),
        "average_active_positions": float(hourly.get_column("active_positions").mean()),
        "max_active_positions": int(hourly.get_column("active_positions").max()),
        "active_hour_fraction": float((hourly.get_column("active_positions") > 0).mean()),
        "average_hourly_turnover": float(hourly.get_column("turnover").mean()),
        "total_turnover": float(hourly.get_column("turnover").sum()),
        "transaction_cost_value": float(hourly.get_column("transaction_cost_value").sum()),
        "rules": asdict(rules),
    }
