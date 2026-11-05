"""Execution simulation for daily equal-weight longs with optional entry stops."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from alpha_web3_research.data.binance_archive import BAR_MS
from alpha_web3_research.strategies.pit_rank_excess import PITRankExcessLongOnlyStrategy

DAY_MS = 86_400_000


@dataclass(frozen=True)
class DailyEqualWeightResult:
    portfolio: pl.DataFrame
    daily: pl.DataFrame
    trades: pl.DataFrame
    data_issues: pl.DataFrame
    metrics: dict[str, Any]


def run_daily_equal_weight(
    predictions: pl.DataFrame, bars: pl.DataFrame,
    strategy: PITRankExcessLongOnlyStrategy | None = None,
    fee_rate: float = 0.0005, slippage_rate: float = 0.0002,
    initial_balance: float = 10_000.0,
) -> DailyEqualWeightResult:
    strategy = strategy or PITRankExcessLongOnlyStrategy()
    if not 0 <= fee_rate < 0.01 or not 0 <= slippage_rate < 0.01 or initial_balance <= 0:
        raise ValueError("invalid execution costs or initial balance")
    if predictions.is_empty() or bars.is_empty():
        raise ValueError("predictions and execution bars must be nonempty")
    if bars.select(pl.struct("symbol", "timestamp").n_unique()).item() != bars.height:
        raise ValueError("duplicate execution bars")
    available_end = int(bars["timestamp"].max()) + BAR_MS
    prediction_days = predictions.filter(pl.col("decision_timestamp") + DAY_MS <= available_end)
    if prediction_days.is_empty():
        raise ValueError("no completed 24h test windows")
    first_decision = int(prediction_days["decision_timestamp"].min())
    final_end = int(prediction_days["decision_timestamp"].max()) + DAY_MS
    prediction_by_time = {
        int(key[0]): part for key, part in prediction_days.sort(["decision_timestamp", "symbol"]).group_by(
            "decision_timestamp", maintain_order=True)
    }
    bars_by_time = {
        int(key[0]): {row["symbol"]: row for row in part.iter_rows(named=True)}
        for key, part in bars.filter((pl.col("timestamp") >= first_decision)
                                     & (pl.col("timestamp") < final_end)).group_by("timestamp")
    }
    cash = initial_balance
    positions: dict[str, dict[str, float]] = {}
    last_marks: dict[str, float] = {}
    cooldown_until: dict[str, int] = {}
    trades: list[dict[str, Any]] = []
    portfolio: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    equity_peak = initial_balance

    def sell(symbol: str, qty: float, raw_price: float, timestamp: int, reason: str) -> None:
        nonlocal cash
        if qty <= 1e-12:
            return
        price = raw_price * (1.0 - slippage_rate)
        notional = qty * price
        fee = notional * fee_rate
        cash += notional - fee
        trades.append({"timestamp": timestamp, "symbol": symbol, "side": "sell", "quantity": qty,
                       "price": price, "fee": fee, "reason": reason})
        remaining = positions[symbol]["qty"] - qty
        if remaining <= 1e-10:
            positions.pop(symbol)
        else:
            positions[symbol]["qty"] = remaining

    for timestamp in range(first_decision, final_end, BAR_MS):
        current_bars = bars_by_time.get(timestamp, {})
        for symbol in list(positions):
            if symbol not in current_bars:
                issues.append({"timestamp": timestamp, "symbol": symbol, "issue": "held_position_missing_execution_bar"})
                continue
            candle = current_bars[symbol]
            last_marks[symbol] = candle["open"]
            if strategy.stop_loss is None:
                continue
            stop = positions[symbol]["entry_price"] * (1.0 - strategy.stop_loss)
            if candle["open"] <= stop:
                sell(symbol, positions[symbol]["qty"], candle["open"], timestamp, "stop_gap")
                cooldown_until[symbol] = min(
                    (value for value in prediction_by_time if value > timestamp), default=final_end,
                )

        if timestamp in prediction_by_time:
            cross_section = prediction_by_time[timestamp]
            candidates = cross_section.filter(
                pl.col("symbol").is_in(list(current_bars))
                & ~pl.col("symbol").is_in([symbol for symbol, until in cooldown_until.items() if until > timestamp])
            )
            symbols = candidates["symbol"].to_list()
            weights = strategy.select_weights(candidates["prediction"].to_numpy(), symbols)
            desired = {symbol: float(weight) for symbol, weight in zip(symbols, weights, strict=True) if weight > 0}
            liquid_equity = cash + sum(position["qty"] * current_bars[symbol]["open"]
                                       for symbol, position in positions.items() if symbol in current_bars)
            budget = liquid_equity / (1 + fee_rate + slippage_rate)
            for symbol in list(positions):
                if symbol not in current_bars:
                    continue
                price = current_bars[symbol]["open"]
                target_qty = budget * desired.get(symbol, 0.0) / price
                reduction = max(positions[symbol]["qty"] - target_qty, 0.0)
                sell(symbol, reduction, price, timestamp, "rebalance")
            for symbol in sorted(desired):
                raw_price = current_bars[symbol]["open"]
                price = raw_price * (1 + slippage_rate)
                existing = positions.get(symbol, {"qty": 0.0})["qty"]
                quantity = min(max(budget * desired[symbol] / raw_price - existing, 0.0),
                               max(cash, 0.0) / (price * (1 + fee_rate)))
                if quantity * price <= initial_balance * 1e-10:
                    continue
                notional = quantity * price
                fee = notional * fee_rate
                cash -= notional + fee
                if symbol in positions:
                    previous = positions[symbol]
                    previous["entry_price"] = (previous["entry_price"] * existing + price * quantity) / (existing + quantity)
                    previous["qty"] += quantity
                else:
                    positions[symbol] = {"qty": quantity, "entry_price": price}
                trades.append({"timestamp": timestamp, "symbol": symbol, "side": "buy", "quantity": quantity,
                               "price": price, "fee": fee, "reason": "rebalance"})

        for symbol in list(positions):
            if symbol not in current_bars:
                continue
            candle = current_bars[symbol]
            if strategy.stop_loss is None:
                last_marks[symbol] = candle["close"]
                continue
            stop = positions[symbol]["entry_price"] * (1.0 - strategy.stop_loss)
            if candle["low"] <= stop:
                sell(symbol, positions[symbol]["qty"], min(candle["open"], stop), timestamp, "stop_intrabar")
                next_decision = min((value for value in prediction_by_time if value > timestamp), default=final_end)
                cooldown_until[symbol] = next_decision
            else:
                last_marks[symbol] = candle["close"]
        marked_positions = sum(position["qty"] * last_marks[symbol] for symbol, position in positions.items())
        equity = cash + marked_positions
        equity_peak = max(equity_peak, equity)
        if cash < -1e-6 or equity < 0:
            raise ValueError("portfolio accounting produced negative cash/equity")
        portfolio.append({
            "timestamp": timestamp + BAR_MS, "equity": equity, "cash": cash,
            "gross_exposure": marked_positions / equity if equity else 0.0,
            "active_positions": len(positions), "drawdown": equity / equity_peak - 1.0,
        })
    equity_frame = pl.DataFrame(portfolio)
    daily_rows = []
    previous_equity = initial_balance
    for decision in sorted(prediction_by_time):
        end_equity = equity_frame.filter(pl.col("timestamp") == decision + DAY_MS)["equity"].item()
        cross_section = prediction_by_time[decision]
        benchmark = cross_section["universe_forward_return"].drop_nulls().unique().to_list()
        if len(benchmark) != 1:
            benchmark_return = None
            issues.append({"timestamp": decision, "symbol": "UNIVERSE", "issue": "missing_or_inconsistent_benchmark_label"})
        else:
            benchmark_return = float(benchmark[0])
        daily_return = end_equity / previous_equity - 1
        daily_rows.append({"decision_timestamp": decision, "net_return": daily_return,
                           "universe_return": benchmark_return,
                           "excess_return": daily_return - benchmark_return if benchmark_return is not None else None,
                           "equity": end_equity})
        previous_equity = end_equity
    daily_frame = pl.DataFrame(daily_rows).with_columns(pl.col("excess_return").cum_sum().alias("cumulative_excess_simple_return"))
    trade_frame = pl.DataFrame(trades) if trades else pl.DataFrame(schema={
        "timestamp": pl.Int64, "symbol": pl.String, "side": pl.String, "quantity": pl.Float64,
        "price": pl.Float64, "fee": pl.Float64, "reason": pl.String,
    })
    issues_frame = pl.DataFrame(issues) if issues else pl.DataFrame(schema={"timestamp": pl.Int64, "symbol": pl.String, "issue": pl.String})
    returns = daily_frame["net_return"].to_numpy()
    metrics = {
        "strategy_key": strategy.strategy_key, "initial_balance": initial_balance,
        "final_equity": float(equity_frame["equity"][-1]),
        "total_return": float(equity_frame["equity"][-1] / initial_balance - 1),
        "max_drawdown": float(equity_frame["drawdown"].min()),
        "daily_sharpe": float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(365)) if len(returns) > 1 and np.std(returns, ddof=1) > 0 else None,
        "cumulative_excess_simple_return": float(daily_frame["excess_return"].sum()),
        "daily_period_count": len(returns), "trade_count": trade_frame.height,
        "stop_count": trade_frame.filter(pl.col("reason").str.starts_with("stop")).height,
        "total_fees": float(trade_frame["fee"].sum()),
        "data_issue_count": issues_frame.height, "execution_data_complete": issues_frame.is_empty(),
        "fee_rate": fee_rate, "slippage_rate": slippage_rate, "funding_rate_assumption": 0.0,
        "stop_loss": strategy.stop_loss, "fraction": strategy.fraction,
        "open_positions_at_end": len(positions), "final_positions_marked_not_forced_liquidated": True,
    }
    return DailyEqualWeightResult(equity_frame, daily_frame, trade_frame, issues_frame, metrics)
