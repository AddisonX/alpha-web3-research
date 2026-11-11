"""Opt-in 1x paper execution of signed, equal-weight daily targets."""
from datetime import datetime, timezone
import math

from AlphaCrossSectionalLiveLongOnlyStrategy import AlphaCrossSectionalLiveLongOnlyStrategy


class AlphaCrossSectionalLiveLongShortStrategy(AlphaCrossSectionalLiveLongOnlyStrategy):
    can_short = True
    position_adjustment_enable = True
    stoploss = -0.99  # Framework emergency stop; the research strategy has no 10% stop.

    def _confirm_live_entry(self, pair, current_time, side):
        if not self._health(current_time).allow_short:
            return False
        return super()._confirm_live_entry(pair, current_time, side)

    def _execution_mode_allowed(self):
        return (super()._execution_mode_allowed() and self.config.get("trading_mode") == "futures"
                and self.config.get("margin_mode") == "isolated")

    def populate_entry_trend(self, dataframe, metadata):
        if dataframe.empty:
            return dataframe
        idx = dataframe.index[-1]
        dataframe.loc[idx, ["enter_long", "enter_short", "enter_tag"]] = (0, 0, None)
        now = datetime.now(timezone.utc)
        health, row = self._health(now), dataframe.iloc[-1]
        if (health.ready and health.allow_short and metadata["pair"] in health.selected_pairs
                and self._recent_candle(row, now) and self._row_matches_decision(row, health, now)
                and row["volume"] > 0):
            direction = "short" if metadata["pair"] in health.short_pairs else "long"
            dataframe.loc[idx, [f"enter_{direction}", "enter_tag"]] = (1, "weighted_model_target")
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        if dataframe.empty:
            return dataframe
        idx = dataframe.index[-1]
        dataframe.loc[idx, ["exit_long", "exit_short", "exit_tag"]] = (0, 0, None)
        now = datetime.now(timezone.utc)
        health, row = self._health(now), dataframe.iloc[-1]
        if health.ready and health.allow_short and self._recent_candle(row, now) and row["volume"] > 0:
            pair = metadata["pair"]
            long_target = pair in health.selected_pairs and pair not in health.short_pairs
            short_target = pair in health.short_pairs
            dataframe.loc[idx, ["exit_long", "exit_short", "exit_tag"]] = (
                int(not long_target), int(not short_target), "left_model_target")
        return dataframe

    def _allocation(self, pair, side, now):
        if not self._confirm_live_entry(pair, now, side):
            return None
        health = self._health(now)
        if not health.ready or not health.allow_short or pair not in health.selected_pairs:
            return None
        if (side == "short") != (pair in health.short_pairs):
            return None
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame.empty or not self._row_matches_decision(frame.iloc[-1], health, now) or not self._recent_candle(frame.iloc[-1], now):
            return None
        weight = abs(float(frame.iloc[-1]["target_weight"]))
        if getattr(self, "_allocation_stamp", None) != health.decision_timestamp:
            from freqtrade.persistence import Trade
            equity = float(self.wallets.get_total_stake_amount())
            for trade in Trade.get_open_trades():
                candles, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                if candles.empty:
                    return None
                equity += trade.calculate_profit(float(candles.iloc[-1]["close"])).profit_abs
            if not math.isfinite(equity) or equity <= 0:
                return None
            self._allocation_equity = equity
            self._allocation_stamp = health.decision_timestamp
        return health.decision_timestamp, self._allocation_equity * weight

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        allocation = self._allocation(pair, side, current_time)
        if allocation is None:
            return 0.
        stake = min(allocation[1], max_stake)
        return stake if stake >= (min_stake or 0) else 0.

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit,
                              min_stake, max_stake, **kwargs):
        if trade.has_open_orders:
            return None
        allocation = self._allocation(trade.pair, "short" if trade.is_short else "long", current_time)
        if allocation is None:
            return None
        stamp, notional = allocation
        if trade.get_custom_data("allocation_done") == stamp:
            return None
        key = f"allocation_{stamp}"
        target_amount = trade.get_custom_data(key)
        if target_amount is None:
            target_amount = notional / current_rate
            trade.set_custom_data(key, target_amount)
        difference = target_amount - trade.amount
        if abs(difference * current_rate) < max(min_stake or 0., notional * .001):
            trade.set_custom_data("allocation_done", stamp)
            return None
        if difference < 0:
            return max(-trade.stake_amount, difference * trade.stake_amount / trade.amount), f"rebalance_{stamp}"
        stake = min(difference * current_rate, max_stake)
        return (stake, f"rebalance_{stamp}") if stake >= (min_stake or 0) else None

    def order_filled(self, pair, trade, order, current_time, **kwargs):
        health = self._health(current_time)
        if not health.ready:
            return
        # Retained positions resize once per decision; a newly opened position already
        # used custom_stake_amount. Partial fills remain eligible until target reached.
        if trade.nr_of_successful_entries == 1 and trade.nr_of_successful_exits == 0:
            if trade.open_date_utc.timestamp()*1000 >= health.decision_timestamp + 14_400_000:
                trade.set_custom_data("allocation_done", health.decision_timestamp)
