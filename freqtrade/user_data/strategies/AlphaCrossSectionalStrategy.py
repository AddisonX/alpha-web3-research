from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class AlphaCrossSectionalStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "4h"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 1

    minimal_roi = {}
    stoploss = -0.10
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    _signals: DataFrame | None = None

    plot_config = {
        "subplots": {
            "Model target": {
                "target_position": {"color": "blue"},
                "model_score_rank_pct": {"color": "green"},
            }
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = str(metadata["pair"])
        pair_signals = self._load_signals()
        pair_signals = pair_signals.loc[pair_signals["pair"] == pair].drop(columns="pair")

        frame = dataframe.copy()
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame = frame.merge(pair_signals, on="date", how="left", validate="one_to_one")
        frame["target_position"] = frame["target_position"].fillna(0).astype("int8")
        frame["signal_valid"] = frame["signal_valid"].fillna(False).astype(bool)
        frame["is_rebalance"] = frame["is_rebalance"].fillna(False).astype(bool)
        return frame

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        previous_target = dataframe["target_position"].shift(1).fillna(0).astype("int8")
        target_changed = dataframe["target_position"].ne(previous_target)
        entry_window = target_changed | target_changed.shift(1).fillna(False)
        tradable = dataframe["signal_valid"] & (dataframe["volume"] > 0)

        dataframe.loc[
            tradable & entry_window & (dataframe["target_position"] == 1),
            ["enter_long", "enter_tag"],
        ] = (1, "model_top_quantile")
        dataframe.loc[
            tradable & entry_window & (dataframe["target_position"] == -1),
            ["enter_short", "enter_tag"],
        ] = (1, "model_bottom_quantile")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        previous_target = dataframe["target_position"].shift(1).fillna(0).astype("int8")
        active_volume = dataframe["volume"] > 0

        dataframe.loc[
            active_volume & (previous_target == 1) & (dataframe["target_position"] != 1),
            ["exit_long", "exit_tag"],
        ] = (1, "left_top_quantile")
        dataframe.loc[
            active_volume & (previous_target == -1) & (dataframe["target_position"] != -1),
            ["exit_short", "exit_tag"],
        ] = (1, "left_bottom_quantile")
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return 1.0

    @classmethod
    def _load_signals(cls) -> DataFrame:
        if cls._signals is not None:
            return cls._signals

        default_path = Path(__file__).resolve().parents[1] / "signals" / "alpha_cross_sectional_signals.parquet"
        path = Path(os.environ.get("ALPHA_FREQTRADE_SIGNAL_PATH", default_path)).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Freqtrade signal file not found: {path}. "
                "Run `alpha-web3-research backtest export-freqtrade-signals` first."
            )

        signals = pd.read_parquet(
            path,
            columns=[
                "date",
                "pair",
                "signal_valid",
                "is_rebalance",
                "target_position",
                "model_score",
                "model_score_rank_pct",
            ],
        )
        signals["date"] = pd.to_datetime(signals["date"], utc=True)
        if signals.duplicated(subset=["date", "pair"]).any():
            raise ValueError(f"duplicate date/pair rows found in Freqtrade signals: {path}")
        cls._signals = signals.sort_values(["pair", "date"])
        return cls._signals
