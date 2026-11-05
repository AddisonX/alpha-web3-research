from __future__ import annotations

from pandas import DataFrame

from AlphaCrossSectionalStrategy import AlphaCrossSectionalStrategy


class AlphaCrossSectionalLongOnlyStrategy(AlphaCrossSectionalStrategy):
    """Trade only the model's top cross-sectional quantile."""

    can_short = False

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        previous_target = dataframe["target_position"].shift(1).fillna(0).astype("int8")
        target_changed = dataframe["target_position"].ne(previous_target)
        entry_window = target_changed | target_changed.shift(1).fillna(False)
        tradable = dataframe["signal_valid"] & (dataframe["volume"] > 0)

        dataframe.loc[
            tradable & entry_window & (dataframe["target_position"] == 1),
            ["enter_long", "enter_tag"],
        ] = (1, "model_top_quantile")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        previous_target = dataframe["target_position"].shift(1).fillna(0).astype("int8")
        active_volume = dataframe["volume"] > 0

        dataframe.loc[
            active_volume & (previous_target == 1) & (dataframe["target_position"] != 1),
            ["exit_long", "exit_tag"],
        ] = (1, "left_top_quantile")
        return dataframe
