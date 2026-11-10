from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from src.live.safety import (
    LiveSignalHealth,
    check_entry_audit,
    check_live_signal_status,
    validate_signal_rows,
)
from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)


class AlphaCrossSectionalLiveLongOnlyStrategy(IStrategy):
    """Reconcile long-only positions against the latest healthy production target."""

    can_short = False
    process_only_new_candles = False

    INTERFACE_VERSION = 3


    timeframe = "4h"


    startup_candle_count = 1


    minimal_roi = {}


    stoploss = -0.10


    trailing_stop = False


    use_exit_signal = True


    exit_profit_only = False


    ignore_roi_if_entry_signal = False


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


    _signals: DataFrame | None = None
    _signals_mtime_ns: int | None = None
    _last_health_reason: str | None = None
    _validated_key: tuple[int, int, int, frozenset[str], int] | None = None
    _validation_error: str | None = None

    @classmethod
    def _signal_path(cls) -> Path:
        default_path = (
            Path(__file__).resolve().parents[1]
            / "instances"
            / "tokyo-lightgbm"
            / "signals"
            / "alpha_live_long_only_signals.parquet"
        )
        return Path(
            os.environ.get("ALPHA_FREQTRADE_LIVE_SIGNAL_PATH", default_path)
        ).expanduser()

    @classmethod
    def _status_path(cls) -> Path:
        default_path = cls._signal_path().with_name("alpha_live_long_only_status.json")
        return Path(os.environ.get("ALPHA_FREQTRADE_LIVE_STATUS_PATH", default_path)).expanduser()

    @classmethod
    def _health(cls, now: datetime) -> LiveSignalHealth:
        try:
            status = json.loads(cls._status_path().read_text(encoding="utf-8"))
            if not isinstance(status, dict):
                raise ValueError("status must be a JSON object")
            health = check_live_signal_status(status, now)
        except (OSError, ValueError, TypeError) as exc:
            health = LiveSignalHealth(False, f"cannot read signal status: {exc}")
        if health.ready:
            try:
                error = cls._validate_signal_file(health)
            except Exception as exc:
                logger.error("Cannot validate live signal file: %s", exc)
                error = "signal file validation failed"
            if error is not None:
                health = LiveSignalHealth(False, error)
        if health.reason != cls._last_health_reason:
            if health.ready:
                logger.info("Live model trading resumed for decision %s", health.decision_timestamp)
            else:
                logger.warning("Live model trading paused: %s", health.reason)
            cls._last_health_reason = health.reason
        return health

    @classmethod
    def _validate_signal_file(cls, health: LiveSignalHealth) -> str | None:
        signals = cls._load_signals()
        if cls._signals_mtime_ns is None or health.decision_timestamp is None:
            return "signal file is missing or unreadable"
        key = (
            cls._signals_mtime_ns,
            health.decision_timestamp,
            health.valid_until,
            health.selected_pairs,
            health.cross_section_count,
        )
        if key == cls._validated_key:
            return cls._validation_error

        error = validate_signal_rows(signals, health)
        cls._validated_key = key
        cls._validation_error = error
        return error

    @classmethod
    def _load_signals(cls) -> DataFrame:
        path = cls._signal_path()
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            cls._signals = None
            cls._signals_mtime_ns = None
            cls._validated_key = None
            return cls._empty_signals()

        if cls._signals is not None and cls._signals_mtime_ns == mtime_ns:
            return cls._signals

        try:
            signals = pd.read_parquet(
                path,
                columns=[
                    "date",
                    "pair",
                    "decision_timestamp",
                    "signal_valid_until",
                    "signal_valid",
                    "is_rebalance",
                    "target_position",
                    "model_score",
                    "model_score_rank_pct",
                ],
            )
            signals["date"] = pd.to_datetime(signals["date"], utc=True)
            if signals.duplicated(subset=["date", "pair"]).any():
                raise ValueError(f"duplicate date/pair rows found in live signals: {path}")
        except Exception as exc:
            logger.error("Cannot load live signals from %s: %s", path, exc)
            cls._signals = None
            cls._signals_mtime_ns = None
            cls._validated_key = None
            return cls._empty_signals()
        cls._signals = signals.sort_values(["pair", "date"])
        cls._signals_mtime_ns = mtime_ns
        return cls._signals

    @staticmethod
    def _empty_signals() -> DataFrame:
        return DataFrame(
            {
                "date": pd.Series([], dtype="datetime64[ns, UTC]"),
                "pair": pd.Series([], dtype="object"),
                "decision_timestamp": pd.Series([], dtype="int64"),
                "signal_valid_until": pd.Series([], dtype="int64"),
                "signal_valid": pd.Series([], dtype="bool"),
                "is_rebalance": pd.Series([], dtype="bool"),
                "target_position": pd.Series([], dtype="int8"),
                "model_score": pd.Series([], dtype="float64"),
                "model_score_rank_pct": pd.Series([], dtype="float64"),
            }
        )

    @staticmethod
    def _recent_candle(row: pd.Series, now: datetime) -> bool:
        candle_date = pd.Timestamp(row["date"])
        age = pd.Timestamp(now) - candle_date
        return pd.Timedelta(hours=4) <= age < pd.Timedelta(hours=8)

    @staticmethod
    def _row_matches_decision(row: pd.Series, health: LiveSignalHealth, now: datetime) -> bool:
        if not bool(row.get("signal_valid", False)) or pd.isna(row.get("decision_timestamp")):
            return False
        if int(row["decision_timestamp"]) != health.decision_timestamp:
            return False
        valid_until = row.get("signal_valid_until")
        if pd.isna(valid_until) or int(valid_until) != health.valid_until:
            return False
        return int(now.timestamp() * 1000) < int(valid_until)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe
        dataframe.loc[dataframe.index[-1], ["enter_long", "enter_tag"]] = (0, None)
        now = datetime.now(timezone.utc)
        health = self._health(now)
        row = dataframe.iloc[-1]
        if (
            health.ready
            and metadata["pair"] in health.selected_pairs
            and self._recent_candle(row, now)
            and self._row_matches_decision(row, health, now)
            and row["target_position"] == 1
            and row["volume"] > 0
        ):
            dataframe.loc[dataframe.index[-1], ["enter_long", "enter_tag"]] = (1, "model_target")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe
        dataframe.loc[dataframe.index[-1], ["exit_long", "exit_tag"]] = (0, None)
        now = datetime.now(timezone.utc)
        health = self._health(now)
        row = dataframe.iloc[-1]
        if (
            health.ready
            and metadata["pair"] not in health.selected_pairs
            and self._recent_candle(row, now)
            and row["volume"] > 0
        ):
            dataframe.loc[dataframe.index[-1], ["exit_long", "exit_tag"]] = (1, "left_model_target")
        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        try:
            return self._confirm_live_entry(pair, current_time, side)
        except Exception:
            logger.exception("Entry blocked: unexpected safety-check failure for %s", pair)
            return False

    def _confirm_live_entry(self, pair: str, current_time: datetime, side: str) -> bool:
        if not self._execution_mode_allowed():
            logger.error("Real-money entry blocked: live execution has not passed safety preflight")
            return False
        health = self._health(current_time)
        if not health.ready or side != "long" or pair not in health.selected_pairs:
            return False
        audit_path = os.environ.get("ALPHA_LIVE_AUDIT_PATH")
        if audit_path:
            try:
                report = json.loads(Path(audit_path).expanduser().read_text(encoding="utf-8"))
                allowed, reason = check_entry_audit(
                    report, current_time, health.decision_timestamp, health.selected_pairs
                )
            except Exception as exc:
                logger.warning("Entry blocked: cannot read safety audit: %s", exc)
                return False
            if not allowed:
                logger.warning("Entry blocked by safety audit: %s", reason)
                return False
        if getattr(self, "dp", None) is None:
            return False
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        except Exception as exc:
            logger.error("Entry blocked: cannot read analyzed candle for %s: %s", pair, exc)
            return False
        if dataframe.empty:
            return False
        row = dataframe.iloc[-1]
        if not (
            self._recent_candle(row, current_time)
            and self._row_matches_decision(row, health, current_time)
            and row["target_position"] == 1
            and row["volume"] > 0
        ):
            return False

        # A stopped-out target must not be reopened until the next model decision.
        from freqtrade.persistence import Trade

        decision_close = datetime.fromtimestamp(
            (health.decision_timestamp + 4 * 3_600_000) / 1000, timezone.utc
        ).replace(tzinfo=None)
        try:
            return not bool(Trade.get_trades_proxy(pair=pair, is_open=False, close_date=decision_close))
        except Exception as exc:
            logger.error("Entry blocked: cannot check closed trades for %s: %s", pair, exc)
            return False

    def _execution_mode_allowed(self) -> bool:
        return self.config.get("dry_run") is True
