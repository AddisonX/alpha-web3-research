from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from alpha_web3_research.config import Settings
from alpha_web3_research.data.storage import ParquetOHLCVStore
from alpha_web3_research.strategies.legacy_cross_sectional import DailyCrossSectionalQuantileStrategy

HOUR_MS = 3_600_000


@dataclass(frozen=True)
class FreqtradeExportResult:
    predictions_path: Path
    signals_path: Path
    metadata_path: Path
    config_path: Path
    long_only_config_path: Path
    pairs_path: Path
    signal_rows: int
    decision_count: int
    pair_count: int
    long_decisions: int
    short_decisions: int


@dataclass(frozen=True)
class FreqtradeOHLCVExportResult:
    data_dir: Path
    pair_count: int
    futures_files: int
    mark_files: int
    futures_rows: int
    mark_rows: int


class FreqtradeSignalExporter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_root = settings.resolved_data_path()
        self.walk_root = self.data_root / "models" / "walk_forward"

    def export(
        self,
        predictions_path: Path | None = None,
        timeframe: str = "4h",
        horizon_bars: int = 6,
        label_column: str = "forward_return",
        top_quantile_fraction: float = 0.10,
        rebalance_interval_hours: int = 24,
        rebalance_hour_utc: int = 0,
        excluded_pairs: list[str] | None = None,
        use_testnet_market_metadata: bool = False,
        output_dir: Path | None = None,
    ) -> FreqtradeExportResult:
        self._validate_inputs(
            top_quantile_fraction=top_quantile_fraction,
            rebalance_interval_hours=rebalance_interval_hours,
            rebalance_hour_utc=rebalance_hour_utc,
        )
        source_path = self._resolve_predictions_path(
            predictions_path=predictions_path,
            timeframe=timeframe,
            horizon_bars=horizon_bars,
            label_column=label_column,
            top_quantile_fraction=top_quantile_fraction,
        )
        predictions = self._read_predictions(
            source_path=source_path,
            timeframe=timeframe,
            horizon_bars=horizon_bars,
        )
        excluded = sorted(set(excluded_pairs or []))
        if excluded:
            unknown = sorted(set(excluded) - set(predictions.get_column("symbol").unique().to_list()))
            if unknown:
                raise ValueError(f"excluded pairs are not present in predictions: {', '.join(unknown)}")
            predictions = predictions.filter(~pl.col("symbol").is_in(excluded))
        signals = self._build_signals(
            predictions=predictions,
            top_quantile_fraction=top_quantile_fraction,
            rebalance_interval_hours=rebalance_interval_hours,
            rebalance_hour_utc=rebalance_hour_utc,
        )

        user_data_dir = (output_dir or self.settings.project_root / "freqtrade" / "user_data").resolve()
        signals_dir = user_data_dir / "signals"
        signals_dir.mkdir(parents=True, exist_ok=True)
        signals_path = signals_dir / "alpha_cross_sectional_signals.parquet"
        metadata_path = signals_dir / "alpha_cross_sectional_signals.json"
        config_path = user_data_dir.parent / "config_backtest.json"
        long_only_config_path = user_data_dir.parent / "config_backtest_long_only.json"
        pairs_path = user_data_dir / "pairs.json"

        pairs = signals.get_column("pair").unique().sort().to_list()
        decision_count = signals.get_column("decision_timestamp").drop_nulls().n_unique()
        decision_rows = signals.filter(pl.col("is_rebalance"))
        long_decisions = decision_rows.filter(pl.col("target_position") == 1).height
        short_decisions = decision_rows.filter(pl.col("target_position") == -1).height
        long_only_max_open_trades = int(
            decision_rows.filter(pl.col("target_position") == 1)
            .group_by("decision_timestamp")
            .len()
            .get_column("len")
            .max()
        )
        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_predictions": str(source_path),
            "timeframe": timeframe,
            "horizon_bars": horizon_bars,
            "label_column": label_column,
            "split": "test",
            "top_quantile_fraction": top_quantile_fraction,
            "rebalance_interval_hours": rebalance_interval_hours,
            "rebalance_hour_utc": rebalance_hour_utc,
            "excluded_pairs": excluded,
            "market_metadata_source": "binance_futures_testnet" if use_testnet_market_metadata else "binance_mainnet",
            "funding_rate_assumption": 0.0,
            "mark_price_source": "local_1h_ohlcv_proxy",
            "signal_rows": signals.height,
            "decision_count": decision_count,
            "pair_count": len(pairs),
            "long_decisions": long_decisions,
            "short_decisions": short_decisions,
            "long_only_max_open_trades": long_only_max_open_trades,
            "signal_start": _format_datetime(signals.get_column("date").min()),
            "signal_end": _format_datetime(signals.get_column("date").max()),
        }

        signals.write_parquet(signals_path)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pairs_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        config_path.write_text(
            json.dumps(
                _freqtrade_config(
                    pairs=pairs,
                    timeframe=timeframe,
                    use_testnet_market_metadata=use_testnet_market_metadata,
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        long_only_config_path.write_text(
            json.dumps(
                _freqtrade_config(
                    pairs=pairs,
                    timeframe=timeframe,
                    use_testnet_market_metadata=use_testnet_market_metadata,
                    max_open_trades=long_only_max_open_trades,
                    bot_name="alpha-cross-sectional-long-only-backtest",
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return FreqtradeExportResult(
            predictions_path=source_path,
            signals_path=signals_path,
            metadata_path=metadata_path,
            config_path=config_path,
            long_only_config_path=long_only_config_path,
            pairs_path=pairs_path,
            signal_rows=signals.height,
            decision_count=decision_count,
            pair_count=len(pairs),
            long_decisions=long_decisions,
            short_decisions=short_decisions,
        )

    def _resolve_predictions_path(
        self,
        predictions_path: Path | None,
        timeframe: str,
        horizon_bars: int,
        label_column: str,
        top_quantile_fraction: float,
    ) -> Path:
        if predictions_path is not None:
            candidate = predictions_path.expanduser()
            if not candidate.is_absolute():
                candidate = self.settings.project_root / candidate
            if not candidate.exists():
                raise FileNotFoundError(f"walk-forward predictions not found: {candidate}")
            return candidate.resolve()

        grid_path = self.walk_root / "grids" / "latest_summary.parquet"
        if not grid_path.exists():
            raise FileNotFoundError(
                "walk-forward grid summary not found; pass --predictions-path or run models walk-forward-grid"
            )
        grid = pl.read_parquet(grid_path).filter(
            (pl.col("timeframe") == timeframe)
            & (pl.col("horizon_bars") == horizon_bars)
            & (pl.col("label_column") == label_column)
            & ((pl.col("top_quantile_fraction") - top_quantile_fraction).abs() < 1e-12)
        )
        if grid.height != 1:
            raise ValueError(
                "expected exactly one matching walk-forward grid row for "
                f"{timeframe}/h{horizon_bars}/{label_column}/top={top_quantile_fraction}, found {grid.height}"
            )
        experiment_id = str(grid.get_column("walk_forward_experiment_id").item())
        candidate = self.walk_root / experiment_id / "predictions.parquet"
        if not candidate.exists():
            raise FileNotFoundError(f"matching predictions file not found: {candidate}")
        return candidate.resolve()

    @staticmethod
    def _read_predictions(source_path: Path, timeframe: str, horizon_bars: int) -> pl.DataFrame:
        required = {
            "symbol",
            "timeframe",
            "timestamp",
            "horizon_bars",
            "split",
            "model_score",
            "model_score_rank_pct",
            "fold",
        }
        predictions = pl.read_parquet(source_path)
        missing = sorted(required - set(predictions.columns))
        if missing:
            raise ValueError(f"predictions file is missing columns: {', '.join(missing)}")
        test = predictions.filter(
            (pl.col("split") == "test")
            & (pl.col("timeframe") == timeframe)
            & (pl.col("horizon_bars") == horizon_bars)
        )
        if test.is_empty():
            raise ValueError("predictions file contains no matching test rows")
        duplicate_count = test.height - test.unique(subset=["timestamp", "symbol"]).height
        if duplicate_count:
            raise ValueError(f"test predictions contain {duplicate_count} duplicate timestamp/symbol rows")
        return test.sort(["symbol", "timestamp"])

    @staticmethod
    def _build_signals(
        predictions: pl.DataFrame,
        top_quantile_fraction: float,
        rebalance_interval_hours: int,
        rebalance_hour_utc: int,
    ) -> pl.DataFrame:
        interval_ms = rebalance_interval_hours * HOUR_MS
        offset_ms = rebalance_hour_utc * HOUR_MS
        strategy = DailyCrossSectionalQuantileStrategy(long_only=False, fraction=top_quantile_fraction)
        timeline = predictions.select("symbol", "timestamp", "fold").sort(["symbol", "timestamp"])
        decisions = (
            predictions.filter(((pl.col("timestamp") - offset_ms) % interval_ms) == 0)
            .sort(["timestamp", "model_score", "symbol"], descending=[False, True, False])
            .with_columns(
                pl.len().over("timestamp").alias("decision_cross_section_count"),
                pl.col("model_score")
                .rank(method="ordinal", descending=True)
                .over("timestamp")
                .alias("decision_score_position"),
            )
            .with_columns(
                (pl.col("decision_cross_section_count") * top_quantile_fraction)
                .ceil()
                .cast(pl.Int32)
                .alias("decision_selected_count")
            )
            .select(
                "symbol",
                pl.col("timestamp").alias("decision_timestamp"),
                pl.col("fold").alias("decision_fold"),
                pl.col("model_score").alias("decision_model_score"),
                pl.col("model_score_rank_pct").alias("decision_score_rank_pct"),
                "decision_cross_section_count",
                "decision_score_position",
                "decision_selected_count",
            )
            .group_by("decision_timestamp", maintain_order=True)
            .map_groups(
                lambda group: group.with_columns(
                    pl.Series(
                        "decision_target_position",
                        strategy.target_positions(
                            group.get_column("decision_model_score").to_numpy(),
                            group.get_column("symbol").to_list(),
                        ),
                    )
                )
            )
            .sort(["symbol", "decision_timestamp"])
        )
        if decisions.is_empty():
            raise ValueError("no rebalance timestamps matched the requested interval and UTC hour")

        signals = timeline.join_asof(
            decisions,
            left_on="timestamp",
            right_on="decision_timestamp",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        is_valid = (
            pl.col("decision_timestamp").is_not_null()
            & (pl.col("timestamp") >= pl.col("decision_timestamp"))
            & (pl.col("timestamp") < pl.col("decision_timestamp") + interval_ms)
        )
        return (
            signals.with_columns(
                pl.from_epoch("timestamp", time_unit="ms").dt.replace_time_zone("UTC").alias("date"),
                pl.col("symbol").alias("pair"),
                is_valid.alias("signal_valid"),
                (pl.col("timestamp") == pl.col("decision_timestamp")).fill_null(False).alias("is_rebalance"),
                pl.when(is_valid)
                .then(pl.col("decision_target_position"))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias("target_position"),
                pl.when(is_valid).then(pl.col("decision_model_score")).otherwise(None).alias("model_score"),
                pl.when(is_valid).then(pl.col("decision_score_rank_pct")).otherwise(None).alias("model_score_rank_pct"),
                pl.when(is_valid)
                .then(pl.col("decision_timestamp") + interval_ms)
                .otherwise(None)
                .alias("signal_valid_until"),
            )
            .select(
                "date",
                "timestamp",
                "pair",
                "fold",
                "decision_timestamp",
                "signal_valid_until",
                "signal_valid",
                "is_rebalance",
                "target_position",
                "model_score",
                "model_score_rank_pct",
            )
            .sort(["date", "pair"])
        )

    @staticmethod
    def _validate_inputs(
        top_quantile_fraction: float,
        rebalance_interval_hours: int,
        rebalance_hour_utc: int,
    ) -> None:
        if not 0 < top_quantile_fraction < 0.5:
            raise ValueError("top_quantile_fraction must be greater than 0 and less than 0.5")
        if rebalance_interval_hours <= 0:
            raise ValueError("rebalance_interval_hours must be positive")
        if not 0 <= rebalance_hour_utc <= 23:
            raise ValueError("rebalance_hour_utc must be between 0 and 23")
        if rebalance_hour_utc >= rebalance_interval_hours:
            raise ValueError("rebalance_hour_utc must be smaller than rebalance_interval_hours")


class FreqtradeOHLCVExporter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = ParquetOHLCVStore(settings=settings)

    def export(
        self,
        pairs_path: Path | None = None,
        output_dir: Path | None = None,
        timeframe: str = "4h",
        mark_timeframe: str = "1h",
    ) -> FreqtradeOHLCVExportResult:
        user_data_dir = (output_dir or self.settings.project_root / "freqtrade" / "user_data").resolve()
        resolved_pairs_path = pairs_path or user_data_dir / "pairs.json"
        if not resolved_pairs_path.is_absolute():
            resolved_pairs_path = self.settings.project_root / resolved_pairs_path
        if not resolved_pairs_path.exists():
            raise FileNotFoundError(
                f"Freqtrade pairs file not found: {resolved_pairs_path}. "
                "Run backtest export-freqtrade-signals first."
            )
        pairs = json.loads(resolved_pairs_path.read_text(encoding="utf-8"))
        if not isinstance(pairs, list) or not all(isinstance(pair, str) for pair in pairs):
            raise ValueError(f"Freqtrade pairs file must contain a JSON string list: {resolved_pairs_path}")

        data_dir = user_data_dir / "data" / "binance" / "futures"
        data_dir.mkdir(parents=True, exist_ok=True)
        futures_files = 0
        mark_files = 0
        futures_rows = 0
        mark_rows = 0
        for pair in pairs:
            futures = self._read_pair(pair=pair, timeframe=timeframe)
            mark = self._read_pair(pair=pair, timeframe=mark_timeframe)
            futures_path = data_dir / f"{_pair_to_filename(pair)}-{timeframe}-futures.feather"
            mark_path = data_dir / f"{_pair_to_filename(pair)}-{mark_timeframe}-mark.feather"
            self._write_feather(futures, futures_path)
            self._write_feather(mark, mark_path)
            futures_files += 1
            mark_files += 1
            futures_rows += futures.height
            mark_rows += mark.height

        return FreqtradeOHLCVExportResult(
            data_dir=data_dir,
            pair_count=len(pairs),
            futures_files=futures_files,
            mark_files=mark_files,
            futures_rows=futures_rows,
            mark_rows=mark_rows,
        )

    def _read_pair(self, pair: str, timeframe: str) -> pl.DataFrame:
        path = self.store.ohlcv_path(
            exchange=self.settings.exchange_id,
            market_type=self.settings.market_type,
            symbol=pair,
            timeframe=timeframe,
        )
        if not path.exists():
            raise FileNotFoundError(f"local OHLCV not found for {pair} {timeframe}: {path}")
        frame = pl.read_parquet(path).select("timestamp", "open", "high", "low", "close", "volume")
        if frame.is_empty():
            raise ValueError(f"local OHLCV is empty for {pair} {timeframe}: {path}")
        return frame.unique(subset=["timestamp"], keep="last").sort("timestamp")

    @staticmethod
    def _write_feather(frame: pl.DataFrame, path: Path) -> None:
        pandas_frame = frame.with_columns(
            pl.from_epoch("timestamp", time_unit="ms").dt.replace_time_zone("UTC").alias("date")
        ).select("date", "open", "high", "low", "close", "volume").to_pandas()
        pandas_frame.to_feather(path, compression="lz4", compression_level=9)


def _freqtrade_config(
    pairs: list[str],
    timeframe: str,
    use_testnet_market_metadata: bool,
    max_open_trades: int = 18,
    bot_name: str = "alpha-cross-sectional-backtest",
) -> dict[str, Any]:
    ccxt_config: dict[str, Any] = {}
    if use_testnet_market_metadata:
        ccxt_config = {
            "options": {
                "defaultType": "swap",
                "fetchMarkets": {"types": ["linear"]},
                "fetchCurrencies": False,
            },
            "urls": {
                "api": {
                    "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPublicV2": "https://testnet.binancefuture.com/fapi/v2",
                    "fapiPublicV3": "https://testnet.binancefuture.com/fapi/v3",
                    "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPrivateV2": "https://testnet.binancefuture.com/fapi/v2",
                    "fapiPrivateV3": "https://testnet.binancefuture.com/fapi/v3",
                    "fapiData": "https://testnet.binancefuture.com/futures/data",
                }
            }
        }
    return {
        "$schema": "https://schema.freqtrade.io/schema.json",
        "bot_name": bot_name,
        "dry_run": True,
        "dry_run_wallet": 10_000,
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "liquidation_buffer": 0.05,
        "futures_funding_rate": 0.0,
        "stake_currency": "USDT",
        "stake_amount": "unlimited",
        "tradable_balance_ratio": 0.99,
        "max_open_trades": max_open_trades,
        "timeframe": timeframe,
        "dataformat_ohlcv": "feather",
        "cancel_open_orders_on_exit": True,
        "unfilledtimeout": {
            "entry": 10,
            "exit": 10,
            "exit_timeout_count": 0,
            "unit": "minutes",
        },
        "entry_pricing": {
            "price_side": "other",
            "use_order_book": True,
            "order_book_top": 1,
            "price_last_balance": 0.0,
            "check_depth_of_market": {"enabled": False, "bids_to_ask_delta": 1},
        },
        "exit_pricing": {
            "price_side": "other",
            "use_order_book": True,
            "order_book_top": 1,
        },
        "exchange": {
            "name": "binance",
            "key": "",
            "secret": "",
            "ccxt_config": ccxt_config,
            "ccxt_async_config": ccxt_config,
            "enable_ws": False,
            "pair_whitelist": pairs,
            "pair_blacklist": [],
        },
        "pairlists": [{"method": "StaticPairList", "allow_inactive": use_testnet_market_metadata}],
        "internals": {"process_throttle_secs": 5},
    }


def _format_datetime(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _pair_to_filename(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")
