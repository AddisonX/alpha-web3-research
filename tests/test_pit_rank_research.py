from __future__ import annotations

import io
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import polars as pl
import pytest

from alpha_web3_research.backtest.daily_equal_weight import run_daily_equal_weight
from alpha_web3_research.data.binance_archive import BAR_MS, native_to_symbol, parse_archive
from alpha_web3_research.models.pit_rank import fixed_train_valid_test_split, purged_training_data
from alpha_web3_research.models.pit_rank_dataset import DAY_MS, build_daily_excess_labels, build_pit_rank_dataset, rank_cross_section
from alpha_web3_research.config import Settings
from alpha_web3_research.research.pit_rank import PITRankResearchConfig
from alpha_web3_research.strategies.pit_rank_excess import PITRankExcessLongOnlyStrategy
from alpha_web3_research.universe.point_in_time import (
    PointInTimeUniverseRules, audit_universe_survival, build_point_in_time_universe,
)


def test_archive_native_symbols_preserve_crypto_multipliers() -> None:
    assert native_to_symbol("1000SHIBUSDT") == "1000SHIB/USDT:USDT"
    with pytest.raises(ValueError):
        native_to_symbol("BTCUSDT_250627")


@pytest.mark.parametrize("header", [True, False])
def test_archive_parser_uses_real_quote_volume_and_normalizes_timestamps(header: bool) -> None:
    target = io.BytesIO()
    with ZipFile(target, "w") as archive:
        archive.writestr("bars.csv", ("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n" if header else "")
                         + "1735689600000000,100,110,90,105,10,1735703999999999,1001,1,1,1,0\n")
    frame = parse_archive(target.getvalue(), "BTCUSDT", "test.zip")
    assert frame["timestamp"].item() == 1735689600000
    assert frame["quote_volume"].item() == 1001
    assert frame["close"].item() == 105


def _universe_bars() -> pl.DataFrame:
    rows = []
    for symbol in ("BTC/USDT:USDT", "1000SHIB/USDT:USDT"):
        for index in range(180 * 6):
            rows.append({"symbol": symbol, "timestamp": index * BAR_MS, "quote_volume": 1_000_000.0})
    return pl.DataFrame(rows)


def test_inception_selection_ignores_future_volume_and_survival() -> None:
    historical = _universe_bars()
    cutoff = 180 * DAY_MS
    rules = PointInTimeUniverseRules(180, 30, 5_000_000, 0.98, 120)
    universe = build_point_in_time_universe(historical, cutoff, rules)
    future = pl.DataFrame({"symbol": ["NEW/USDT:USDT", "BTC/USDT:USDT"],
                           "timestamp": [cutoff, cutoff], "quote_volume": [1e15, 1e15]})
    after = build_point_in_time_universe(pl.concat([historical, future]), cutoff, rules)
    assert universe.equals(after)
    assert universe.filter(pl.col("included")).height == 2
    audit = audit_universe_survival(universe, pl.concat([historical, future]), cutoff + BAR_MS)
    assert audit.filter(~pl.col("data_reaches_audit_end"))["symbol"].to_list() == ["1000SHIB/USDT:USDT"]
    assert audit.height == 2


def test_rank_features_are_cross_sectional_and_ties_use_average_rank() -> None:
    raw = pl.DataFrame({"decision_timestamp": [1, 1, 1, 2, 2, 2],
                        "symbol": ["A", "B", "C", "A", "B", "C"],
                        "factor": [10.0, 10.0, 20.0, 2000.0, 3000.0, 1000.0]})
    result = rank_cross_section(raw, ["factor"], 3)
    assert result.filter(pl.col("decision_timestamp") == 1)["factor"].to_list() == [0.25, 0.25, 1.0]
    changed_future = raw.with_columns(pl.when(pl.col("decision_timestamp") == 2)
                                     .then(-pl.col("factor")).otherwise(pl.col("factor")).alias("factor"))
    assert result.filter(pl.col("decision_timestamp") == 1).equals(
        rank_cross_section(changed_future, ["factor"], 3).filter(pl.col("decision_timestamp") == 1))


def test_exact_daily_labels_center_to_zero_and_do_not_bridge_gaps() -> None:
    rows = []
    for symbol, price in [("A", 100.0), ("B", 200.0)]:
        for index in range(13):
            rows.append({"symbol": symbol, "timestamp": index * BAR_MS,
                         "open": price * (1 + index * (0.01 if symbol == "A" else -0.01))})
    bars = pl.DataFrame(rows)
    labels = build_daily_excess_labels(bars, 0)
    first = labels.filter(pl.col("decision_timestamp") == 0).sort("symbol")
    assert np.allclose(first["forward_return"].to_numpy(), [0.06, -0.06])
    assert np.allclose(first["excess_forward_return"].to_numpy(), [0.06, -0.06])
    missing = bars.filter(~((pl.col("symbol") == "A") & (pl.col("timestamp") == BAR_MS)))
    gap_labels = build_daily_excess_labels(missing, 0)
    assert gap_labels.filter((pl.col("symbol") == "A") & (pl.col("decision_timestamp") == 0)).is_empty()


def test_training_label_purge_never_reaches_holdout() -> None:
    dataset = pl.DataFrame({"decision_timestamp": [10, 20, 30], "label_end_timestamp": [19, 30, 40],
                            "excess_forward_return": [0.1, 0.2, 0.3]})
    assert purged_training_data(dataset, 0, 30)["decision_timestamp"].to_list() == [10]


def test_fixed_split_separates_validation_tail_and_keeps_unlabelled_test_rows() -> None:
    config = PITRankResearchConfig()
    stamp = lambda year, month, day: int(datetime(year, month, day, 4, tzinfo=timezone.utc).timestamp() * 1000)
    times = [stamp(2025, 1, 1), stamp(2025, 4, 1), stamp(2026, 3, 30), stamp(2026, 3, 31),
             stamp(2026, 4, 1), stamp(2026, 4, 29), stamp(2026, 4, 30),
             stamp(2026, 5, 1), stamp(2026, 10, 21)]
    dataset = pl.DataFrame({
        "symbol": ["A"] * 9, "decision_timestamp": times,
        "label_end_timestamp": [value + DAY_MS for value in times[:8]] + [None],
        "excess_forward_return": [0.1] * 8 + [None],
    })
    train, valid, test, meta = fixed_train_valid_test_split(dataset, config)
    assert train["decision_timestamp"].to_list() == times[:3]
    assert valid["decision_timestamp"].to_list() == times[4:6]
    assert test["decision_timestamp"].to_list() == times[7:]
    assert test["excess_forward_return"].null_count() == 1
    assert meta["training_protocol"] == "fixed_train_valid_test"
    assert meta["train_boundary_label_purged_rows"] == 1
    assert meta["validation_boundary_label_purged_rows"] == 1
    assert meta["fit_count"] == 1 and meta["validation_set_count"] == 1
    assert meta["early_stopping"] is True and meta["parameter_search"] is False


def test_training_hyperparameters_do_not_invalidate_raw_data_contract() -> None:
    config = PITRankResearchConfig()
    changed = config.model_copy(update={"n_estimators": 600, "num_leaves": 31})
    assert config.data_contract() == changed.data_contract()
    assert config.data_contract() != config.model_copy(update={"max_symbols": 80}).data_contract()


def test_fixed_protocol_rejects_rolling_settings() -> None:
    with pytest.raises(ValueError):
        PITRankResearchConfig(train_window_days=365)
    with pytest.raises(ValueError):
        PITRankResearchConfig(training_protocol="rolling_validation")


def test_strategy_exact_top_ten_percent_equal_weights_and_deterministic_ties() -> None:
    strategy = PITRankExcessLongOnlyStrategy()
    symbols = [f"C{index:02d}" for index in reversed(range(21))]
    weights = strategy.select_weights(np.ones(21), symbols)
    assert sorted(symbols[index] for index in np.flatnonzero(weights)) == ["C00", "C01", "C02"]
    assert np.isclose(weights.sum(), 1)
    assert np.allclose(weights[weights > 0], 1 / 3)


def _execution_bars(gap_stop: bool = False, missing: bool = False) -> pl.DataFrame:
    rows = []
    for index in range(12):
        if missing and index == 2:
            continue
        price = 80.0 if gap_stop and index >= 1 else 100.0
        rows.append({"symbol": "A", "timestamp": index * BAR_MS, "open": price,
                     "high": price + 1, "low": 85.0 if index == 1 and not gap_stop else price - 1,
                     "close": price})
    return pl.DataFrame(rows)


def test_stop_loss_uses_low_and_blocks_same_day_reentry() -> None:
    predictions = pl.DataFrame({"symbol": ["A"], "decision_timestamp": [0], "prediction": [1.0],
                                "universe_forward_return": [0.0]})
    result = run_daily_equal_weight(predictions, _execution_bars(), fee_rate=0, slippage_rate=0)
    sells = result.trades.filter(pl.col("side") == "sell")
    assert sells["price"].to_list() == [90.0]
    assert result.metrics["stop_count"] == 1
    assert result.trades.filter(pl.col("side") == "buy").height == 1
    assert result.metrics["total_return"] == pytest.approx(-0.1)


def test_gap_stop_executes_at_worse_open_not_at_stop_price() -> None:
    predictions = pl.DataFrame({"symbol": ["A"], "decision_timestamp": [0], "prediction": [1.0],
                                "universe_forward_return": [0.0]})
    result = run_daily_equal_weight(predictions, _execution_bars(gap_stop=True), fee_rate=0, slippage_rate=0)
    sells = result.trades.filter(pl.col("side") == "sell")
    assert sells["price"].to_list() == [80.0]
    assert sells["reason"].to_list() == ["stop_gap"]


def test_gap_stop_allows_reentry_at_next_scheduled_daily_rebalance() -> None:
    predictions = pl.DataFrame({"symbol": ["A", "A"], "decision_timestamp": [0, DAY_MS],
                                "prediction": [1.0, 1.0], "universe_forward_return": [0.0, 0.0]})
    result = run_daily_equal_weight(predictions, _execution_bars(gap_stop=True), fee_rate=0, slippage_rate=0)
    assert result.trades.filter(pl.col("side") == "buy")["timestamp"].to_list() == [0, DAY_MS]


def test_missing_execution_data_is_explicit_not_silently_zero_return() -> None:
    predictions = pl.DataFrame({"symbol": ["A"], "decision_timestamp": [0], "prediction": [1.0],
                                "universe_forward_return": [0.0]})
    # Avoid the stop candle so the position is still held at the missing bar.
    bars = _execution_bars(missing=True).with_columns(pl.lit(99.0).alias("low"))
    result = run_daily_equal_weight(predictions, bars, fee_rate=0, slippage_rate=0)
    assert result.metrics["execution_data_complete"] is False
    assert "held_position_missing_execution_bar" in result.data_issues["issue"].to_list()


def test_frozen_feature_manifest_is_the_requested_64_set() -> None:
    root = Path(__file__).resolve().parents[1]
    definition = json.loads((root / "strategies/features/pit_rank64.v1.json").read_text())
    features = definition["features"]
    assert len(set(features)) == len(features) == 64
    expected = "53e2e0c9940dfd9721d10dd4b478937905b43a40c827b803d77dd17ca3477cd5"
    assert hashlib.sha256("\n".join(sorted(features)).encode()).hexdigest() == expected


def test_full_factor_pipeline_preserves_unlabelled_predictions_and_causal_features() -> None:
    root = Path(__file__).resolve().parents[1]
    names = json.loads((root / "strategies/features/pit_rank64.v1.json").read_text())["features"]
    rows = []
    for symbol_index, symbol in enumerate(["A/USDT:USDT", "B/USDT:USDT", "C/USDT:USDT"]):
        for index in range(72):
            price = 100 + symbol_index * 10 + index * 0.1 + np.sin(index + symbol_index)
            rows.append({"exchange": "binanceusdm", "market_type": "swap", "symbol": symbol,
                         "timeframe": "4h", "timestamp": index * BAR_MS,
                         "open": price, "high": price + 2, "low": price - 2, "close": price + 0.1,
                         "volume": float(100 + index + symbol_index * 3), "fetched_at": "test", "source": "fixture"})
    bars = pl.DataFrame(rows)
    dataset, _ = build_pit_rank_dataset(bars, names, Settings(), min_cross_section=3)
    assert not dataset.filter(pl.col("excess_forward_return").is_null()).is_empty()
    changed = bars.with_columns([
        pl.when(pl.col("timestamp") >= 60 * BAR_MS).then(pl.col(column) * 5)
        .otherwise(pl.col(column)).alias(column)
        for column in ("open", "high", "low", "close", "volume")
    ])
    altered, _ = build_pit_rank_dataset(changed, names, Settings(), min_cross_section=3)
    columns = ["symbol", "decision_timestamp", *names]
    assert dataset.filter(pl.col("decision_timestamp") < 60 * BAR_MS).select(columns).equals(
        altered.filter(pl.col("decision_timestamp") < 60 * BAR_MS).select(columns))


def test_runner_fits_once_and_early_stops_on_validation_tail_only(tmp_path, monkeypatch) -> None:
    import alpha_web3_research.models.pit_rank as runner_module
    from alpha_web3_research.research.pit_rank import research_root

    repo = Path(__file__).resolve().parents[1]
    feature_text = (repo / "strategies/features/pit_rank64.v1.json").read_text()
    names = json.loads(feature_text)["features"]
    feature_path = tmp_path / "strategies/features/pit_rank64.v1.json"
    feature_path.parent.mkdir(parents=True)
    feature_path.write_text(feature_text)
    settings = Settings(project_root=tmp_path, data_path=Path("data"), reports_path=Path("reports"), _env_file=None)
    config = PITRankResearchConfig(n_estimators=6, early_stopping_rounds=2, min_cross_section=2)
    root = research_root(settings, config)
    (root / "bars").mkdir(parents=True)
    stamp = lambda year, month, day: int(datetime(year, month, day, 4, tzinfo=timezone.utc).timestamp() * 1000)
    may_start = stamp(2026, 5, 1)
    symbols = ["A/USDT:USDT", "B/USDT:USDT"]
    bars = pl.DataFrame([
        {"symbol": symbol, "timestamp": may_start + index * BAR_MS,
         "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        for symbol in symbols for index in range(12)
    ])
    bars.write_parquet(root / "bars/fixture.parquet")
    pl.DataFrame({"symbol": symbols, "included": [True, True]}).write_parquet(root / "universe.parquet")
    pl.DataFrame({"symbol": symbols, "data_reaches_audit_end": [True, True], "liquidity_rank": [1, 2],
                  "median_daily_quote_volume": [1e7, 1e7], "observed_last_open_ms": [may_start + 11 * BAR_MS] * 2,
                  }).write_parquet(root / "survival_audit.parquet")
    (root / "data_manifest.json").write_text(json.dumps({"config": config.model_dump(),
                                                         "data_contract": config.data_contract()}))
    rows = []
    for timestamp, target in [(stamp(2025, 1, 1), 0.01), (stamp(2026, 3, 30), 0.01),
                              (stamp(2026, 4, 1), 0.02), (stamp(2026, 4, 29), 0.02),
                              (may_start, 0.03), (may_start + DAY_MS, 0.03), (may_start + 2 * DAY_MS, None)]:
        for index, symbol in enumerate(symbols):
            rows.append({"symbol": symbol, "decision_timestamp": timestamp,
                         "label_end_timestamp": timestamp + DAY_MS if target is not None else None,
                         "excess_forward_return": target * (1 if index == 0 else -1) if target is not None else None,
                         "universe_forward_return": 0.0 if target is not None else None,
                         **{name: float(index) for name in names}})
    dataset = pl.DataFrame(rows)
    monkeypatch.setattr(runner_module, "build_pit_rank_dataset", lambda *args: (dataset, dataset))
    original_regressor = runner_module.lgb.LGBMRegressor
    fit_calls = []

    def recording_regressor(**params):
        model = original_regressor(**params)
        original_fit = model.fit

        def fit(*args, **kwargs):
            fit_calls.append((args[1].copy(), kwargs))
            return original_fit(*args, **kwargs)

        model.fit = fit
        return model

    monkeypatch.setattr(runner_module.lgb, "LGBMRegressor", recording_regressor)
    run = runner_module.PITRankResearchRunner(settings, config).run(lambda message: None)
    assert len(fit_calls) == 1
    train_y, fit_kwargs = fit_calls[0]
    assert set(abs(train_y)) == {0.01}
    assert len(fit_kwargs["eval_set"]) == 1
    assert set(abs(fit_kwargs["eval_set"][0][1])) == {0.02}
    assert fit_kwargs["callbacks"]
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["training_protocol"] == "fixed_train_valid_test"
    assert 1 <= manifest["best_iteration"] <= 6
    assert manifest["final_tree_count"] <= manifest["best_iteration"]
    predictions = pl.read_parquet(run / "predictions.parquet")
    assert predictions["excess_forward_return"].null_count() == 2
    assert (run / "validation_predictions.parquet").exists()
    assert not (run / "rolling_validation.parquet").exists()
    frozen = runner_module.lgb.Booster(model_file=str(run / "model.txt"))
    assert np.allclose(frozen.predict(predictions.select(names).to_numpy()), predictions["prediction"].to_numpy())
