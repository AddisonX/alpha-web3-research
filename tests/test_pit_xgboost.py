from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from alpha_web3_research.backtest.daily_equal_weight import run_daily_equal_weight
from alpha_web3_research.config import Settings
from alpha_web3_research.data.binance_archive import BAR_MS
from alpha_web3_research.data.time import parse_timestamp_ms
from alpha_web3_research.models.pit_xgboost import PITXGBoostConfig, fixed_year_split
from alpha_web3_research.strategies.pit_rank_xgboost import PITRankXGBoostLongOnlyStrategy

DAY_MS = 86_400_000


@pytest.mark.parametrize("gap", [False, True])
def test_no_stop_holds_through_intrabar_and_gap_losses_until_rebalance(gap):
    bars = pl.DataFrame([
        {"symbol": symbol, "timestamp": i * BAR_MS, "open": 80.0 if gap and i > 0 else 100.0,
         "high": 101.0, "low": 50.0, "close": 80.0 if gap and i > 0 else 100.0}
        for symbol in ["A", "B"] for i in range(12)
    ])
    predictions = pl.DataFrame({
        "symbol": ["A", "B"], "decision_timestamp": [0, DAY_MS],
        "prediction": [1.0, 1.0], "universe_forward_return": [-0.2 if gap else 0.0, 0.0],
    })
    result = run_daily_equal_weight(predictions, bars, PITRankXGBoostLongOnlyStrategy(),
                                    fee_rate=0, slippage_rate=0)
    assert result.metrics["stop_loss"] is None
    assert result.metrics["stop_count"] == 0
    assert result.metrics["total_return"] == pytest.approx(-0.2 if gap else 0.0)
    sells = result.trades.filter(pl.col("side") == "sell")
    assert sells["timestamp"].to_list() == [DAY_MS]
    assert sells["symbol"].to_list() == ["A"]
    assert sells["reason"].to_list() == ["rebalance"]


def test_year_split_purges_cross_year_targets_and_preserves_unlabelled_predictions():
    boundary = parse_timestamp_ms("2026-01-01T00:00:00Z")
    data = pl.DataFrame({
        "decision_timestamp": [boundary - 2 * DAY_MS, boundary - DAY_MS, boundary, boundary + DAY_MS],
        "label_end_timestamp": [boundary - DAY_MS, boundary, boundary + DAY_MS, None],
        "excess_forward_return": [0.1, 0.2, 0.3, None],
    })
    train, test, meta = fixed_year_split(data, PITXGBoostConfig())
    assert train.height == 1 and test.height == 2
    assert test["excess_forward_return"].null_count() == 1
    assert meta["boundary_label_purged_rows"] == 1
    assert meta["early_stopping"] is False and meta["validation_set_count"] == 0
    for params in [{"n_estimators": 299}, {"stop_loss": 0.1}, {"early_stopping_rounds": 40}]:
        with pytest.raises(ValueError):
            PITXGBoostConfig(**params)


def test_runner_fits_only_2025_and_saves_exactly_300_trees(tmp_path, monkeypatch):
    import alpha_web3_research.models.pit_xgboost as module

    repo = Path(__file__).resolve().parents[1]
    feature_text = (repo / "strategies/features/pit_rank64.v1.json").read_text()
    names = json.loads(feature_text)["features"]
    feature_file = tmp_path / "strategies/features/pit_rank64.v1.json"
    feature_file.parent.mkdir(parents=True)
    feature_file.write_text(feature_text)
    settings = Settings(project_root=tmp_path, data_path=Path("data"), reports_path=Path("reports"), _env_file=None)
    config = PITXGBoostConfig(min_cross_section=2, min_child_weight=1)
    source = tmp_path / "data/research" / config.source_dataset_namespace
    (source / "bars").mkdir(parents=True)
    boundary = parse_timestamp_ms(config.test_start)
    symbols = ["A", "B"]
    bars = pl.DataFrame([
        {"symbol": symbol, "timestamp": boundary + i * BAR_MS,
         "open": 100.0, "high": 101.0, "low": 50.0, "close": 100.0}
        for symbol in symbols for i in range(12)
    ])
    bars.write_parquet(source / "bars/fixture.parquet")
    pl.DataFrame({"symbol": symbols, "included": [True, True]}).write_parquet(source / "universe.parquet")
    (source / "data_manifest.json").write_text(json.dumps({
        "inception_as_of_ms": parse_timestamp_ms(config.train_start), "future_survival_used_for_selection": False,
    }))
    rows = []
    for timestamp, target in [(boundary - 3 * DAY_MS, 0.01), (boundary - DAY_MS, 0.02),
                              (boundary, 0.03), (boundary + DAY_MS, 0.03), (boundary + 2 * DAY_MS, None)]:
        for i, symbol in enumerate(symbols):
            rows.append({"symbol": symbol, "decision_timestamp": timestamp,
                         "label_end_timestamp": timestamp + DAY_MS if target is not None else None,
                         "excess_forward_return": target * (1 if i == 0 else -1) if target is not None else None,
                         "universe_forward_return": 0.0 if target is not None else None,
                         **{name: float(i) for name in names}})
    dataset = pl.DataFrame(rows)
    monkeypatch.setattr(module, "build_pit_rank_dataset", lambda *args: (dataset, dataset))
    original = module.xgb.XGBRegressor
    calls = []

    def recording_regressor(**params):
        model = original(**params)
        original_fit = model.fit

        def fit(*args, **kwargs):
            calls.append((params, args[1].copy(), kwargs))
            return original_fit(*args, **kwargs)

        model.fit = fit
        return model

    monkeypatch.setattr(module.xgb, "XGBRegressor", recording_regressor)
    run = module.PITXGBoostResearchRunner(settings, config).run(lambda message: None)
    assert len(calls) == 1
    params, labels, kwargs = calls[0]
    assert set(abs(labels)) == {0.01}
    assert params["n_estimators"] == 300
    assert "early_stopping_rounds" not in params and "eval_set" not in kwargs
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["final_tree_count"] == manifest["boosting_rounds"] == 300
    assert manifest["test_portfolio"]["stop_count"] == 0
    assert manifest["test_portfolio"]["stop_loss"] is None
    frozen = original()
    frozen.load_model(run / "model.ubj")
    predictions = pl.read_parquet(run / "predictions.parquet")
    assert predictions["excess_forward_return"].null_count() == 2
    assert np.array_equal(frozen.predict(predictions.select(names).to_pandas()), predictions["prediction"].to_numpy())
