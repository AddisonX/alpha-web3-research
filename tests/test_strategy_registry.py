from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from alpha_web3_research.config import Settings, load_settings
from alpha_web3_research.strategies.hourly_multihorizon import (
    HourlyMultiHorizonAbsoluteScoreLongOnlyStrategy,
)
from alpha_web3_research.strategies.legacy_cross_sectional import DailyCrossSectionalQuantileStrategy
from alpha_web3_research.strategies.registry import StrategyRegistry


def test_registry_preserves_legacy_and_current_strategies() -> None:
    registry = StrategyRegistry(load_settings())
    keys = {definition.strategy_key for definition in registry.definitions()}

    assert keys == {
        "cross_sectional_4h_h6_daily_long_short@v1",
        "cross_sectional_4h_h6_daily_long_only@v1",
        "hourly_multihorizon_absolute_score_long_only@v1",
        "pit_rank64_daily_excess_long_only@v1",
        "pit_rank64_xgboost_daily_excess_long_only@v1",
    }
    assert registry.get("hourly_multihorizon_absolute_score_long_only").status == "research_active"


def test_registry_discovers_existing_historical_runs(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    shutil.copytree(source_root / "strategies", tmp_path / "strategies")
    fixtures = {
        "data/models/walk_forward/walk_forward_4h_h6_forward_return_example/config.json": {
            "experiment_id": "walk_forward_4h_h6_example",
            "timeframe": "4h", "horizon_bars": 6, "label_column": "forward_return",
        },
        "freqtrade/user_data/backtest_results/example.meta.json": {
            "AlphaCrossSectionalStrategy": {"run_id": "synthetic_long_short"},
            "AlphaCrossSectionalLongOnlyStrategy": {"run_id": "synthetic_long_only"},
        },
        "data/models/hourly_rolling/hourly_rolling_example/manifest.json": {
            "run_id": "hourly_rolling_example",
        },
    }
    for relative_path, metadata in fixtures.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata), encoding="utf-8")
    registry = StrategyRegistry(Settings(project_root=tmp_path))
    runs = registry.discover_runs()

    assert any(run.kind == "model_research" and run.run_id.startswith("walk_forward_4h_h6") for run in runs)
    assert any(run.kind == "freqtrade_backtest" and run.strategy_key.endswith("daily_long_short@v1") for run in runs)
    assert any(run.kind == "freqtrade_backtest" and run.strategy_key.endswith("daily_long_only@v1") for run in runs)
    assert any(run.kind == "hourly_research" and run.run_id.startswith("hourly_rolling_") for run in runs)


def test_legacy_policy_reproduces_exact_tail_count_and_symbol_tie_breaking() -> None:
    strategy = DailyCrossSectionalQuantileStrategy(long_only=False, fraction=0.10)
    symbols = [f"COIN{index:02d}" for index in reversed(range(20))]
    targets = strategy.target_positions(np.ones(20), symbols)

    long_symbols = sorted(symbols[index] for index in np.flatnonzero(targets == 1))
    short_symbols = sorted(symbols[index] for index in np.flatnonzero(targets == -1))
    assert long_symbols == ["COIN00", "COIN01"]
    assert short_symbols == ["COIN18", "COIN19"]


def test_hourly_policy_returns_continuous_capped_weights_without_count_limit() -> None:
    strategy = HourlyMultiHorizonAbsoluteScoreLongOnlyStrategy()
    weights = strategy.target_weights(
        scores=np.full(20, 0.10),
        risks=np.full(20, 0.05),
        current_weights=np.zeros(20),
        entry_threshold=0.03,
        exit_threshold=0.021,
        gross_exposure=0.90,
        max_asset_weight=0.12,
    )

    assert np.count_nonzero(weights) == 20
    assert np.isclose(weights.sum(), 0.90)
    assert weights.max() <= 0.12
