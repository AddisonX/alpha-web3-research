import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from src.config import Settings
from src.live.market import MarketDataProvider, PreparedMarket
from src.live.safety import check_live_signal_status, validate_signal_rows
from src.live.signal_store import HOUR_MS, SignalStore
from src.live.signals import LiveLongOnlySignalProducer
from src.strategies.base import (
    DEFAULT_STRATEGY_ID, DataRequirements, StrategyContext, StrategyDecision, StrategyMetadata,
)
from src.strategies.lightgbm import LightGBMLongOnlyStrategy

STAMP = int(datetime(2026, 7, 27, tzinfo=timezone.utc).timestamp() * 1000)


class RuleStrategy:
    strategy_id = "test-reversal-v1"
    requirements = DataRequirements(("return_1",))
    metadata = StrategyMetadata("test-rule-no-model")

    def decide(self, context):
        return StrategyDecision(context.decision_timestamp, context.features.select(
            pl.col("symbol").alias("pair"), (pl.col("return_1") < 0).cast(pl.Int8).alias("target_position"),
        ))


class FakeProvider:
    calls = 0

    def prepare(self, pairs, stamp, requirements):
        self.calls += 1
        assert requirements.features == ("return_1",)
        return PreparedMarket(pl.DataFrame({"symbol": pairs, "return_1": [-0.1] + [0.1] * (len(pairs) - 1)}), len(pairs), {})


def producer(tmp_path, strategy=None, provider=None):
    pairs = [f"COIN{i:02}/USDT:USDT" for i in range(60)]
    (tmp_path / "pairs.json").write_text(json.dumps(pairs))
    exchange = SimpleNamespace(parse_timeframe=lambda _: 14400, urls={"api": {}}, milliseconds=lambda: STAMP + 5 * HOUR_MS)
    return LiveLongOnlySignalProducer(
        Settings(project_root=tmp_path), exchange, tmp_path / "pairs.json", tmp_path / "targets.parquet",
        tmp_path / "status.json", strategy=strategy or RuleStrategy(), data_provider=provider or FakeProvider(),
    )


def test_rule_strategy_uses_same_pipeline_without_models(tmp_path):
    provider = FakeProvider()
    runtime = producer(tmp_path, provider=provider)
    result = runtime.run()
    assert result.selected_pairs == ["COIN00/USDT:USDT"]
    assert result.cross_section_count == 60
    assert not result.skipped
    status = json.loads(runtime.status_path.read_text())
    assert status["model_count"] == 0 and status["strategy_id"] == RuleStrategy.strategy_id
    # Check against the actual generated-at timestamp with a deterministic in-window status.
    status["generated_at"] = "2026-07-27T04:05:00+00:00"
    health = check_live_signal_status(status, datetime(2026, 7, 27, 6, tzinfo=timezone.utc))
    assert health.ready
    import pandas as pd
    assert validate_signal_rows(pd.read_parquet(runtime.signal_path), health) is None
    cached = runtime.run()
    assert cached.skipped and provider.calls == 1
    runtime.run(force=True)
    signals = pl.read_parquet(runtime.signal_path)
    assert signals.height == 60 * 7
    assert signals.select(pl.struct("date", "pair").n_unique()).item() == signals.height


@pytest.mark.parametrize("mode", ["raise", "empty", "short", "partial", "duplicate", "timestamp", "cash"])
def test_bad_decision_freezes_without_overwriting_targets(tmp_path, mode):
    runtime = producer(tmp_path)
    runtime.run()
    before = runtime.signal_path.read_bytes()

    class Broken(RuleStrategy):
        def decide(self, context):
            if mode == "raise":
                raise RuntimeError("strategy failed")
            decision = super().decide(context)
            frame = decision.targets
            if mode == "empty": frame = frame.head(0)
            if mode == "short": frame = frame.with_columns(pl.lit(-1).alias("target_position"))
            if mode == "partial": frame = frame.head(1)
            if mode == "duplicate": frame = pl.concat([frame, frame.head(1)])
            if mode == "cash": frame = frame.with_columns(pl.lit(0).alias("target_position"))
            return StrategyDecision(context.decision_timestamp + (1 if mode == "timestamp" else 0), frame)

    runtime.strategy = Broken()
    with pytest.raises((ValueError, RuntimeError)):
        runtime.run(force=True)
    assert runtime.signal_path.read_bytes() == before
    assert json.loads(runtime.status_path.read_text())["state"] == "failed"


def test_directory_owner_mismatch_does_not_change_any_files(tmp_path):
    runtime = producer(tmp_path)
    runtime.run()
    before = runtime.status_path.read_bytes()
    runtime.strategy_id = "another-strategy"
    with pytest.raises(ValueError, match="different strategy"):
        runtime.run(force=True)
    assert runtime.status_path.read_bytes() == before


def test_alternative_strategy_cannot_use_production_paths(tmp_path):
    runtime = producer(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        LiveLongOnlySignalProducer(
            runtime.settings, runtime.exchange, runtime.pairs_path,
            tmp_path / "freqtrade/instances/tokyo-lightgbm/signals/alpha_live_long_only_signals.parquet",
            runtime.status_path, strategy=RuleStrategy(),
        )


def test_lightgbm_policy_is_pure_with_model_diagnostics(monkeypatch):
    import src.strategies.lightgbm as module
    model = SimpleNamespace(features=["return_1"], timeframe="4h", source_experiment="fixture", boosters=[1]*5,
                            horizon_bars=6, label_column="forward_return", top_quantile_fraction=.1,
                            predict=lambda f: np.asarray(f["return_1"]))
    monkeypatch.setattr(module, "ProductionEnsemble", lambda _: model)
    strategy = LightGBMLongOnlyStrategy(Path("not-read"))
    frame = pl.DataFrame({"symbol": [f"S{i:02}" for i in range(60)], "return_1": np.arange(60, dtype=float)})
    context = StrategyContext(STAMP, frame)
    decision = strategy.decide(context)
    decision.validate(context)
    assert decision.targets.filter(pl.col("target_position") == 1)["pair"].to_list() == [f"S{i:02}" for i in range(59,53,-1)]


def test_market_provider_obeys_data_requirements(monkeypatch):
    import src.live.market as module
    calls = []
    def fetch(self, **kwargs):
        calls.append(kwargs)
        if kwargs["symbol"] == "BAD": raise RuntimeError("no candles")
        return pl.DataFrame({"exchange": ["binanceusdm"]*40, "market_type": ["swap"]*40,
            "symbol": [kwargs["symbol"]]*40, "timeframe": ["4h"]*40,
            "timestamp": [STAMP - (39-i)*4*HOUR_MS for i in range(40)],
            "open": [100.+i for i in range(40)], "close": [101.+i for i in range(40)],
            "high": [102.+i for i in range(40)], "low": [99.+i for i in range(40)], "volume": [100.]*40})
    monkeypatch.setattr(module.OHLCVFetcher, "fetch_ohlcv_frame", fetch)
    exchange = SimpleNamespace(parse_timeframe=lambda _: 14400)
    prepared = MarketDataProvider(Settings(), exchange).prepare(["GOOD", "BAD"], STAMP, DataRequirements(("return_1",)))
    assert prepared.features.columns == ["symbol", "return_1"]
    assert prepared.features.height == 1 and "BAD" in prepared.errors
    assert prepared.requested_pair_count == 2
    assert calls[0]["include_unclosed"] is False
    assert calls[0]["until"] == "2026-07-27T04:00:00+00:00"


def test_provider_failure_and_small_cross_section_freeze(tmp_path):
    runtime = producer(tmp_path)
    runtime.run()
    before = runtime.signal_path.read_bytes()
    class SmallProvider(FakeProvider):
        def prepare(self, pairs, stamp, requirements):
            return super().prepare(pairs[:2], stamp, requirements)
    runtime.data_provider = SmallProvider()
    with pytest.raises(RuntimeError, match="cross section"):
        runtime.run(force=True)
    assert runtime.signal_path.read_bytes() == before
    assert json.loads(runtime.status_path.read_text())["state"] == "failed"


def test_cli_chooses_registered_id_and_isolates_paths(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    import src.cli as cli
    captured = {}
    monkeypatch.setattr(cli, "_settings", lambda _: Settings(project_root=tmp_path))
    monkeypatch.setattr(cli, "create_exchange", lambda *args, **kwargs: object())
    class FakeProducer:
        def __init__(self, **kwargs): captured.update(kwargs)
        def run(self, **kwargs):
            return SimpleNamespace(skipped=False, cross_section_count=60, selected_count=1,
                selected_pairs=["A"], signal_path=captured["signal_path"], status_path=captured["status_path"])
    monkeypatch.setattr(cli, "LiveLongOnlySignalProducer", FakeProducer)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["live", "generate-signals", "--strategy", "test-rule-v1"]).exit_code == 0
    assert captured["strategy_id"] == "test-rule-v1"
    assert captured["signal_path"].parent.parent.name == "test-rule-v1"
    assert captured["signal_path"].parent.name == "signals"
    assert runner.invoke(cli.app, ["live", "generate-signals", "--strategy", "../../bad"]).exit_code != 0


def test_existing_legacy_status_remains_a_default_strategy_cache(tmp_path):
    rule = RuleStrategy()
    rule.strategy_id = DEFAULT_STRATEGY_ID
    runtime = producer(tmp_path, strategy=rule)
    runtime.run()
    status = json.loads(runtime.status_path.read_text())
    del status["strategy_id"]
    runtime.status_path.write_text(json.dumps(status))
    assert runtime.run().skipped


def test_invalid_metadata_publishes_failed_status(tmp_path):
    runtime = producer(tmp_path)
    runtime.run()
    before = runtime.signal_path.read_bytes()
    runtime.strategy.metadata = None
    with pytest.raises(ValueError, match="StrategyMetadata"):
        runtime.run(force=True)
    assert runtime.signal_path.read_bytes() == before
    assert json.loads(runtime.status_path.read_text())["state"] == "failed"


def test_missing_production_model_publishes_failed_status(tmp_path):
    pairs = tmp_path / "pairs.json"
    pairs.write_text('["A"]')
    runtime = LiveLongOnlySignalProducer(
        Settings(project_root=tmp_path), SimpleNamespace(), pairs,
        tmp_path / "targets.parquet", tmp_path / "status.json",
    )
    with pytest.raises(FileNotFoundError):
        runtime.run()
    assert json.loads(runtime.status_path.read_text())["state"] == "failed"
    assert not runtime.signal_path.exists()
