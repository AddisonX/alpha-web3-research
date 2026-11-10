"""Offline acceptance tests; adapt policy-specific assertions for a new strategy."""

import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from src.config import Settings
from src.live.market import PreparedMarket
from src.live.safety import check_live_signal_status, validate_signal_rows
from src.live.signal_store import HOUR_MS
from src.live.signals import LiveLongOnlySignalProducer
from src.strategies.base import StrategyContext

BUNDLE = Path(__file__).resolve().parent
STAMP = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

# Load only this explicitly chosen local fixture, never an unreviewed package path.
spec = importlib.util.spec_from_file_location("alpha_strategy_import_template", BUNDLE / "strategy.py")
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Strategy = module.ImportedLongOnlyStrategy


def feature_frame(count=60):
    return pl.DataFrame({
        "symbol": [f"COIN{i:03}/USDT:USDT" for i in range(count)],
        "return_1": [i / 10000 for i in range(count)],
    })


def selected_pairs(decision):
    return decision.targets.filter(pl.col("target_position") == 1)["pair"].to_list()


def test_manifest_matches_code():
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    strategy = Strategy()
    assert manifest["schema_version"] == 1
    assert manifest["strategy_id"] == strategy.strategy_id
    assert manifest["entrypoint"] == {"file": "strategy.py", "class": "ImportedLongOnlyStrategy"}
    assert manifest["requirements"] == json.loads(json.dumps(asdict(strategy.requirements)))
    assert manifest["metadata"] == asdict(strategy.metadata)
    assert manifest["parameters"]["top_fraction"] == strategy.top_fraction
    assert manifest["parameters"]["max_selected"] == strategy.max_selected
    assert manifest["preprocessing"]["feature_order"] == list(strategy.requirements.features)
    assert manifest["model"]["files"] == []  # Replace with real model/hash tests for ML policies.


def test_complete_binary_targets_and_example_policy():
    context = StrategyContext(STAMP, feature_frame())
    decision = Strategy().decide(context)
    decision.validate(context)
    assert decision.decision_timestamp == STAMP
    assert decision.targets.height == 60
    assert set(decision.targets["target_position"].to_list()) == {0, 1}
    assert selected_pairs(decision) == [f"COIN{i:03}/USDT:USDT" for i in range(59, 53, -1)]


def test_deterministic_permutation_and_input_unchanged():
    strategy = Strategy()
    frame = feature_frame()
    before = frame.clone()
    first = strategy.decide(StrategyContext(STAMP, frame)).targets
    second = strategy.decide(StrategyContext(STAMP, frame.reverse())).targets
    assert first.equals(second)
    assert frame.equals(before)


def test_tie_breaks_by_symbol_and_respects_execution_capacity():
    strategy = Strategy()
    ties = pl.DataFrame({"symbol": ["C", "B", "A"], "return_1": [.1, .1, .1]})
    assert selected_pairs(strategy.decide(StrategyContext(STAMP, ties))) == ["A"]
    assert len(selected_pairs(strategy.decide(StrategyContext(STAMP, feature_frame(120))))) == 9


def test_nonfinite_scores_are_never_selected_but_keep_zero_targets():
    frame = pl.DataFrame({
        "symbol": ["A", "B", "C", "D", "E"],
        "return_1": [None, float("nan"), float("inf"), -float("inf"), -.1],
    })
    context = StrategyContext(STAMP, frame)
    decision = Strategy().decide(context)
    decision.validate(context)
    assert selected_pairs(decision) == ["E"]
    assert decision.targets.height == frame.height


@pytest.mark.parametrize("frame,message", [
    (feature_frame(0), "empty"),
    (feature_frame().drop("return_1"), "missing"),
    (pl.DataFrame({"symbol": ["A", "A"], "return_1": [.1, .2]}), "unique"),
    (pl.DataFrame({"symbol": [None], "return_1": [.1]}), "unique"),
    (pl.DataFrame({"symbol": [""], "return_1": [.1]}), "unique"),
    (pl.DataFrame({"symbol": ["A"], "return_1": ["bad"]}), "numeric"),
    (pl.DataFrame({"symbol": ["A"], "return_1": [None]}), "no finite"),
    (pl.DataFrame({"symbol": ["A"], "return_1": [float("nan")]}), "no finite"),
])
def test_invalid_input_fails_explicitly(frame, message):
    with pytest.raises(ValueError, match=message):
        Strategy().decide(StrategyContext(STAMP, frame))


class OfflineMarket:
    def __init__(self):
        self.fail_scores = False

    def prepare(self, pairs, stamp, requirements):
        frame = feature_frame(len(pairs)).with_columns(pl.Series("symbol", pairs))
        if self.fail_scores:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("return_1"))
        return PreparedMarket(frame, len(pairs), {})


def test_offline_publication_and_failure_preserves_previous_signals(tmp_path):
    pairs = feature_frame()["symbol"].to_list()
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps(pairs))
    provider = OfflineMarket()
    runtime = LiveLongOnlySignalProducer(
        settings=Settings(project_root=tmp_path),
        exchange=SimpleNamespace(
            parse_timeframe=lambda _: 14400,
            milliseconds=lambda: STAMP + 5 * HOUR_MS,
            urls={"api": {}},
        ),
        pairs_path=pairs_path,
        signal_path=tmp_path / "signals/alpha_live_long_only_signals.parquet",
        status_path=tmp_path / "signals/alpha_live_long_only_status.json",
        strategy=Strategy(),
        data_provider=provider,
    )
    result = runtime.run()
    assert result.cross_section_count == 60
    assert result.selected_count == 6
    assert runtime.run().skipped
    status = json.loads(runtime.status_path.read_text())
    assert status["strategy_id"] == Strategy.strategy_id
    assert status["selected_pairs"] == result.selected_pairs
    # Control the clock only in the test; publication uses the real current time.
    status["generated_at"] = "2026-01-01T04:05:00+00:00"
    health = check_live_signal_status(status, datetime(2026, 1, 1, 5, tzinfo=timezone.utc))
    assert health.ready
    import pandas as pd

    assert validate_signal_rows(pd.read_parquet(runtime.signal_path), health) is None
    before = runtime.signal_path.read_bytes()
    provider.fail_scores = True
    with pytest.raises(ValueError, match="no finite"):
        runtime.run(force=True)
    assert runtime.signal_path.read_bytes() == before
    assert json.loads(runtime.status_path.read_text())["state"] == "failed"
