import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.strategies.base import StrategyContext, StrategyDecision, validate_signed_weights
from src.strategies.xgb15 import FEATURES, STRATEGY_ID, XGB15LongShortStrategy, buffered_targets, percentile_features
from src.live.signal_store import SignalStore
from src.live.safety import check_live_signal_status, validate_signal_rows
from src.live.dashboard import position_snapshot
from src.factors.xgb15 import build_xgb15_features

STAMP = int(pd.Timestamp("2026-10-26T12:00:00Z").timestamp()*1000)
PAIRS = [f"C{i:02}/USDT:USDT" for i in range(60)]


@pytest.fixture
def model(tmp_path, monkeypatch):
    file = tmp_path / "model.ubj"
    file.write_bytes(b"test loader fixture")
    manifest = {"features": list(FEATURES), "preprocessing": "percentile_average", "models": {
        "2026": {"file": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                 "fit_cutoff_utc": "2026-01-01T00:00:00Z", "pairs": PAIRS}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    class Booster:
        feature_names = list(FEATURES)
        def load_model(self, path): pass
        def num_boosted_rounds(self): return 120
        def set_param(self, params): pass
        def predict(self, matrix): return matrix.get_data().toarray()[:, 0]
    monkeypatch.setattr("src.strategies.xgb15.xgb.Booster", Booster)
    return XGB15LongShortStrategy(tmp_path)


def context():
    return StrategyContext(STAMP, pl.DataFrame({"symbol": PAIRS, **{f: np.arange(1,61)/60 for f in FEATURES}}), True)


def test_complete_deterministic_signed_decision(model):
    c = context()
    first = model.decide(c)
    second = model.decide(StrategyContext(STAMP, c.features.reverse().select(reversed(c.features.columns)), True))
    assert first.targets.equals(second.targets)
    assert first.targets.filter(pl.col("target_position") == 1).height == 6
    assert first.targets.filter(pl.col("target_position") == -1).height == 6
    validate_signed_weights(first.targets)
    with pytest.raises(ValueError, match="explicitly enabled"):
        first.validate(StrategyContext(STAMP, c.features))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -.1, 1.01, 0])
def test_invalid_percentiles_fail(model, bad):
    c = context()
    with pytest.raises(ValueError, match="percentile"):
        model.decide(StrategyContext(STAMP, c.features.with_columns(pl.lit(bad).alias(FEATURES[0])), True))


def test_model_contract_hash_missing_and_duplicate(model):
    c = context()
    with pytest.raises(ValueError, match="duplicate"):
        model.decide(StrategyContext(STAMP, pl.concat([c.features, c.features.head(1)]), True))
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        model.decide(StrategyContext(STAMP, c.features.drop(FEATURES[0]), True))
    (model.model_dir / "model.ubj").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        model.decide(c)
    (model.model_dir / "model.ubj").unlink()
    with pytest.raises(FileNotFoundError):
        model.decide(c)
    model.manifest["features"] = list(reversed(FEATURES))
    (model.model_dir / "manifest.json").write_text(json.dumps(model.manifest))
    with pytest.raises(ValueError, match="contract"):
        XGB15LongShortStrategy(model.model_dir)


def test_buffer_retains_inside_30_percent_and_equal_weights():
    scores = np.arange(60, dtype=float)
    first = buffered_targets(PAIRS, scores, None)
    scores[59] = 44.5  # Long falls out of top 10%, remains within top 30%.
    second = buffered_targets(PAIRS, scores, first)
    assert second.filter(pl.col("pair") == PAIRS[59])["target_position"][0] == 1
    scores[59] = 30.5
    third = buffered_targets(PAIRS, scores, second)
    assert third.filter(pl.col("pair") == PAIRS[59])["target_position"][0] == 0
    validate_signed_weights(third)


def test_rank_denominators_before_complete_row_filter():
    raw = pd.DataFrame({"symbol": ["A", "B", "C"], **{f: [1., 2., 3.] for f in FEATURES}})
    raw.loc[2, FEATURES[1]] = np.nan
    ranked = percentile_features(raw)
    assert ranked.loc[1, FEATURES[0]] == pytest.approx(2/3)
    assert ranked.loc[1, FEATURES[1]] == 1.


def test_signed_publication_and_safety(model, tmp_path):
    decision = model.decide(context())
    store = SignalStore(tmp_path / "signals.parquet", tmp_path / "status.json")
    store.publish(STRATEGY_ID, decision, model.requirements, model.metadata, "test", 60, {}, 24)
    status = store.read_status()
    status["generated_at"] = "2026-10-26T16:01:00+00:00"
    now = datetime(2026,10,26,16,5,tzinfo=timezone.utc)
    assert not check_live_signal_status(status, now).ready
    health = check_live_signal_status(status, now, allow_short=True)
    assert health.ready
    frame = pd.read_parquet(store.signal_path)
    assert validate_signal_rows(frame, health) is None
    frame.loc[frame.target_position == -1, "target_weight"] *= -1
    assert validate_signal_rows(frame, health) == "invalid signed target weights"
    before = store.signal_path.read_bytes()
    store.write_failure(STRATEGY_ID, STAMP+86400000, model.metadata, {}, "unavailable features")
    assert store.signal_path.read_bytes() == before
    assert store.read_status()["state"] == "failed"


def test_factors_ignore_future_bars_and_caps():
    bars = pd.DataFrame({"symbol": ["A"]*64, "timestamp": np.arange(STAMP-63*14400000,STAMP+1,14400000),
                         "close": np.arange(64)+100., "high": np.arange(64)+102., "low": np.arange(64)+98., "volume": np.arange(64)+50.})
    caps = pd.DataFrame({"symbol": ["A"]*80, "date": pd.date_range("2026-09-01", periods=80),
                         "cmc_id": 1, "market_cap_usd": 100., "marketcap_identity_error": 0.})
    baseline = build_xgb15_features(bars, caps, STAMP)
    future = bars.tail(1).copy()
    future.timestamp += 14400000
    future.close = 1e12
    caps.loc[caps.date > pd.Timestamp("2026-10-24"), "market_cap_usd"] = 1e12
    actual = build_xgb15_features(pd.concat([bars, future]), caps, STAMP)
    pd.testing.assert_frame_equal(actual, baseline)
    # Missing OHLCV for another pool member must not shrink the cap percentile denominator.
    other = caps.assign(symbol="B", market_cap_usd=200.)
    both = build_xgb15_features(bars, pd.concat([caps, other]), STAMP, ["A", "B"])
    assert both.loc[both.symbol == "A", FEATURES[0]].iloc[0] == .5
    assert both.loc[both.symbol == "B", "return_6"].isna().all()


def test_missing_cap_date_fails_before_network(tmp_path):
    from src.config import Settings
    from src.live.market import MarketDataProvider, MarketDataError
    folder = tmp_path / "data/imports/xgb15"
    folder.mkdir(parents=True)
    pd.DataFrame({"date": [pd.Timestamp("2026-01-01")]}).to_parquet(folder / "caps.parquet")
    provider = MarketDataProvider(Settings(project_root=tmp_path), SimpleNamespace(parse_timeframe=lambda _: 14400))
    with pytest.raises(MarketDataError, match="OHLCV fetch not started"):
        provider.prepare(PAIRS, STAMP, XGB15LongShortStrategy.requirements)


@pytest.mark.parametrize("price,positive", [(90., True), (110., False)])
def test_dashboard_short_profit_direction(price, positive):
    trade = dict(id=1, pair="A", is_short=True, leverage=1., fee_open=.0005, fee_close=.0005,
                 open_rate=100., stake_amount=100., amount=1., open_date="2026-10-25", stop_loss=199., enter_tag="test")
    snapshot = position_snapshot(trade, price, {}, datetime(2026,10,26,tzinfo=timezone.utc))
    assert (snapshot["pnl_abs"] > 0) == positive
    assert snapshot["side"] == "short"


def test_short_execution_side_and_daily_adjustment(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "freqtrade/strategies"))
    module = importlib.import_module("AlphaCrossSectionalLiveLongShortStrategy")
    strategy = module.AlphaCrossSectionalLiveLongShortStrategy({"dry_run": True, "trading_mode": "futures", "margin_mode": "isolated"})
    assert strategy.can_short and strategy.leverage("A", None, 1, 3, 10, None, "short") == 1
    monkeypatch.setattr(strategy, "_allocation", lambda *args: (STAMP, 50.))
    data = {}
    trade = SimpleNamespace(has_open_orders=False, pair="A", is_short=True, amount=10., stake_amount=100.,
        get_custom_data=lambda k: data.get(k), set_custom_data=lambda k,v: data.update({k:v}))
    result = strategy.adjust_trade_position(trade, None, 10., 0., 1., 100.)
    assert result[0] == -50.  # Reduce an overweight short, not add exposure.
    trade.amount, trade.stake_amount = 5., 50.
    assert strategy.adjust_trade_position(trade, None, 10., 0., 1., 100.) is None
    assert strategy.adjust_trade_position(trade, None, 12., 0., 1., 100.) is None  # No intraday churn.


def test_signed_producer_idempotency_and_failure_preserves_targets(model, tmp_path):
    from src.config import Settings
    from src.live.signals import LiveLongOnlySignalProducer
    from src.live.market import PreparedMarket
    (tmp_path / "pairs.json").write_text(json.dumps(PAIRS))
    provider = SimpleNamespace(prepare=lambda *args: PreparedMarket(context().features, 60, {}))
    producer = LiveLongOnlySignalProducer(Settings(project_root=tmp_path),
        SimpleNamespace(parse_timeframe=lambda _: 14400, urls={"api": {}}), tmp_path / "pairs.json",
        tmp_path / "signals.parquet", tmp_path / "status.json", strategy=model, data_provider=provider)
    now = STAMP + 14_400_000 + 60_000
    assert not producer.run(now_ms=now).skipped
    before = producer.signal_path.read_bytes()
    assert producer.run(now_ms=now).skipped
    provider.prepare = lambda *args: PreparedMarket(context().features.with_columns(pl.lit(float("nan")).alias(FEATURES[0])), 60, {})
    with pytest.raises(ValueError, match="percentile"):
        producer.run(now_ms=now+86400000)
    assert before == producer.signal_path.read_bytes()
    assert producer.store.read_status()["state"] == "failed"


def test_short_entries_exits_stakes_and_real_money_block(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "freqtrade/strategies"))
    module = importlib.import_module("AlphaCrossSectionalLiveLongShortStrategy")
    strategy = module.AlphaCrossSectionalLiveLongShortStrategy({"dry_run": True, "trading_mode": "futures", "margin_mode": "isolated"})
    now = datetime(2026,10,26,16,5,tzinfo=timezone.utc)
    from src.live.safety import LiveSignalHealth
    health = LiveSignalHealth(True, "healthy", STAMP, STAMP+28*3600000, frozenset(["A", "B"]), 60, True, frozenset(["A"]))
    monkeypatch.setattr(strategy, "_health", lambda _: health)
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return now
    monkeypatch.setattr(module, "datetime", Clock)
    frame = pd.DataFrame({"date": [pd.Timestamp(STAMP, unit="ms", tz="UTC")], "decision_timestamp": [STAMP],
        "signal_valid_until": [health.valid_until], "signal_valid": [True], "target_position": [-1],
        "target_weight": [-.5], "volume": [1.], "close": [100.]})
    result = strategy.populate_entry_trend(frame.copy(), {"pair": "A"})
    assert result.enter_short.iloc[0] == 1 and result.enter_long.iloc[0] == 0
    exits = strategy.populate_exit_trend(frame.copy(), {"pair": "A"})
    assert exits.exit_long.iloc[0] == 1 and exits.exit_short.iloc[0] == 0
    strategy.dp = SimpleNamespace(get_analyzed_dataframe=lambda *args: (frame, None))
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10000.)
    monkeypatch.delenv("ALPHA_LIVE_AUDIT_PATH", raising=False)
    from freqtrade.persistence import Trade
    monkeypatch.setattr(Trade, "get_open_trades", lambda: [])
    assert strategy._confirm_live_entry("A", now, "short")
    assert not strategy._confirm_live_entry("A", now, "long")
    assert strategy.custom_stake_amount("A", now, 100., 1., 5., 10000., 1., None, "short") == 5000.
    strategy.config["dry_run"] = False
    assert not strategy._confirm_live_entry("A", now, "short")
