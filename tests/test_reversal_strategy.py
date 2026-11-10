import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest

from src.live.safety import check_live_signal_status
from src.strategies.base import StrategyContext
from src.strategies.registry import create_strategy
from src.strategies.reversal import STRATEGY_ID, ReturnOneReversalStrategy
from scripts.replay_reversal import BAR_MS, DAY_MS, replay
from scripts.run_reversal_dryrun import prepare_instance, environment, trade_command

ROOT = Path(__file__).resolve().parents[1]
STAMP = int(datetime(2026, 7, 27, tzinfo=timezone.utc).timestamp() * 1000)


def test_lowest_decile_tie_break_and_input_permutation():
    strategy = ReturnOneReversalStrategy()
    frame = pl.DataFrame({"symbol": [f"S{i:02}" for i in reversed(range(88))], "return_1": [0.1]*88})
    result = strategy.decide(StrategyContext(STAMP, frame))
    result.validate(StrategyContext(STAMP, frame))
    selected = result.targets.filter(pl.col("target_position") == 1)["pair"].to_list()
    assert selected == [f"S{i:02}" for i in range(9)]
    assert strategy.decide(StrategyContext(STAMP, frame.reverse())).targets.equals(result.targets)


def test_nonfinite_values_are_not_selected_and_no_finite_values_fail():
    frame = pl.DataFrame({"symbol": ["A", "B", "C", "D"], "return_1": [None, float("nan"), float("inf"), -.1]})
    decision = ReturnOneReversalStrategy().decide(StrategyContext(STAMP, frame))
    assert decision.targets.filter(pl.col("target_position") == 1)["pair"].to_list() == ["D"]
    with pytest.raises(ValueError, match="no finite"):
        ReturnOneReversalStrategy().decide(StrategyContext(STAMP, frame.head(3)))


def test_registered_rule_does_not_import_or_load_lightgbm():
    code = "from pathlib import Path; from src.strategies.registry import create_strategy; import sys; s=create_strategy('" + STRATEGY_ID + "',Path('/nonexistent')); assert s.metadata.model_count==0; assert 'lightgbm' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def snapshot(tmp_path):
    rows = []
    for i in range(60):
        for j in range(52):
            price = 100 + j*.1 + math.sin(i+j)*.2
            rows.append({"symbol": f"COIN{i:02}/USDT:USDT", "timestamp": STAMP+(j-39)*BAR_MS,
                         "open": price, "high": price+1, "low": price-1, "close": price+.1, "volume": 100.+j})
    path = tmp_path / "bars.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def test_two_day_replay_is_causal_and_preserves_default_files(tmp_path):
    path = snapshot(tmp_path)
    output = tmp_path / "replay"
    report = replay(path, output, STAMP, 2)
    assert all(r["safety_validated"] and len(r["selected_pairs"]) == 6 for r in report["runs"])
    mutated = pl.read_parquet(path).with_columns([
        pl.when(pl.col("timestamp") > STAMP).then(pl.col(c)*100).otherwise(pl.col(c)).alias(c)
        for c in ["open", "high", "low", "close", "volume"]])
    other = tmp_path / "future_changed.parquet"
    mutated.write_parquet(other)
    changed = replay(other, tmp_path / "other", STAMP, 1)
    assert changed["runs"][0]["selected_pairs"] == report["runs"][0]["selected_pairs"]
    assert report["model_count"] == 0 and report["performance_claim"] is False


def test_instance_isolation_and_dryrun_enforcement(tmp_path):
    pairs = tmp_path / "pairs.json"
    pairs.write_text('["BTC/USDT:USDT"]')
    instance = prepare_instance(tmp_path, pairs)
    config = json.loads((instance / "config.json").read_text())
    assert config["dry_run"] is True
    from freqtrade.configuration.config_validation import validate_config_schema
    validate_config_schema(dict(config))
    assert config["exchange"]["ccxt_config"]["requests_trust_env"] is True
    assert config["exchange"]["ccxt_async_config"]["aiohttp_trust_env"] is True
    assert str(instance) in config["db_url"]
    assert "tradesv3.live-long-only" not in config["db_url"]
    env = environment(instance)
    assert str(instance) in env["ALPHA_FREQTRADE_LIVE_SIGNAL_PATH"]
    assert str(instance) in env["ALPHA_LIVE_AUDIT_PATH"]
    assert "--dry-run" in trade_command(instance)
    config["dry_run"] = False
    (instance / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="dry-run"):
        prepare_instance(tmp_path, pairs)


def test_existing_environment_is_opt_in_for_exchange():
    from src.config import Settings
    from src.data.exchange import create_exchange
    ordinary = create_exchange(Settings(ccxt_trust_env=False))
    instance = create_exchange(Settings(ccxt_trust_env=True))
    try:
        assert ordinary.session.trust_env is False
        assert instance.session.trust_env is True
    finally:
        ordinary.close()
        instance.close()


def test_tokyo_instance_has_structured_and_separate_paths(tmp_path):
    from src.strategies.base import DEFAULT_STRATEGY_ID
    from scripts.run_reversal_dryrun import instance_path, database_path, instance_info
    pairs = tmp_path / "pairs.json"
    pairs.write_text('["BTC/USDT:USDT"]')
    instance = prepare_instance(tmp_path, pairs, strategy_id=DEFAULT_STRATEGY_ID,
                                instance_name="tokyo-lightgbm", structured=True)
    config = json.loads((instance / "config.json").read_text())
    assert instance_info(instance)["strategy_id"] == DEFAULT_STRATEGY_ID
    assert config["db_url"] == "sqlite:///" + str(database_path(instance))
    assert database_path(instance).parent.name == "database"
    assert instance_path(instance, "runtime").name == "runtime"
    command = trade_command(instance)
    assert command[command.index("--userdir") + 1] == str(instance / "user_data")
    with pytest.raises(ValueError, match="differs"):
        prepare_instance(tmp_path, pairs, instance_name="tokyo-lightgbm")


def test_canonical_paths_do_not_recreate_top_level_user_data(tmp_path):
    from src.live.paths import instance_directory
    pairs = tmp_path / "pairs.json"
    pairs.write_text('["BTC/USDT:USDT"]')
    instance = prepare_instance(tmp_path, pairs)
    assert instance == instance_directory(tmp_path, STRATEGY_ID)
    assert instance.name == "return-1-reversal"
    assert not (tmp_path / "freqtrade/user_data").exists()
    assert trade_command(instance)[trade_command(instance).index("--strategy-path") + 1] == str(ROOT / "freqtrade/strategies")


def test_shared_adapter_default_points_to_tokyo_instance(monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(ROOT / "freqtrade/strategies"))
    monkeypatch.delenv("ALPHA_FREQTRADE_LIVE_SIGNAL_PATH", raising=False)
    module = importlib.import_module("AlphaCrossSectionalLiveLongOnlyStrategy")
    assert module.AlphaCrossSectionalLiveLongOnlyStrategy._signal_path() == ROOT / "freqtrade/instances/tokyo-lightgbm/signals/alpha_live_long_only_signals.parquet"


def test_freqtrade_entry_exit_stoploss_and_restart_with_real_sqlite(tmp_path, monkeypatch):
    import importlib
    from freqtrade.persistence import Trade, init_db
    from freqtrade.enums import ExitType
    import pandas as pd

    monkeypatch.syspath_prepend(str(ROOT / "freqtrade/strategies"))
    module = importlib.import_module("AlphaCrossSectionalLiveLongOnlyStrategy")
    LiveStrategy = module.AlphaCrossSectionalLiveLongOnlyStrategy
    report = replay(snapshot(tmp_path), tmp_path / "replay", STAMP, 2)
    status_path = tmp_path / "replay/alpha_live_long_only_status.json"
    signal_path = tmp_path / "replay/alpha_live_long_only_signals.parquet"
    monkeypatch.setenv("ALPHA_FREQTRADE_LIVE_STATUS_PATH", str(status_path))
    monkeypatch.setenv("ALPHA_FREQTRADE_LIVE_SIGNAL_PATH", str(signal_path))
    monkeypatch.delenv("ALPHA_LIVE_AUDIT_PATH", raising=False)
    now = datetime.fromtimestamp((STAMP+DAY_MS+BAR_MS+300_000)/1000, timezone.utc)
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return now if tz else now.replace(tzinfo=None)
    monkeypatch.setattr(module, "datetime", Clock)
    for name in ["_signals", "_signals_mtime_ns", "_validated_key", "_last_health_reason"]:
        monkeypatch.setattr(LiveStrategy, name, None)
    init_db("sqlite:///" + str(tmp_path / "acceptance.sqlite"))
    strategy = LiveStrategy({"dry_run": True})
    pair = report["runs"][-1]["selected_pairs"][0]
    def analyzed(pair):
        candles = pd.DataFrame({"date": [pd.Timestamp(now).floor("4h") - pd.Timedelta(hours=4)], "volume": [100.]})
        return strategy.populate_indicators(candles, {"pair": pair})
    strategy.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair,timeframe:(analyzed(pair),now))
    def confirm(): return strategy.confirm_trade_entry(pair,"limit",1.,100.,"gtc",now,None,"long")
    try:
        assert strategy.populate_entry_trend(analyzed(pair), {"pair":pair})["enter_long"].iloc[-1] == 1
        assert confirm()
        trade = Trade(pair=pair, exchange="binance", stake_currency="USDT", base_currency=pair.split('/')[0],
                      stake_amount=100., amount=1., open_rate=100., fee_open=.0005, fee_close=.0005,
                      open_date=now.replace(tzinfo=None), is_open=True, is_short=False, leverage=1., timeframe=240)
        Trade.session.add(trade)
        Trade.commit()
        assert Trade.get_trades_proxy(is_open=True)[0].pair == pair
        exit_result = strategy.ft_stoploss_reached(89., trade, now, -.11, 0)
        assert exit_result.exit_type == ExitType.STOP_LOSS
        trade.close(89.)
        trade.close_date = now.replace(tzinfo=None)
        Trade.commit()
        assert not confirm()  # Same decision must not re-enter a stopped-out position.
        other = next(f"COIN{i:02}/USDT:USDT" for i in range(60) if f"COIN{i:02}/USDT:USDT" not in report["runs"][-1]["selected_pairs"])
        assert strategy.populate_exit_trend(analyzed(other), {"pair":other})["exit_long"].iloc[-1] == 1
        bind = Trade.session.get_bind()
        Trade.session.remove()
        bind.dispose()
        init_db("sqlite:///" + str(tmp_path / "acceptance.sqlite"))
        for name in ["_signals", "_signals_mtime_ns", "_validated_key", "_last_health_reason"]:
            monkeypatch.setattr(LiveStrategy, name, None)
        restarted = LiveStrategy({"dry_run": True})
        assert restarted.populate_entry_trend(analyzed(pair), {"pair":pair})["enter_long"].iloc[-1] == 1
        restarted.dp = strategy.dp
        assert not restarted.confirm_trade_entry(pair,"limit",1.,100.,"gtc",now,None,"long")
    finally:
        bind = Trade.session.get_bind()
        Trade.session.remove()
        bind.dispose()
