from __future__ import annotations

import importlib
import json
from pathlib import Path

import ccxt
import pytest

from alpha_web3_research.live.testnet import (
    TESTNET_API_URLS,
    load_testnet_credentials,
    validate_testnet_config,
    write_testnet_config,
)


def _config(tmp_path: Path) -> dict:
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps(["BTC/USDT:USDT"]), encoding="utf-8")
    output = tmp_path / "config.json"
    write_testnet_config(pairs, output, tmp_path / "testnet-trades.sqlite")
    return json.loads(output.read_text(encoding="utf-8"))


def test_testnet_config_pins_every_endpoint_and_limits_exposure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    validate_testnet_config(config)
    assert config["exchange"]["ccxt_config"]["urls"]["api"] == TESTNET_API_URLS
    assert ccxt.binance(config["exchange"]["ccxt_config"]).urls["api"] == TESTNET_API_URLS
    assert config["max_open_trades"] == 1
    assert config["stake_amount"] == 20
    assert config["order_types"]["stoploss_on_exchange"] is True


@pytest.mark.parametrize("change", [
    lambda c: c.update(dry_run=True),
    lambda c: c.update(stake_amount=200),
    lambda c: c.update(max_open_trades=9),
    lambda c: c["exchange"]["ccxt_config"]["urls"]["api"].update(fapiPrivate="https://fapi.binance.com/fapi/v1"),
    lambda c: c["exchange"]["ccxt_async_config"]["urls"]["api"].update(fapiPrivate="https://fapi.binance.com/fapi/v1"),
    lambda c: c["order_types"].update(stoploss_on_exchange=False),
])
def test_unsafe_testnet_config_fails_closed(tmp_path: Path, change) -> None:
    config = _config(tmp_path)
    change(config)
    with pytest.raises(ValueError):
        validate_testnet_config(config)


def test_credentials_permissions_and_shape(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"exchange": {"key": "test", "secret": "secret"}}), encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        load_testnet_credentials(path)
    path.chmod(0o600)
    assert load_testnet_credentials(path) == ("test", "secret")
    path.write_text(json.dumps({"exchange": {"key": "test", "secret": "secret", "urls": {}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="only"):
        load_testnet_credentials(path)


def test_testnet_strategy_requires_armed_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy_dir = Path(__file__).resolve().parents[1] / "freqtrade" / "user_data" / "strategies"
    monkeypatch.syspath_prepend(str(strategy_dir))
    strategy_type = importlib.import_module("AlphaTestnetLiveLongOnlyStrategy").AlphaTestnetLiveLongOnlyStrategy
    strategy = strategy_type(_config(tmp_path))
    monkeypatch.delenv("ALPHA_TESTNET_ARM_FILE", raising=False)
    assert not strategy._execution_mode_allowed()
    arm = tmp_path / "ARMED"
    monkeypatch.setenv("ALPHA_TESTNET_ARM_FILE", str(arm))
    assert not strategy._execution_mode_allowed()
    arm.touch()
    assert strategy._execution_mode_allowed()
    strategy.config["exchange"]["ccxt_config"]["urls"]["api"]["fapiPrivate"] = "https://fapi.binance.com/fapi/v1"
    assert not strategy._execution_mode_allowed()
