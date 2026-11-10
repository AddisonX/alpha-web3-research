from pathlib import Path
import hashlib
import json
import numpy as np
import lightgbm as lgb
import polars as pl
import pytest
from typer.testing import CliRunner
from src.cli import app
from src.live.config import write_live_long_only_config
from src.strategies.lightgbm import ProductionEnsemble


def test_production_bundle_loads_and_predicts_without_research_modules(tmp_path):
    features = [f"factor_{index}" for index in range(30)]
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(80, len(features)))
    dataset = lgb.Dataset(matrix, label=matrix[:, 0], feature_name=features)
    bundle = tmp_path / "fixture-v1"
    bundle.mkdir()
    models = []
    for index in range(5):
        model = lgb.train({"objective": "regression", "verbosity": -1, "num_threads": 1,
                           "num_leaves": 4, "min_data_in_leaf": 2}, dataset, num_boost_round=2)
        path = bundle / f"model_{index}.txt"
        model.save_model(str(path))
        models.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (tmp_path / "latest.json").write_text(json.dumps({"bundle": bundle.name}))
    (bundle / "manifest.json").write_text(json.dumps({
        "source_experiment": "synthetic-loader-test-not-performance",
        "source_config": {"features": features, "timeframe": "4h", "horizon_bars": 6,
                          "label_column": "forward_return", "top_quantile_fraction": .1},
        "models": models,
    }))
    ensemble = ProductionEnsemble(tmp_path)
    assert len(ensemble.boosters) == 5
    assert len(ensemble.features) == 30
    assert ensemble.timeframe == "4h" and ensemble.horizon_bars == 6
    manifest = json.loads(ensemble.manifest_path.read_text())
    for item in manifest["models"]:
        assert hashlib.sha256((ensemble.bundle_dir / item["file"]).read_bytes()).hexdigest() == item["sha256"]
    assert np.isfinite(ensemble.predict(pl.DataFrame({n: [0.0, 1.0] for n in ensemble.features}))).all()


def test_optional_local_production_bundle():
    root = Path(__file__).resolve().parents[1] / "data/models/production"
    if not (root / "latest.json").exists():
        pytest.skip("private production weights are not distributed")
    ensemble = ProductionEnsemble(root)
    assert len(ensemble.boosters) == 5 and len(ensemble.features) == 30
    manifest = json.loads(ensemble.manifest_path.read_text())
    for item in manifest["models"]:
        assert hashlib.sha256((ensemble.bundle_dir / item["file"]).read_bytes()).hexdigest() == item["sha256"]


def test_only_runtime_command_groups_remain():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "live" in result.output and "config" in result.output
    for name in ["research", "models", "backtest", "strategies"]:
        assert CliRunner().invoke(app, [name, "--help"]).exit_code != 0
    assert CliRunner().invoke(app, ["config", "show"]).exit_code == 0


def test_configuration_defaults_to_mainnet_dry_run(tmp_path):
    pairs = tmp_path / "pairs.json"
    pairs.write_text('["BTC/USDT:USDT"]')
    path = write_live_long_only_config(pairs, tmp_path / "config.json")
    config = json.loads(path.read_text())
    assert config["dry_run"] is True
    assert config["max_open_trades"] == 9
    assert "testnet" not in json.dumps(config)
