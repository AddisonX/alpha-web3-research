import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from scripts.prepare_paper import prepare_paper, verify_bundle
from src.live.instances import prepare_instance

ROOT = Path(__file__).resolve().parents[1]


def copy_bundle(root):
    manifest = verify_bundle(ROOT)
    for relative in ["configs/paper/bundle_manifest.json"] + [item["path"] for item in manifest["files"]]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return manifest


def test_all_published_assets_have_matching_checksums():
    bundle = verify_bundle(ROOT)
    assert [item["pair_count"] for item in bundle["instances"]] == [85, 85, 297]
    assert len(bundle["files"]) == 32
    seed = pd.read_parquet(ROOT / "configs/paper/xgb15-paper/caps_seed.parquet")
    assert str(seed.date.max().date()) == bundle["caps_seed"]["end"]
    assert seed.date.nunique() == 60
    for profile in bundle["instances"]:
        folder = ROOT / "configs/paper" / profile["instance"]
        config = json.loads((folder / "config.json").read_text())
        pairs = json.loads((folder / "pairs.json").read_text())
        assert config["exchange"]["pair_whitelist"] == pairs
        assert len(pairs) == profile["pair_count"]
        assert config["dry_run"] is True and config["dry_run_wallet"] == 10000
        assert not config["exchange"]["key"] and not config["exchange"]["secret"]
        assert "api_server" not in config
        assert not config["telegram"]["enabled"] and not config["telegram"]["token"]
        text = json.dumps(config)
        assert "/Users/" not in text and "/home/" not in text and "testnet" not in text


def test_clean_checkout_prepares_all_three_without_starting_workers(tmp_path):
    from freqtrade.configuration.config_validation import validate_config_schema
    copy_bundle(tmp_path)
    instances = prepare_paper(tmp_path)
    assert len(instances) == 3
    for instance in instances:
        config = json.loads((instance / "config.json").read_text())
        validate_config_schema(dict(config))
        assert config["db_url"] == "sqlite:///" + str(instance / "database/trades.dryrun.sqlite")
        assert not list((instance / "database").iterdir())
        assert not list((instance / "runtime").iterdir())
        assert not list((instance / "signals").iterdir())
    xgb = json.loads((instances[-1] / "config.json").read_text())
    assert xgb["strategy"] == "AlphaCrossSectionalLiveLongShortStrategy"
    assert xgb["fee"] == .0005 and xgb["max_open_trades"] == 297
    assert xgb["stoploss"] == -.99 and xgb["position_adjustment_enable"] is True
    live = tmp_path / "data/imports/xgb15/runtime/caps.parquet"
    assert live.read_bytes() == (tmp_path / "configs/paper/xgb15-paper/caps_seed.parquet").read_bytes()


def test_preparation_keeps_existing_caps_database_and_custom_config(tmp_path):
    copy_bundle(tmp_path)
    instance = prepare_paper(tmp_path, ["xgb15-paper"])[0]
    live = tmp_path / "data/imports/xgb15/runtime/caps.parquet"
    live.write_bytes(b"existing observation must not be overwritten")
    database = instance / "database/trades.dryrun.sqlite"
    database.write_bytes(b"existing account must not be overwritten")
    config_path = instance / "config.json"
    config = json.loads(config_path.read_text())
    config["dry_run_wallet"] = 12345
    config_path.write_text(json.dumps(config))
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in [live, database]]
    prepare_paper(tmp_path, ["xgb15-paper"])
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in [live, database]] == before
    assert json.loads(config_path.read_text())["dry_run_wallet"] == 12345


def test_modified_published_profile_fails_before_any_instance_is_created(tmp_path):
    copy_bundle(tmp_path)
    path = tmp_path / "configs/paper/tokyo-lightgbm/config.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_paper(tmp_path)
    assert not (tmp_path / "freqtrade/instances").exists()


@pytest.mark.parametrize("field,value", [("dry_run", False), ("credentials", "not-allowed")])
def test_direct_profile_loading_preserves_dry_run_guards(tmp_path, field, value):
    copy_bundle(tmp_path)
    path = tmp_path / "configs/paper/tokyo-lightgbm/config.json"
    config = json.loads(path.read_text())
    if field == "credentials":
        config["exchange"]["key"] = value
    else:
        config[field] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="dry-run|credentials"):
        prepare_instance(tmp_path, strategy_id="tokyo-lightgbm-long-only-v1")


def test_only_selected_profile_is_prepared(tmp_path):
    copy_bundle(tmp_path)
    result = prepare_paper(tmp_path, ["return-1-reversal"])
    assert [p.name for p in result] == ["return-1-reversal"]
    assert not (tmp_path / "data/imports/xgb15/runtime").exists()
    with pytest.raises(ValueError, match="unknown"):
        prepare_paper(tmp_path, ["missing"])
