import hashlib
import json
from datetime import datetime, timezone
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.data.xgb15_caps import cap_coverage, normalize_caps, parse_history, refresh_caps, required_cap_date
from src.live.paths import default_instance_name
from src.strategies.registry import create_strategy
from src.strategies.xgb15 import FEATURES, STRATEGY_ID
from src.strategies.xgb15_assets import asset_paths, check_assets, export_research_bundle, import_bundle

NOW = datetime(2026, 11, 11, 1, tzinfo=timezone.utc)
STAMP = int(pd.Timestamp("2026-11-10T12:00:00Z").timestamp() * 1000)
PAIRS = [f"C{i:02}/USDT:USDT" for i in range(60)]


def caps_frame():
    last = required_cap_date(STAMP)
    return pd.DataFrame([{"symbol": pair, "date": day, "cmc_id": i + 1,
                          "market_cap_usd": 100. + i, "marketcap_identity_error": 0.}
                         for i, pair in enumerate(PAIRS)
                         for day in pd.date_range(last - pd.Timedelta(days=39), last)])


@pytest.fixture
def exported(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    matrix = np.random.default_rng(1).normal(size=(80, 15))
    data = xgb.DMatrix(matrix, label=matrix[:, 0], feature_names=list(FEATURES))
    model = xgb.train({"objective": "reg:squarederror", "max_depth": 2, "nthread": 1}, data, num_boost_round=120)
    path = source / "model_2026.ubj"
    model.save_model(path)
    manifest = {"strategy_id": STRATEGY_ID, "features": list(FEATURES), "preprocessing": "percentile_average",
                "models": {"2026": {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "pairs": PAIRS, "fit_cutoff_utc": "2026-01-01T00:00:00Z"}}}
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "pairs.json").write_text(json.dumps(PAIRS))
    (source / "asset_mapping.json").write_text(json.dumps([
        {"contract": pair, "cmc_ids": [i + 1]} for i, pair in enumerate(PAIRS)]))
    caps_frame().to_parquet(source / "caps.parquet", index=False)
    return source


def test_missing_assets_report_is_read_only(tmp_path):
    report = check_assets(tmp_path, NOW)
    assert report["ready"] is False and report["scope"] == "offline_assets_only"
    assert report["model_year"] == 2026 and report["decision_timestamp"] == STAMP
    assert report["required_cap_date"] == "2026-11-08"
    assert [check["name"] for check in report["checks"]] == ["model", "pairs", "caps"]
    assert not list(tmp_path.iterdir())


def test_import_real_model_checks_registry_and_does_not_start_worker(tmp_path, exported):
    root = tmp_path / "runtime"
    root.mkdir()
    result = import_bundle(root, exported, 2026)
    assert result["installed"] and not result["worker_started"]
    report = check_assets(root, NOW)
    assert report["ready"] and report["cap_refresh_mapping_ready"]
    assert not report["automatic_refresh_scheduled"] and not report["live_execution_verified"]
    strategy = create_strategy(STRATEGY_ID, root / "data/models/production")
    assert strategy._model(2026).num_boosted_rounds() == 120
    assert default_instance_name(STRATEGY_ID) == "xgb15-paper"
    assert not (root / "freqtrade").exists()
    with pytest.raises(ValueError, match="overwrite"):
        import_bundle(root, exported, 2026)


def test_public_manifest_without_pool_is_rejected_before_install(tmp_path, exported):
    path = exported / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["models"]["2026"]["pairs"]
    path.write_text(json.dumps(manifest))
    root = tmp_path / "runtime"
    root.mkdir()
    with pytest.raises(KeyError):
        import_bundle(root, exported, 2026)
    assert not (root / "data").exists()


def test_corrupt_model_or_pool_mismatch_is_rejected(tmp_path, exported):
    root = tmp_path / "runtime"
    root.mkdir()
    (exported / "pairs.json").write_text(json.dumps(PAIRS[:-1] + ["OTHER/USDT:USDT"]))
    with pytest.raises(ValueError, match="match"):
        import_bundle(root, exported, 2026)
    (exported / "model_2026.ubj").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        import_bundle(root, exported, 2026)
    assert not (root / "data").exists()


def test_stale_caps_block_readiness(tmp_path, exported):
    root = tmp_path / "runtime"
    root.mkdir()
    import_bundle(root, exported, 2026)
    _, private = asset_paths(root)
    caps = pd.read_parquet(private / "caps.parquet")
    caps.loc[caps.symbol == PAIRS[0], "cmc_id"] = [1] * 39 + [999]
    caps.to_parquet(private / "caps.parquet", index=False)
    report = check_assets(root, NOW)
    assert not report["ready"]
    coverage = next(item for item in report["checks"] if item["name"] == "cap_coverage")
    assert coverage["details"]["complete_count"] == 59
    assert coverage["details"]["incomplete_pairs"] == [PAIRS[0]]


def test_future_caps_cannot_fill_missing_history_and_duplicates_fail():
    caps = caps_frame()
    baseline = cap_coverage(caps, PAIRS, STAMP)
    assert baseline["complete_count"] == 60
    future = caps.loc[caps.date == caps.date.max()].copy()
    future.date += pd.Timedelta(days=10)
    future.market_cap_usd = 1e15
    assert cap_coverage(pd.concat([caps, future]), PAIRS, STAMP) == baseline
    with pytest.raises(ValueError, match="duplicate"):
        normalize_caps(pd.concat([caps, caps.head(1)]))


def payload(identifier, start, end):
    return {"data": {"id": identifier, "quotes": [
        {"timeOpen": day.tz_localize("UTC").isoformat(),
         "timeClose": (day.tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).isoformat(),
         "quote": {"marketCap": 100., "close": 10., "circulatingSupply": 10.}}
        for day in pd.date_range(start, end)]}, "status": {"error_code": "0"}}


def test_refresh_uses_explicit_ids_and_preserves_old_observations(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    (private / "asset_mapping.json").write_text(json.dumps([{"contract": pair, "cmc_ids": [1]} for pair in PAIRS]))
    old = caps_frame().assign(cmc_id=1)
    old.to_parquet(private / "caps.parquet", index=False)
    start = required_cap_date(STAMP) - pd.Timedelta(days=39)
    end = pd.Timestamp(NOW).tz_localize(None).normalize() - pd.Timedelta(days=1)
    urls = []

    def fetch(url):
        urls.append(url)
        return payload(1, start, end)

    result = refresh_caps(private, PAIRS, STAMP, NOW, fetch)
    assert len(urls) == 1 and "id=1&" in urls[0]
    assert result["complete_count"] == 60
    actual = pd.read_parquet(private / "caps.parquet")
    assert actual.loc[(actual.symbol == PAIRS[-1]) & (actual.date == start), "market_cap_usd"].iloc[0] == 159.
    assert actual.date.max() == end


def test_access_denial_leaves_previous_parquet_unchanged(tmp_path):
    (tmp_path / "asset_mapping.json").write_text(json.dumps([{"contract": pair, "cmc_ids": [1]} for pair in PAIRS]))
    path = tmp_path / "caps.parquet"
    caps_frame().to_parquet(path, index=False)
    before = path.read_bytes()
    calls = []

    def denied(url):
        calls.append(url)
        raise HTTPError(url, 403, "forbidden", {}, None)

    with pytest.raises(HTTPError):
        refresh_caps(tmp_path, PAIRS, STAMP, NOW, denied)
    assert len(calls) == 1 and path.read_bytes() == before


def test_parse_rejects_provider_errors_and_incomplete_days():
    day = pd.Timestamp("2026-11-08")
    with pytest.raises(ValueError, match="error status"):
        parse_history({"status": {"error_code": "401"}}, 1, day, day)
    data = payload(1, day, day)
    data["data"]["quotes"][0]["timeClose"] = "2026-11-08T12:00:00Z"
    with pytest.raises(ValueError, match="completed daily"):
        parse_history(data, 1, day, day)


def test_completed_day_millisecond_precision_is_supported():
    day = pd.Timestamp("2026-11-08")
    data = payload(1, day, day)
    data["data"]["quotes"][0]["timeClose"] = "2026-11-08T23:59:59.999Z"
    assert parse_history(data, 1, day, day)[day]["market_cap_usd"] == 100.


def test_export_research_assets_without_training_or_source_changes(tmp_path, exported):
    import shutil

    research = tmp_path / "research"
    reference = research / "reports/xgb15_gpu_expanding_120_depth4_lr005_2022_2026"
    (reference / "2026").mkdir(parents=True)
    shutil.copyfile(exported / "model_2026.ubj", reference / "2026/model.ubj")
    runtime_manifest = json.loads((exported / "manifest.json").read_text())
    metadata = {"features": list(FEATURES), "num_boost_round": 120, "params": {"max_depth": 4},
                "xgboost_version": xgb.__version__, "model_records": [{
                    "prediction_year": 2026, "fit_cutoff_utc": runtime_manifest["models"]["2026"]["fit_cutoff_utc"]}]}
    (reference / "manifest.json").write_text(json.dumps(metadata))
    tables = research / "data/model_tables/factors15_2021_2026/predict_table_2022-2026"
    tables.mkdir(parents=True)
    pd.DataFrame({"symbol": PAIRS}).to_parquet(tables / "predict_table_2026.parquet", index=False)
    fundamentals = research / "data/fundamentals/cmc_2021_2025"
    fundamentals.mkdir(parents=True)
    caps_frame().rename(columns={"symbol": "contract"}).to_parquet(fundamentals / "contract_daily.parquet", index=False)
    shutil.copyfile(exported / "asset_mapping.json", fundamentals / "asset_mapping.json")
    original = {str(path): path.read_bytes() for path in research.rglob("*") if path.is_file()}
    output = tmp_path / "package"
    result = export_research_bundle(research, output, 2026)
    assert result["exported"] and not result["source_modified"]
    root = tmp_path / "runtime"
    root.mkdir()
    import_bundle(root, output, 2026)
    assert check_assets(root, NOW)["ready"]
    assert original == {str(path): path.read_bytes() for path in research.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="overwrite"):
        export_research_bundle(research, output, 2026)
