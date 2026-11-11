"""Offline asset checks and non-overwriting installation for the XGB15 runtime."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.xgb15_caps import BAR_MS, cap_coverage, normalize_caps, read_mapping, required_cap_date
from src.live.signals import latest_closed_decision_timestamp
from src.strategies.xgb15 import STRATEGY_ID, XGB15LongShortStrategy


def asset_paths(root: Path) -> tuple[Path, Path]:
    return (root / "data/models/strategies" / STRATEGY_ID / "v1", root / "data/imports/xgb15")


def read_pairs(path: Path) -> list[str]:
    return validate_pairs(json.loads(path.read_text(encoding="utf-8")))


def validate_pairs(pairs) -> list[str]:
    if not isinstance(pairs, list) or len(pairs) < 60:
        raise ValueError("annual pool must contain at least 60 pairs")
    if any(not isinstance(pair, str) or not pair.endswith("/USDT:USDT") or not pair.split("/")[0] for pair in pairs):
        raise ValueError("expected Binance USDT perpetual pair names")
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate annual pool pairs")
    return pairs


def validate_bundle(directory: Path, year: int) -> dict:
    strategy = XGB15LongShortStrategy(directory)
    manifest = strategy.manifest
    if manifest.get("strategy_id") != STRATEGY_ID:
        raise ValueError("wrong strategy_id in runtime manifest")
    record = manifest["models"][str(year)]
    validate_pairs(record["pairs"])
    cutoff = pd.Timestamp(record["fit_cutoff_utc"])
    if cutoff.tzinfo is None or cutoff > pd.Timestamp(f"{year}-01-01", tz="UTC"):
        raise ValueError("annual model cutoff must be timezone-aware and not after year start")
    strategy._model(year)  # Apply the exact runtime hash, path, tree-count and feature checks.
    return manifest


def check_assets(root: Path, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("as-of must include timezone")
    stamp = latest_closed_decision_timestamp(int(now.timestamp() * 1000), BAR_MS, 24, 12)
    signal_time = pd.Timestamp(stamp + BAR_MS, unit="ms", tz="UTC")
    year = signal_time.tz_convert("Asia/Shanghai").year
    model_dir, private = asset_paths(root)
    checks, pairs = [], None

    def inspect(name, action):
        try:
            details = action()
            checks.append({"name": name, "status": "ok", "details": details})
            return details
        except (OSError, ValueError, KeyError, TypeError) as error:
            checks.append({"name": name, "status": "missing" if isinstance(error, FileNotFoundError) else "invalid", "message": str(error)})
            return None

    # These checks never start a model worker, collect data, or create an instance.
    def model_check():
        nonlocal pairs
        manifest = validate_bundle(model_dir, year)
        pairs = manifest["models"][str(year)]["pairs"]
        return {"year": year, "pair_count": len(pairs), "model": manifest["models"][str(year)]["file"]}

    inspect("model", model_check)

    def pool_check():
        configured = read_pairs(private / "pairs.json")
        if pairs is None:
            raise ValueError("model must validate before annual pool can be matched")
        if set(configured) != set(pairs):
            raise ValueError("pairs.json differs from the annual model pool")
        return {"pair_count": len(configured)}

    inspect("pairs", pool_check)

    def caps_check():
        caps = normalize_caps(pd.read_parquet(private / "caps.parquet"))
        if pairs is None:
            raise ValueError("model must validate before cap coverage can be checked")
        coverage = cap_coverage(caps, pairs, stamp)
        if coverage["complete_count"] < 60:
            checks.append({"name": "cap_coverage", "status": "invalid", "details": coverage,
                           "message": "fewer than 60 complete 40-day stable-identity histories"})
        return coverage

    inspect("caps", caps_check)
    mapping_ready = False
    try:
        if pairs is not None:
            read_mapping(private / "asset_mapping.json", pairs)
            mapping_ready = True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {"strategy_id": STRATEGY_ID, "scope": "offline_assets_only", "as_of": now.isoformat(),
            "signal_time": signal_time.isoformat(), "model_year": year, "decision_timestamp": stamp,
            "required_cap_date": str(required_cap_date(stamp).date()),
            "ready": all(check["status"] == "ok" for check in checks), "checks": checks,
            "cap_refresh_mapping_ready": mapping_ready, "automatic_refresh_scheduled": False,
            "live_execution_verified": False,
            "acceptance_note": "Private prediction/target parity and fresh OHLCV still require independent verification."}


def import_bundle(root: Path, source: Path, year: int) -> dict:
    """Install an already exported runtime package, not the private research project."""
    source = source.resolve()
    manifest = validate_bundle(source, year)
    record = manifest["models"][str(year)]
    pairs = record["pairs"]
    if (source / "pairs.json").exists() and set(read_pairs(source / "pairs.json")) != set(pairs):
        raise ValueError("exported pairs do not match model manifest")
    caps = normalize_caps(pd.read_parquet(source / "caps.parquet"))
    mapping_path = source / "asset_mapping.json"
    if mapping_path.exists():
        read_mapping(mapping_path, pairs)
    model_dir, private = asset_paths(root)
    for target in (model_dir, private):
        if target.exists() or target.is_symlink():
            raise ValueError(f"refusing to overwrite existing assets: {target}")
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("asset destination escapes project root")
    data = root / "data"
    data.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xgb15-import-", dir=data) as temporary:
        stage = Path(temporary)
        staged_model, staged_private = stage / "model", stage / "private"
        staged_model.mkdir()
        staged_private.mkdir()
        target = staged_model / record["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / record["file"], target)
        # Only include the exported year: historical models are not required to run it.
        (staged_model / "manifest.json").write_text(json.dumps({**manifest, "models": {str(year): record}}, indent=2))
        (staged_private / "pairs.json").write_text(json.dumps(pairs, indent=2))
        caps.to_parquet(staged_private / "caps.parquet", index=False)
        if mapping_path.exists():
            shutil.copyfile(mapping_path, staged_private / mapping_path.name)
        validate_bundle(staged_model, year)
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        private.parent.mkdir(parents=True, exist_ok=True)
        staged_model.rename(model_dir)
        try:
            staged_private.rename(private)
        except Exception:
            shutil.rmtree(model_dir)  # Roll back only this newly installed, owned directory.
            raise
    return {"installed": True, "strategy_id": STRATEGY_ID, "year": year,
            "sha256": hashlib.sha256((model_dir / record["file"]).read_bytes()).hexdigest(),
            "worker_started": False, "cap_freshness_verified": False}


def export_research_bundle(research: Path, output: Path, year: int) -> dict:
    """Export actual 120-tree research artifacts; do not train or truncate donors."""
    reference = research / "reports/xgb15_gpu_expanding_120_depth4_lr005_2022_2026"
    metadata = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    records = [record for record in metadata["model_records"] if record["prediction_year"] == year]
    if len(records) != 1 or metadata["num_boost_round"] != 120:
        raise ValueError("expected one reference record and exactly 120 training rounds")
    table = research / f"data/model_tables/factors15_2021_2026/predict_table_2022-2026/predict_table_{year}.parquet"
    pairs = validate_pairs(sorted(pd.read_parquet(table, columns=["symbol"]).symbol.unique().tolist()))
    model = reference / str(year) / "model.ubj"
    record = {**records[0], "file": f"model_{year}.ubj", "pairs": pairs,
              "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    from src.strategies.xgb15 import FEATURES

    if metadata["features"] != list(FEATURES):
        raise ValueError("research feature order differs from the registered strategy")
    frames = []
    for path in (research / "data/fundamentals/cmc_2021_2025/contract_daily.parquet",
                 research / "data/fundamentals/cmc_2026_supplement/contract_market_cap_2026.parquet"):
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError("no research CMC daily observations found")
    caps = pd.concat(frames, ignore_index=True)
    caps["symbol"] = caps.contract.where(caps.contract.str.contains("/"), caps.contract.str.replace(r"USDT$", "/USDT:USDT", regex=True))
    caps = normalize_caps(caps.loc[caps.symbol.isin(pairs), ["symbol", "date", "cmc_id", "market_cap_usd", "marketcap_identity_error"]])
    mapping = research / "data/fundamentals/cmc_2021_2025/asset_mapping.json"
    if mapping.exists():
        read_mapping(mapping, pairs)
    if output.exists() or output.is_symlink():
        raise ValueError("refusing to overwrite an exported package")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xgb15-export-", dir=output.parent) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        shutil.copyfile(model, stage / record["file"])
        manifest = {"strategy_id": STRATEGY_ID, "features": list(FEATURES), "preprocessing": "percentile_average",
                    "models": {str(year): record}, "params": metadata["params"], "rounds": 120,
                    "xgboost_version": metadata["xgboost_version"],
                    "source": "research:analysis/gpu_xgb15_expanding_120_lr005.py",
                    "source_manifest_sha256": hashlib.sha256((reference / "manifest.json").read_bytes()).hexdigest()}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (stage / "pairs.json").write_text(json.dumps(pairs, indent=2))
        caps.to_parquet(stage / "caps.parquet", index=False)
        if mapping.exists():
            shutil.copyfile(mapping, stage / mapping.name)
        validate_bundle(stage, year)
        stage.rename(output)
    return {"exported": True, "year": year, "pair_count": len(pairs), "worker_started": False,
            "source_modified": False, "cap_refresh_mapping_included": mapping.exists()}
