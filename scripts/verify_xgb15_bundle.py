"""Verify the published XGB15 assets and reproduce the historical acceptance case.

Run: python -m scripts.verify_xgb15_bundle
Offline inference only: no training, downloads, signal publication or trading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.factors.xgb15 import build_xgb15_features
from src.strategies.base import StrategyContext, validate_signed_weights
from src.strategies.xgb15 import FEATURES, STRATEGY_ID, XGB15LongShortStrategy


def verify(root: Path) -> dict:
    root = root.resolve()
    inputs = root / "data/imports/xgb15"
    bundle = json.loads((inputs / "bundle_manifest.json").read_text(encoding="utf-8"))
    if bundle.get("strategy_id") != STRATEGY_ID or bundle.get("schema_version") != 1:
        raise ValueError("unsupported bundle manifest")
    for name, record in bundle["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("artifact path escapes project")
        raw = path.read_bytes()
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {name}")

    model_dir = root / "data/models/strategies" / STRATEGY_ID / "v1"
    strategy = XGB15LongShortStrategy(model_dir)
    for year, record in strategy.manifest["models"].items():
        if len(record["pairs"]) != len(set(record["pairs"])):
            raise ValueError("duplicate annual pool symbols")
        strategy._model(int(year))  # Verify weight hashes, feature order and 120-tree count.
    pairs = json.loads((inputs / "pairs.json").read_text(encoding="utf-8"))
    if len(pairs) != len(set(pairs)) or set(pairs) != set(strategy.manifest["models"]["2026"]["pairs"]):
        raise ValueError("runtime pair list differs from complete 2026 pool")
    mapping = json.loads((inputs / "asset_mapping.json").read_text(encoding="utf-8"))
    mapped = {r["contract"] for r in mapping if r.get("cmc_ids")}
    if not set(pairs).issubset({r["contract"] for r in mapping}):
        raise ValueError("asset mapping does not cover the annual pool")

    meta = json.loads((inputs / "acceptance.json").read_text(encoding="utf-8"))
    stamp = int(meta["decision_timestamp"])
    signal = pd.Timestamp(meta["signal_time_utc"])
    if stamp + 14_400_000 != int(signal.timestamp()*1000):
        raise ValueError("signal is not at decision candle close")
    if int(meta["previous_decision_timestamp"]) != stamp - 86_400_000:
        raise ValueError("previous targets are not from the prior daily decision")

    features = pl.read_parquet(inputs / "acceptance_features.parquet").sort("symbol")
    previous = pl.read_parquet(inputs / "acceptance_previous_targets.parquet")
    if previous["pair"].n_unique() != previous.height or not set(previous["pair"]).issubset(pairs):
        raise ValueError("invalid previous target universe")
    validate_signed_weights(previous)
    expected = pd.read_parquet(inputs / "acceptance_expected_predictions.parquet").sort_values("pair").reset_index(drop=True)
    context = StrategyContext(stamp, features, True, previous)
    actual = strategy.decide(context).targets.to_pandas().sort_values("pair").reset_index(drop=True)
    if actual.pair.tolist() != expected.pair.tolist() or len(actual) != meta["prediction_rows"]:
        raise ValueError("prediction universe differs from reference")
    np.testing.assert_allclose(actual.model_score, expected.model_score, atol=meta["prediction_atol"], rtol=0)
    np.testing.assert_array_equal(actual.target_position, expected.target_position)
    np.testing.assert_allclose(actual.target_weight, expected.target_weight, atol=meta["weight_atol"], rtol=0)

    caps = pd.read_parquet(inputs / "caps.parquet")
    bars = pd.read_parquet(inputs / "acceptance_bars.parquet")
    if bars.timestamp.max() > stamp:
        raise ValueError("acceptance OHLCV contains future bars")
    raw_ranks = build_xgb15_features(bars, caps, stamp, pairs).set_index("symbol")
    complete = raw_ranks.dropna(subset=list(FEATURES)).sort_index()
    reference = features.to_pandas().set_index("symbol").sort_index()
    if complete.index.tolist() != reference.index.tolist():
        raise ValueError("raw input reconstruction yields a different prediction universe")
    np.testing.assert_allclose(complete[list(FEATURES)], reference[list(FEATURES)], atol=meta["feature_atol"], rtol=0)
    latest = pd.to_datetime(caps.date).max()
    cap_only = build_xgb15_features(bars.iloc[:0], caps, int((latest+pd.Timedelta(days=2, hours=12)).tz_localize("UTC").timestamp()*1000), pairs)
    covered = int(cap_only[list(FEATURES[:4])].notna().all(axis=1).sum())
    if covered < 60:
        raise ValueError("fewer than 60 pool assets have complete stable-identity 40-day cap history")
    return {
        "status": "passed", "verified_files": len(bundle["files"]),
        "models": len(strategy.manifest["models"]), "annual_pool_members": len(pairs),
        "mapped_pool_members": len(set(pairs) & mapped),
        "pool_members_without_token_market_cap_id": sorted(set(pairs) - mapped),
        "caps_first_date": str(pd.to_datetime(caps.date).min().date()), "caps_last_date": str(latest.date()),
        "complete_40_day_cap_assets_at_last_date": covered,
        "signal_time_beijing": meta["signal_time_beijing"], "prediction_rows": len(actual),
        "long_count": int((actual.target_position == 1).sum()), "short_count": int((actual.target_position == -1).sum()),
        "max_prediction_difference": float(np.max(np.abs(actual.model_score - expected.model_score))),
        "max_feature_difference": float(np.max(np.abs(complete[list(FEATURES)].to_numpy() - reference[list(FEATURES)].to_numpy()))),
        "max_weight_difference": float(np.max(np.abs(actual.target_weight - expected.target_weight))),
        "training_performed": False, "trading_started": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    print(json.dumps(verify(parser.parse_args().root), ensure_ascii=False, indent=2))
