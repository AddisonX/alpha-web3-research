"""Recover 120-tree prefixes and verify every historical decile before importing.

Usage: python -m scripts.import_xgb15_research --research-root /private/research
No training, network requests, trade database writes, or worker startup.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from src.strategies.xgb15 import FEATURES, STRATEGY_ID, buffered_targets
from src.live.instances import prepare_instance, runtime_status
from src.live.signal_store import SignalStore
from src.strategies.xgb15 import XGB15LongShortStrategy
from src.strategies.base import StrategyContext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    research = args.research_root
    reference = research / "reports/xgb15_gpu_expanding_120_depth4_lr005_2022_2026"
    donor = research / "reports/xgb15_gpu_expanding_200_depth4_lr005_2022_2026"
    original = json.loads((reference / "manifest.json").read_text())
    other = json.loads((donor / "manifest.json").read_text())
    assert original["params"] == other["params"]
    assert original["features"] == other["features"] == list(FEATURES)
    assert original["xgboost_version"] == other["xgboost_version"] == xgb.__version__
    model_dir = root / f"data/models/strategies/{STRATEGY_ID}/v1"
    model_dir.mkdir(parents=True, exist_ok=True)
    private = root / "data/imports/xgb15"
    private.mkdir(parents=True, exist_ok=True)
    ref = pd.read_parquet(reference / "all_years_decile_daily.parquet")
    models = {}
    base = research / "data/model_tables/factors15_2021_2026"
    for year in range(2022, 2027):
        a = next(r for r in original["model_records"] if r["prediction_year"] == year)
        b = next(r for r in other["model_records"] if r["prediction_year"] == year)
        assert a["training_rows"] == b["training_rows"]
        model = xgb.Booster()
        model.load_model(donor / str(year) / "model.ubj")
        model = model[:120]
        model.set_param({"device": "cpu", "nthread": 4})
        data = pd.read_parquet(base / f"predict_table_2022-2026/predict_table_{year}.parquet")
        pairs = sorted(data.symbol.unique())
        forecast = data.loc[np.isfinite(data[list(FEATURES)]).all(axis=1) & data.pool_known_at_decision].sort_values(["entry_time_utc", "symbol"]).copy()
        forecast["score"] = model.predict(xgb.DMatrix(forecast[list(FEATURES)], feature_names=list(FEATURES)))
        checked = 0
        previous = None
        for stamp, frame in forecast.groupby("entry_time_utc", sort=True):
            saved_previous = previous
            previous = buffered_targets(frame.symbol.tolist(), frame.score.to_numpy(), previous)
            daily_ref = ref.loc[(ref.prediction_year == year) & (ref.decision_timestamp == int(stamp.timestamp()*1000))].sort_values("decile")
            if daily_ref.empty:
                continue
            groups = np.array_split(np.argsort(frame.score.to_numpy(), kind="stable"), 10)
            actual = [frame.iloc[ix].excess_return_24h.mean() for ix in groups]
            np.testing.assert_allclose(actual, daily_ref.excess_return.to_numpy(), atol=1e-12, rtol=0)
            ic = frame.score.corr(frame.excess_return_24h, method="spearman")
            np.testing.assert_allclose(ic, daily_ref.rank_ic.iloc[0], atol=1e-12, rtol=0)
            checked += 1
            if year == 2026:
                snapshot = frame[["symbol", *FEATURES]].copy()
                snapshot_stamp = int(stamp.timestamp()*1000) - 14_400_000
                snapshot_previous = saved_previous
        assert checked == len(ref.loc[ref.prediction_year == year, "decision_timestamp"].unique())
        path = model_dir / f"model_{year}.ubj"
        model.save_model(path)
        models[str(year)] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pairs": pairs, "fit_cutoff_utc": a["fit_cutoff_utc"], "training_years": a["training_years"],
            "training_rows": a["training_rows"], "validated_daily_deciles": checked,
            "provenance": "first 120 trees of same-parameter GPU 200-tree fit; all reference daily deciles and Rank IC matched at 1e-12"}
        print(year, "verified days", checked, flush=True)
    manifest = {"strategy_id": STRATEGY_ID, "features": list(FEATURES), "preprocessing": "percentile_average",
        "models": models, "xgboost_version": xgb.__version__, "params": original["params"], "rounds": 120,
        "label": "BJT 00:00 to next BJT 00:00 return minus fixed annual pool available-return equal weight mean",
        "missing": "rank each available factor independently; predict only complete rows; never filter predictions by outcomes",
        "annual_pool": "target-year pool, selected using complete previous-year history; known from January 1 UTC",
        "portfolio": {"long": .5, "short": -.5, "entry_fraction": .1, "retention_fraction": .3,
                      "rebalance": "daily equal weight BJT 00:00", "fee": .0005},
        "limitations": original["limitations"]}
    (model_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (private / "pairs.json").write_text(json.dumps(models["2026"]["pairs"], indent=2))
    caps = pd.concat([pd.read_parquet(research / "data/fundamentals/cmc_2021_2025/contract_daily.parquet"),
                      pd.read_parquet(research / "data/fundamentals/cmc_2026_supplement/contract_market_cap_2026.parquet")], ignore_index=True)
    caps["symbol"] = caps.contract.where(caps.contract.str.contains("/"), caps.contract.str.replace(r"USDT$", "/USDT:USDT", regex=True))
    caps["date"] = pd.to_datetime(caps.date)
    caps[["symbol", "date", "cmc_id", "market_cap_usd", "marketcap_identity_error"]].to_parquet(private / "caps.parquet", index=False)
    snapshot.to_parquet(private / "acceptance_features.parquet", index=False)
    from src.factors.xgb15 import build_xgb15_features
    bars = pd.read_parquet(research / "reports/alpha191_16/bars.parquet")
    bars = bars.loc[bars.symbol.isin(models["2026"]["pairs"]) & (bars.timestamp <= snapshot_stamp)
                    & (bars.timestamp >= snapshot_stamp-63*14_400_000) & (bars.amount > 0)]
    recomputed = build_xgb15_features(bars, caps, snapshot_stamp, models["2026"]["pairs"]).set_index("symbol")
    expected_features = snapshot.set_index("symbol")
    np.testing.assert_allclose(recomputed.loc[expected_features.index, list(FEATURES)].to_numpy(),
                               expected_features[list(FEATURES)].to_numpy(), atol=1e-12, rtol=0)
    strategy = XGB15LongShortStrategy(model_dir)
    context = StrategyContext(snapshot_stamp, pl.from_pandas(snapshot), True, snapshot_previous)
    decision = strategy.decide(context)
    instance = prepare_instance(root, private / "pairs.json", strategy_id=STRATEGY_ID, instance_name="xgb15-paper")
    store = SignalStore(instance / "signals/alpha_live_long_only_signals.parquet", instance / "signals/alpha_live_long_only_status.json")
    store.publish(STRATEGY_ID, decision, strategy.requirements, strategy.metadata, "historical_import_acceptance", len(models["2026"]["pairs"]), {}, 24)
    status = store.read_status()
    status.update({"state": "offline_verified", "reason": "Historical acceptance only. Worker stopped; fresh public OHLCV and cap coverage required."})
    store.status_path.write_text(json.dumps(status, indent=2))
    runtime_status(instance, "stopped", worker_started=False, entry_allowed=False, reason=status["reason"])
    docs = root / "docs/strategies" / STRATEGY_ID
    docs.mkdir(parents=True, exist_ok=True)
    public_manifest = {**manifest, "models": {y: {k:v for k,v in r.items() if k != "pairs"} for y,r in models.items()}}
    (docs / "manifest.json").write_text(json.dumps(public_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Imported verified private weights; paper worker remains stopped.", flush=True)


if __name__ == "__main__":
    main()
