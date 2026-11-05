"""Extend the frozen inception universe and diagnose its 64 factors, without training."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import polars as pl

from alpha_web3_research.config import Settings
from alpha_web3_research.data.binance_archive import BinanceArchiveClient, parse_archive, ARCHIVE_BASE, BAR_MS
from alpha_web3_research.data.time import parse_timestamp_ms
from alpha_web3_research.models.pit_rank_dataset import build_pit_rank_dataset
from build_factor_2026_curves import calculate, DAY_MS, ROOT, local_time
from build_factor_2026_deciles import compute, render

SOURCE = ROOT / "data/research/pit_rank64_202501_202610"
DATA = ROOT / "data/research/factor_deciles_20250101_20261031"
OUTPUT = ROOT / "reports/factors/rank64_20250101_20261031_20261105"
START = parse_timestamp_ms("2025-01-01T04:00:00Z")
END = parse_timestamp_ms("2026-11-01T04:00:00Z")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extend_symbol(path: Path, client: BinanceArchiveClient) -> dict:
    native = path.stem
    original = pl.read_parquet(path)
    pieces, archives, absent = [original], [], []
    for kind, date in [("monthly", "2026-10"), ("daily", "2026-11-01")]:
        key = f"data/futures/um/{kind}/klines/{native}/4h/{native}-4h-{date}.zip"
        try:
            content, provenance = client.download_key(key)
        except HTTPError as error:
            if error.code != 404:
                raise
            absent.append(key)
            continue
        extra = parse_archive(content, native, ARCHIVE_BASE + key)
        # Do not silently revise previously archived candles when extending their tail.
        shared = original.join(extra, on=["symbol", "timestamp"], suffix="_new")
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            if not np.allclose(shared[col], shared[f"{col}_new"], atol=1e-12, rtol=1e-12):
                raise ValueError(f"archive revision detected: {native}/{col}")
        pieces.append(extra.filter(pl.col("timestamp") > original["timestamp"].max()))
        archives.append(provenance)
    combined = pl.concat(pieces, how="vertical_relaxed").filter(pl.col("timestamp") <= END).sort("timestamp")
    assert combined["timestamp"].n_unique() == combined.height
    destination = DATA / "bars" / path.name
    combined.write_parquet(destination)
    return {"symbol": original["symbol"][0], "original_file": str(path), "original_sha256": sha(path),
            "extended_sha256": sha(destination), "rows": combined.height,
            "last_bar_open_ms": combined["timestamp"].max(), "new_archives": archives, "missing_archives_404": absent}


def prepare() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "bars").mkdir(exist_ok=True)
    universe = pl.read_parquet(SOURCE / "universe.parquet")
    included = universe.filter(pl.col("included"))
    assert included.height == 120
    paths = [SOURCE / "bars" / (s.split("/")[0] + "USDT.parquet") for s in included["symbol"]]
    assert all(p.exists() for p in paths)
    client = BinanceArchiveClient(ROOT / "data/raw/binance_archive", workers=8)
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(extend_symbol, p, client): p.stem for p in paths}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append({"native": futures[future], "error": repr(error)})
            if i % 10 == 0:
                print(f"Extended {i}/120 symbols; errors={len(errors)}", flush=True)
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "universe_source": str(SOURCE / "universe.parquet"),
                "universe_sha256": sha(SOURCE / "universe.parquet"), "results": results, "errors": errors,
                "decision_start_ms": START, "decision_end_exclusive_ms": END,
                "last_required_exit_ms": END, "source_selection_unchanged": True}
    (DATA / "acquisition.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("archive extension failed; inspect acquisition.json")
    if sum(r["last_bar_open_ms"] >= END for r in results) < 20:
        raise RuntimeError("insufficient final-day coverage")
    universe.write_parquet(DATA / "universe.parquet")
    print(f"Archive extension complete: {sum(r['last_bar_open_ms'] >= END for r in results)}/120 reach the final exit", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-data", action="store_true")
    args = parser.parse_args()
    if not args.reuse_data:
        prepare()
    if args.prepare_only:
        return
    acquisition = json.loads((DATA / "acquisition.json").read_text())
    assert not acquisition["errors"] and len(acquisition["results"]) == 120
    assert sha(SOURCE / "universe.parquet") == acquisition["universe_sha256"]
    for record in acquisition["results"]:
        path = DATA / "bars" / Path(record["original_file"]).name
        assert sha(path) == record["extended_sha256"]
    feature_file = ROOT / "strategies/features/pit_rank64.v1.json"
    names = json.loads(feature_file.read_text())["features"]
    bars = pl.concat([pl.read_parquet(p) for p in sorted((DATA / "bars").glob("*.parquet"))], how="vertical_relaxed")
    print(f"Building daily features from {bars.height:,} bars", flush=True)
    dataset, labels = build_pit_rank_dataset(bars, names, Settings(project_root=ROOT))
    frame = dataset.filter((pl.col("decision_timestamp") >= START) & (pl.col("decision_timestamp") < END))
    expected = list(range(START, END, DAY_MS))
    missing_features = sorted(set(expected) - set(frame["decision_timestamp"].unique().to_list()))
    if missing_features:
        print(f"Missing feature dates (left as gaps): {[local_time(t) for t in missing_features]}", flush=True)
    frame.write_parquet(DATA / "dataset.parquet")
    labels.write_parquet(DATA / "labels.parquet")
    # Old overlapping values must reproduce before interpreting the extended charts.
    old = ROOT / "data/models/pit_rank64_xgboost_daily_excess_long_only/pit_rank64_xgboost_daily_excess_long_only_20261023T032315096488Z/predictions.parquet"
    overlap = pl.read_parquet(old).join(frame, on=["symbol", "decision_timestamp"], suffix="_new")
    for name in names + ["forward_return", "universe_forward_return"]:
        valid = overlap.filter(pl.col(name).is_finite())
        assert np.allclose(valid[name], valid[f"{name}_new"], equal_nan=True, atol=1e-12), name
    print(f"Reproduced {overlap.height:,} archived rows; calculating 64 factor deciles", flush=True)
    metrics, _, incomplete = calculate(frame, names)
    exclusions = sorted(set(incomplete["decision_timestamp"].to_list()) | set(missing_features))
    usable = frame.filter(~pl.col("decision_timestamp").is_in(exclusions))
    daily, spread, summary = compute(usable, names, metrics)
    count = usable["decision_timestamp"].n_unique()
    assert count + len(exclusions) == len(expected)
    assert daily.height == 64 * 10 * count and spread.height == 64 * count
    summary = summary.join(metrics.select("factor_name", "rank_ic_days"), on="factor_name")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    daily.write_parquet(OUTPUT / "decile_daily.parquet")
    spread.write_parquet(OUTPUT / "spread_daily.parquet")
    summary.write_csv(OUTPUT / "summary.csv")
    yearly = []
    for year in [2025, 2026]:
        part = frame.filter(pl.from_epoch("decision_timestamp", time_unit="ms").dt.year() == year)
        year_metrics, _, _ = calculate(part, names)
        for row in year_metrics.to_dicts():
            name = row["factor_name"]
            d = daily.filter((pl.col("factor_name") == name) & (pl.from_epoch("decision_timestamp", time_unit="ms").dt.year() == year))
            row.update({"year": year, "g1_simple_return": d.filter(pl.col("decile") == 1)["return"].sum(),
                        "g10_simple_return": d.filter(pl.col("decile") == 10)["return"].sum()})
            row["raw_spread"] = row["g10_simple_return"] - row["g1_simple_return"]
            yearly.append(row)
    pl.DataFrame(yearly).write_csv(OUTPUT / "yearly_summary.csv")
    metadata = {
        "source_run_id": "fixed_inception_rank64_factor_diagnostics_20250101_20261031",
        "period_label": "2025-01 to 2026-10", "page_title": "2025.01–2026.10 因子十分组收益与 spread",
        "source_dataset": str(DATA / "dataset.parquet"), "source_dataset_sha256": sha(DATA / "dataset.parquet"),
        "universe_source": acquisition["universe_source"], "universe_sha256": acquisition["universe_sha256"],
        "feature_set_sha256": sha(feature_file), "feature_count": 64, "fixed_universe_count": 120,
        "eligible_count_min": usable.group_by("decision_timestamp").len()["len"].min(),
        "eligible_count_max": usable.group_by("decision_timestamp").len()["len"].max(),
        "start_beijing": local_time(START), "end_beijing": local_time(END),
        "last_decision_beijing": local_time(END - DAY_MS), "expected_daily_windows": len(expected),
        "valid_daily_windows": count, "excluded_decision_timestamps": exclusions,
        "missing_feature_decision_timestamps": missing_features,
        "actual_last_valid_decision_beijing": local_time(usable["decision_timestamp"].max()),
        "actual_last_valid_exit_beijing": local_time(usable["decision_timestamp"].max() + DAY_MS),
        "requested_interval_complete": not exclusions,
        "coverage_note": f"请求包含2026-10-31调仓，但数据存在缺口；最后有效调仓为{local_time(usable['decision_timestamp'].max())}，其收益于{local_time(usable['decision_timestamp'].max() + DAY_MS)}兑现。尾部缺失不填零、不延续净值。",
        "coverage_label": f"Last valid exit: {local_time(usable['decision_timestamp'].max() + DAY_MS)[:10]} | excluded windows: {len(exclusions)}",
        "excluded_observations": incomplete.to_dicts(), "old_prediction_overlap_rows_verified": overlap.height,
        "direction_source": "full-period mean RankIC, retrospective only", "group_count": 10,
        "split_rule": "numpy.array_split; ascending factor and symbol", "icir_annualized": False,
        "costs_included": False, "stop_loss": None, "spread_gross_exposure": 2.0,
        "caveats": ["64-factor selection previously used 2026 July data", "not a clean out-of-sample model backtest",
                    "whole incomplete-label dates excluded, never individual future-missing assets",
                    "last October 31 signal return realizes on November 1 at 12:00 Beijing"],
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rendering {count} valid days; excluded={len(exclusions)}", flush=True)
    render(OUTPUT, daily, spread, summary, expected, metadata)
    with (OUTPUT / "report.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## 本次扩展口径\n\n固定2025年初120成员，未重训模型。每天北京时间12:00，以当时完整64维特征排名分组；收益为未来24h开盘到开盘。\n\n")
        stream.write(f"覆盖2025-01-01至2026-10-31的决策，共{len(expected)}个计划窗口、{count}个有效窗口。10月31日的收益到11月1日12:00兑现，因此横轴延伸至11月1日。\n\n")
        stream.write(metadata["coverage_note"] + "\n\n")
        stream.write("10月月度4h归档缺少10月28日，断档后遵循原策略至少25根连续bar重新预热。BTC的同日4h/1h/1m日归档返回404；1h月归档也缺少当天，本机主网REST返回地区限制，未绕过访问限制。因而无法声称完整覆盖到10月31日。\n\n")
        stream.write("64因子选择历史使用过2026年信息，定向线也使用本期RankIC，仅为回看。年度表 yearly_summary.csv 分开列出2025/2026的RankIC、RankICIR、G1/G10和原始spread；年度方向各自事后确定，不应用来拼接交易策略。\n\n")
        stream.write("数据下载清单与官方CHECKSUM校验见独立数据目录 acquisition.json；原始历史行情、旧报告、模型及线上服务均未修改。\n")
    print(json.dumps({"output": str(OUTPUT), "valid_days": count, "excluded": exclusions,
                      "volatility_4": summary.filter(pl.col("factor_name") == "volatility_4").to_dicts()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
