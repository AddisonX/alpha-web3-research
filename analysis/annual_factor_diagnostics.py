"""Retrospective annual factor diagnostics for the already screened annual pools."""
from __future__ import annotations

import calendar
import argparse
import hashlib
import html
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import PercentFormatter
from plotly.offline import get_plotlyjs
from scipy.stats import rankdata

from alpha_web3_research.config import Settings
from alpha_web3_research.data.binance_archive import BAR_MS, BinanceArchiveClient, parse_archive
from alpha_web3_research.data.ohlcv import OHLCV_COLUMNS
from alpha_web3_research.factors.builder import FactorBuilder
from alpha_web3_research.factors.builder import GROUP_COLUMNS
from alpha_web3_research.factors.turnover import build_turnover_factors

ROOT = Path("/workspace/alpha-web3-research")
OUTPUT = ROOT / "reports/annual_factor_diagnostics_2021_2025_midnight"
WORK = Path("/tmp/alpha-annual-factor-diagnostics")
DAY_MS = 86_400_000
LOCAL_OFFSET = 8 * 3_600_000
FEATURE_FILE = ROOT / "strategies/features/pit_rank64.v1.json"
NAMES = json.loads(FEATURE_FILE.read_text())["features"]
IS_TURNOVER = False
INDEX_PATH = ROOT / "data/research/universe_365d_20250101/archive_index.json"
INDEX = json.loads(INDEX_PATH.read_text())["monthly_keys"]
CLIENT = BinanceArchiveClient(Path("/tmp/alpha-universe-365/archives"), workers=16)


def utc_ms(year: int, month: int = 1, day: int = 1) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)


def local_date(stamp: int) -> str:
    return datetime.fromtimestamp((stamp + LOCAL_OFFSET) / 1000, timezone.utc).strftime("%Y-%m-%d")


def pool_path(pool_year: int) -> Path:
    if pool_year == 2025:
        return ROOT / "data/research/universe_full_2024/selected.csv"
    return ROOT / f"data/research/annual_universes/{pool_year}/selected.csv"


def build_symbol(native: str, year: int) -> dict:
    destination = WORK / str(year) / f"{native}.parquet"
    provenance_path = destination.with_suffix(".json")
    if destination.exists() and provenance_path.exists():
        return json.loads(provenance_path.read_text())
    lower_month = f"{year - 1}-{'11' if IS_TURNOVER else '12'}"
    upper_month = f"{year}-12"
    keys = [key for key in INDEX[native] if lower_month <= key[-11:-4] <= upper_month]
    frames, sources = [], []
    for key in keys:
        content, source = CLIENT.download_key(key)
        frames.append(parse_archive(content, native, key))
        sources.append(source)
    if not frames:
        raise ValueError(f"no source archives for {native}/{year}")
    bars = pl.concat(frames).unique(subset="timestamp").sort("timestamp")
    start, end = utc_ms(year) - LOCAL_OFFSET, utc_ms(year + 1) - LOCAL_OFFSET
    valid_bars = bars.filter((pl.col("quote_volume") > 0) & (pl.col("volume") > 0))
    valid_bars = valid_bars.with_columns(
        (pl.col("timestamp").diff().fill_null(BAR_MS) != BAR_MS).cum_sum().alias("segment")
    )
    feature_parts = []
    builder = FactorBuilder(Settings(project_root=ROOT))
    for segment in valid_bars.partition_by("segment", maintain_order=True):
        if segment.height < 25:
            continue
        first_ready = segment["timestamp"][24]
        if IS_TURNOVER:
            wide = build_turnover_factors(segment, GROUP_COLUMNS).filter(
                (pl.col("timestamp") >= first_ready)
                & (((pl.col("timestamp") + BAR_MS + LOCAL_OFFSET) % DAY_MS) == 0)
            )
            feature_parts.append(wide.with_columns((pl.col("timestamp") + BAR_MS).alias("decision_timestamp")))
            continue
        long = builder._build_factor_long_frame(segment.select(OHLCV_COLUMNS)).filter(
            pl.col("factor_name").is_in(NAMES)
            & (pl.col("timestamp") >= first_ready)
            & (((pl.col("timestamp") + BAR_MS + LOCAL_OFFSET) % DAY_MS) == 0)
        )
        if long.is_empty():
            continue
        wide = long.pivot(index=["symbol", "timestamp"], on="factor_name", values="factor_value")
        feature_parts.append(wide.with_columns((pl.col("timestamp") + BAR_MS).alias("decision_timestamp")))
    if not feature_parts:
        raise ValueError(f"no features for {native}/{year}")
    features = pl.concat(feature_parts, how="vertical_relaxed").filter(
        (pl.col("decision_timestamp") >= start) & (pl.col("decision_timestamp") < end)
    )
    # Seven exact-clock, positive-volume bars cover entry, holding interval, and exit.
    forward = valid_bars.sort("timestamp").with_columns(
        pl.col("timestamp").shift(-6).alias("exit_timestamp"),
        pl.col("open").shift(-6).alias("exit_open"),
    ).filter(pl.col("exit_timestamp") == pl.col("timestamp") + DAY_MS).select(
        pl.col("timestamp").alias("decision_timestamp"),
        (pl.col("exit_open") / pl.col("open") - 1.0).alias("forward_return"),
    )
    dataset = features.join(forward, on="decision_timestamp", how="left")
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(destination)
    record = {"native": native, "year": year, "rows": dataset.height, "sources": sources,
              "last_source_bar_utc_ms": int(bars["timestamp"].max())}
    provenance_path.write_text(json.dumps(record))
    return record


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    result = float(np.corrcoef(x, y)[0, 1])
    return result if np.isfinite(result) else None


def moments(values: list[float | None]) -> tuple[float | None, float | None, float | None, int]:
    finite = np.array([value for value in values if value is not None and np.isfinite(value)])
    if len(finite) < 2:
        return None, None, None, len(finite)
    mean, std = float(finite.mean()), float(finite.std(ddof=1))
    return mean, std, mean / std if std > 0 else None, len(finite)


def analyze_year(year: int) -> dict:
    pool_year = year + 1
    year_dir = OUTPUT / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = year_dir / "manifest.json"
    if manifest_path.exists():
        print(f"Reusing completed diagnostics {year}", flush=True)
        return json.loads(manifest_path.read_text())
    source_pool = pool_path(pool_year)
    symbols = pl.read_csv(source_pool)["symbol"].sort().to_list()
    natives = [symbol.split("/")[0] + "USDT" for symbol in symbols]
    print(f"YEAR {year}, POOL {pool_year}: acquire and build {len(symbols)} symbols", flush=True)
    records, errors = [], []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(build_symbol, native, year): native for native in natives}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                records.append(future.result())
            except Exception as error:
                errors.append({"native": futures[future], "error": repr(error)})
            if number % 20 == 0 or number == len(natives):
                print(f"YEAR {year}: {number}/{len(natives)} symbols, errors={len(errors)}", flush=True)
    (year_dir / "acquisition.json").write_text(json.dumps({"records": records, "errors": errors}, indent=2))
    if errors:
        raise RuntimeError(f"data errors for {year}: {errors[:3]}")
    dataset = pl.concat([pl.read_parquet(WORK / str(year) / f"{native}.parquet") for native in natives],
                        how="vertical_relaxed")
    assert dataset.select(pl.struct("symbol", "decision_timestamp").n_unique()).item() == dataset.height
    dataset.write_parquet(year_dir / "dataset.parquet")
    finite = dataset.filter(pl.all_horizontal([
        pl.col(name).is_finite().fill_null(False) for name in [*NAMES, "forward_return"]
    ]))
    counts = finite.group_by("decision_timestamp").len()
    complete_dates = counts.filter(pl.col("len") == len(symbols))["decision_timestamp"].to_list()
    start, end = utc_ms(year) - LOCAL_OFFSET, utc_ms(year + 1) - LOCAL_OFFSET
    expected_dates = list(range(start, end, DAY_MS))
    excluded = sorted(set(expected_dates) - set(complete_dates))
    usable = finite.filter(pl.col("decision_timestamp").is_in(complete_dates)).sort(["decision_timestamp", "symbol"])
    if not complete_dates:
        raise ValueError(f"no full-pool complete dates for {year}")
    days = list(usable.partition_by("decision_timestamp", maintain_order=True))
    daily_rows, metric_rows, coverage_rows = [], [], []
    for stamp in expected_dates:
        coverage_rows.append({"decision_beijing": local_date(stamp), "decision_timestamp": stamp,
                              "included": stamp in complete_dates,
                              "finite_full_feature_and_label_members": finite.filter(pl.col("decision_timestamp") == stamp).height,
                              "pool_members": len(symbols)})
    pl.DataFrame(coverage_rows).write_csv(year_dir / "date_coverage.csv")
    for factor in NAMES:
        daily_ic, daily_rank_ic = [], []
        factor_rows = []
        for frame in days:
            stamp = int(frame["decision_timestamp"][0])
            x = frame[factor].to_numpy()
            returns = frame["forward_return"].to_numpy()
            benchmark = float(returns.mean())
            excess = returns - benchmark
            ic = correlation(x, excess)
            rank_ic = correlation(rankdata(x, method="average"), rankdata(excess, method="average"))
            daily_ic.append(ic)
            daily_rank_ic.append(rank_ic)
            ordered = np.argsort(x, kind="stable")  # input symbols are ascending, including ties
            for decile, indices in enumerate(np.array_split(ordered, 10), 1):
                group_return = float(returns[indices].mean())
                factor_rows.append({"data_year": year, "pool_year": pool_year, "factor": factor,
                                    "decision_timestamp": stamp, "decision_beijing": local_date(stamp),
                                    "exit_timestamp": stamp + DAY_MS, "decile": decile,
                                    "members": len(indices), "group_return": group_return,
                                    "benchmark_return": benchmark, "excess_return": group_return - benchmark,
                                    "ic": ic, "rank_ic": rank_ic})
            # Unequal count deciles must reconcile using member counts, not 1/10 weights.
            last = factor_rows[-10:]
            assert np.isclose(sum(row["excess_return"] * row["members"] for row in last), 0, atol=1e-10)
        mean_ic, std_ic, icir, ic_days = moments(daily_ic)
        mean_rank_ic, std_rank_ic, rank_icir, rank_ic_days = moments(daily_rank_ic)
        row = {"data_year": year, "pool_year": pool_year, "pool_size": len(symbols), "factor": factor,
               "mean_ic": mean_ic, "std_ic": std_ic, "icir": icir, "ic_days": ic_days,
               "mean_rank_ic": mean_rank_ic, "std_rank_ic": std_rank_ic, "rank_icir": rank_icir,
               "rank_ic_days": rank_ic_days, "valid_return_days": len(days)}
        for decile in range(1, 11):
            row[f"g{decile}_excess_simple"] = sum(r["excess_return"] for r in factor_rows if r["decile"] == decile)
        metric_rows.append(row)
        daily_rows.extend(factor_rows)
    metrics = pl.DataFrame(metric_rows)
    daily = pl.DataFrame(daily_rows).sort(["factor", "decile", "decision_timestamp"]).with_columns(
        pl.col("excess_return").cum_sum().over("factor", "decile").alias("cumulative_excess_simple_return")
    )
    assert daily.height == len(NAMES) * 10 * len(days)
    metrics.write_csv(year_dir / "metrics.csv")
    daily.write_parquet(year_dir / "decile_daily.parquet")
    manifest = {"data_year": year, "pool_year": pool_year, "pool_members": len(symbols),
                "expected_days": len(expected_dates), "valid_days": len(days),
                "excluded_dates_beijing": [local_date(stamp) for stamp in excluded],
                "pool_file": str(source_pool), "pool_sha256": hashlib.sha256(source_pool.read_bytes()).hexdigest(),
                "feature_file": str(FEATURE_FILE.relative_to(ROOT)), "factor_count": len(NAMES),
                "features_sha256": hashlib.sha256(FEATURE_FILE.read_bytes()).hexdigest(),
                "decision_time": "00:00 Asia/Shanghai", "holding_hours": 24,
                "ic": "Pearson correlation of raw factor and next-24h excess return, daily cross-section",
                "rank_ic": "Spearman correlation, average ranks for ties",
                "icir": "mean daily IC / sample standard deviation ddof=1; not annualized",
                "grouping": "ascending factor, ascending symbol tie-break, numpy.array_split into 10 groups",
                "benchmark": "all fixed pool members equal-weight, identical 24h open-to-open return window",
                "return_aggregation": "sum of daily group return minus daily benchmark; no compounding",
                "incomplete_dates": "exclude entire day if any pool member lacks any requested finite factor or a complete return window",
                "costs": "gross, no fees, funding, slippage or stops",
                "retrospective": "pool chosen using the analyzed full year" + ("; new turnover definitions" if IS_TURNOVER else "; features previously selected using July 2026 data"),
                "factor_formulas": "relative turnover proxies from base volume and closed prices" if IS_TURNOVER else "original FactorBuilder; its quote-volume factors use close * base volume proxy"}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"YEAR {year}: diagnostics done, {len(days)}/{len(expected_dates)} dates", flush=True)
    return manifest


def render_reports(manifests: list[dict]) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = [plt.get_cmap("RdYlBu_r")(value) for value in np.linspace(.05, .95, 10)]
    all_metrics, payload = [], {}
    for manifest in manifests:
        year = manifest["data_year"]
        year_dir = OUTPUT / str(year)
        chart_dir = year_dir / "charts"
        chart_dir.mkdir(exist_ok=True)
        metrics = pl.read_csv(year_dir / "metrics.csv")
        daily = pl.read_parquet(year_dir / "decile_daily.parquet")
        all_metrics.append(metrics)
        payload[str(year)] = {}
        for row in metrics.iter_rows(named=True):
            factor = row["factor"]
            part = daily.filter(pl.col("factor") == factor)
            ordered_dates = sorted(part["decision_timestamp"].unique().to_list())
            full_dates = list(range(utc_ms(year) - LOCAL_OFFSET, utc_ms(year + 1) - LOCAL_OFFSET, DAY_MS))
            dates = [datetime.fromtimestamp((stamp + DAY_MS + LOCAL_OFFSET) / 1000, timezone.utc) for stamp in full_dates]
            curves = []
            fig, ax = plt.subplots(figsize=(11.5, 5.3), layout="constrained")
            for decile, color in enumerate(colors, 1):
                group = part.filter(pl.col("decile") == decile)
                lookup = dict(group.select("decision_timestamp", "cumulative_excess_simple_return").iter_rows())
                values = [lookup.get(stamp) for stamp in full_dates]
                ax.plot(dates, values, label=f"G{decile}", color=color, linewidth=1.25)
                curves.append(values)
            ax.axhline(0, color="#606875", linewidth=.7)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(axis="y", alpha=.18)
            ax.set_ylabel("Cumulative excess simple return")
            ax.set_xlabel("Exit date, Beijing time (rebalance daily at 00:00)")
            ax.set_title(f"{factor} | data {year}, pool {year + 1} (N={manifest['pool_members']})\n"
                         f"IC {row['mean_ic']:+.4f} | ICIR {row['icir']:+.4f} | "
                         f"RankIC {row['mean_rank_ic']:+.4f} | RankICIR {row['rank_icir']:+.4f}\n"
                         f"Full-pool equal-weight benchmark | {manifest['valid_days']}/{manifest['expected_days']} days | gross", fontsize=11)
            ax.legend(ncol=10, loc="upper left", fontsize=8, frameon=False)
            fig.savefig(chart_dir / f"{factor}.png", dpi=135)
            plt.close(fig)
            payload[str(year)][factor] = {"metrics": row, "dates": [date.strftime("%Y-%m-%d") for date in dates], "curves": curves}
        print(f"YEAR {year}: rendered {len(NAMES)} charts", flush=True)
    combined = pl.concat(all_metrics)
    combined.write_csv(OUTPUT / "all_years_metrics.csv")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifests, indent=2))
    # An offline report lets the reader select all 320 factor/year combinations.
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>64因子年度分层分析</title>
<style>body{font:15px system-ui;margin:24px;color:#172033}select{padding:8px;margin:8px}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}.note{max-width:1100px;line-height:1.7;color:#495368}#chart{height:620px}</style>
<h1>64因子年度 IC 与十档累计超额单利收益</h1>
<p class="note">数据年2021—2025，对应币池年2022—2026。北京时间每日00:00调仓，持有24小时。G1为因子最低档，G10为最高档，保留原始因子方向。每档等权收益减去同日固定全币池等权收益后逐日相加。IC使用原始因子的Pearson相关，RankIC使用Spearman相关；两类IR均不年化。收益为毛收益。</p>
<p class="note">币池由所分析全年数据筛选，64因子曾使用2026年信息选取，结果属于回看诊断。若任一币缺少完整特征或收益，整日剔除并在曲线上留空；日期覆盖详见各年date_coverage.csv。因子公式沿用原项目，其中成交额类因子内部采用收盘价乘成交量的近似值。</p>
<label>因子<select id="factor"></select></label><label>数据年份<select id="year"></select></label><span id="coverage"></span><div id="chart"></div>
<h2>当前因子的年度指标及十档累计超额单利收益</h2><div id="table"></div>
<script>__PLOTLY__</script><script>const data=__DATA__;const names=__NAMES__;const colors=__COLORS__;
const f=document.getElementById('factor'),y=document.getElementById('year');names.forEach(n=>f.add(new Option(n,n)));Object.keys(data).forEach(n=>y.add(new Option(n,n)));
function number(v){return v==null?'n.a.':v.toFixed(4)}function pct(v){return v==null?'n.a.':(100*v).toFixed(2)+'%'}
function update(){const p=data[y.value][f.value],m=p.metrics;document.getElementById('coverage').textContent=`币池 ${m.pool_year}，${m.pool_size} 个币，有效 ${m.valid_return_days} 天`;
Plotly.react('chart',p.curves.map((v,i)=>({x:p.dates,y:v,type:'scatter',mode:'lines',name:'G'+(i+1),connectgaps:false,line:{color:colors[i],width:1.5}})),{title:`${f.value} · ${y.value}年数据 · IC ${number(m.mean_ic)} / ICIR ${number(m.icir)}`,yaxis:{title:'累计超额单利收益',tickformat:'.0%'},xaxis:{title:'收益兑现日期（北京时间）'},legend:{orientation:'h'},margin:{l:80,r:20,t:60,b:60}},{responsive:true});
let t='<table><tr><th>数据年 / 币池年</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>有效天数</th>'+Array.from({length:10},(_,i)=>'<th>G'+(i+1)+'</th>').join('')+'</tr>';
Object.keys(data).forEach(year=>{let r=data[year][f.value].metrics;t+=`<tr><td>${year} / ${r.pool_year}</td><td>${number(r.mean_ic)}</td><td>${number(r.icir)}</td><td>${number(r.mean_rank_ic)}</td><td>${number(r.rank_icir)}</td><td>${r.valid_return_days}</td>`+Array.from({length:10},(_,i)=>'<td>'+pct(r['g'+(i+1)+'_excess_simple'])+'</td>').join('')+'</tr>'});document.getElementById('table').innerHTML=t+'</table>'}
f.onchange=update;y.onchange=update;update();</script></html>"""
    color_hex = [matplotlib.colors.to_hex(color) for color in colors]
    if IS_TURNOVER:
        page = page.replace("64因子", "16个相对换手因子")
        page = page.replace("16个相对换手因子曾使用2026年信息选取，", "")
        page = page.replace("因子公式沿用原项目，其中成交额类因子内部采用收盘价乘成交量的近似值。", "因子由成交量与收盘价计算，采用完整窗口并在数据缺口后重新预热；不使用流通量或合约持仓量作分母。")
    page = page.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", json.dumps(payload, allow_nan=False))
    page = page.replace("__NAMES__", json.dumps(NAMES)).replace("__COLORS__", json.dumps(color_hex))
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    # A five-panel example preserves identical factor direction in every year.
    preview_factor = "return_1" if "return_1" in NAMES else NAMES[0]
    fig, axes = plt.subplots(5, 1, figsize=(12, 18), layout="constrained")
    for ax, manifest in zip(axes, manifests, strict=True):
        year = str(manifest["data_year"])
        item = payload[year][preview_factor]
        dates = [datetime.strptime(value, "%Y-%m-%d") for value in item["dates"]]
        for decile, (values, color) in enumerate(zip(item["curves"], colors, strict=True), 1):
            ax.plot(dates, values, color=color, label=f"G{decile}", linewidth=1)
        row = item["metrics"]
        ax.set_title(f"Data {year} / pool {int(year)+1} | N={row['pool_size']} | "
                     f"IC {row['mean_ic']:+.4f} | ICIR {row['icir']:+.4f} | valid days {row['valid_return_days']}", fontsize=11)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.axhline(0, color="gray", linewidth=.5)
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(ncol=10, fontsize=8, frameon=False)
    fig.suptitle(f"{preview_factor} | cumulative decile excess simple returns vs full-pool equal weight", fontsize=13)
    fig.savefig(OUTPUT / f"{preview_factor}_all_years.png", dpi=130)
    plt.close(fig)


def main() -> None:
    global NAMES, FEATURE_FILE, OUTPUT, WORK, IS_TURNOVER
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=("rank64", "turnover16"), default="rank64")
    args = parser.parse_args()
    if args.feature_set == "turnover16":
        IS_TURNOVER = True
        FEATURE_FILE = ROOT / "strategies/features/turnover_proxy16.v1.json"
        NAMES = json.loads(FEATURE_FILE.read_text())["features"]
        OUTPUT = ROOT / "reports/turnover16_2021_2025_midnight"
        WORK = Path("/tmp/alpha-turnover16-diagnostics")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifests = [analyze_year(year) for year in range(2021, 2026)]
    render_reports(manifests)
    print("ALL REPORTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
