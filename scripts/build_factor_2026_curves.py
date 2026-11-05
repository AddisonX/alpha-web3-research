"""Single-factor retrospective diagnostics on a frozen 2026 model dataset."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
from matplotlib.ticker import PercentFormatter
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
DAY_MS = 86_400_000
SHANGHAI = ZoneInfo("Asia/Shanghai")
COLORS = {"excess": "#007c78", "selected": "#a34e31", "benchmark": "#7b8490"}


def local_time(stamp: int) -> str:
    return datetime.fromtimestamp(stamp / 1000, timezone.utc).astimezone(SHANGHAI).strftime("%Y-%m-%d %H:%M")


def calculate(frame: pl.DataFrame, names: list[str]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if frame.select(pl.struct("symbol", "decision_timestamp").n_unique()).item() != frame.height:
        raise ValueError("duplicate symbol/decision rows")
    # Drop the whole incomplete date, not just a future-unlabelled asset from the selection pool.
    incomplete = frame.filter(~pl.all_horizontal([
        pl.col(name).is_finite().fill_null(False)
        for name in [*names, "forward_return", "excess_forward_return", "universe_forward_return"]
    ])).select("symbol", "decision_timestamp", "label_end_timestamp")
    excluded_dates = incomplete["decision_timestamp"].unique()
    usable = frame.filter(~pl.col("decision_timestamp").is_in(excluded_dates.implode()))
    if usable.is_empty():
        raise ValueError("no complete daily cross sections")
    groups = list(usable.sort(["decision_timestamp", "symbol"]).group_by("decision_timestamp", maintain_order=True))
    summaries, daily_rows = [], []
    for name in names:
        correlations = []
        for _, group in groups:
            if group[name].n_unique() < 2 or group["excess_forward_return"].n_unique() < 2:
                correlations.append(None)
                continue
            ranks = group.select(
                pl.col(name).rank(method="average").alias("factor_rank"),
                pl.col("excess_forward_return").rank(method="average").alias("return_rank"),
            )
            correlations.append(ranks.select(pl.corr("factor_rank", "return_rank")).item())
        finite = np.asarray([v for v in correlations if v is not None and np.isfinite(v)])
        mean_ic = float(finite.mean()) if len(finite) else None
        std_ic = float(finite.std(ddof=1)) if len(finite) > 1 else None
        icir = mean_ic / std_ic if std_ic and mean_ic is not None else None
        if mean_ic is None:
            raise ValueError(f"no defined daily RankIC for {name}")
        direction = 1 if mean_ic >= 0 else -1
        cumulative_selected = cumulative_benchmark = cumulative_excess = 0.0
        for ((timestamp,), group), rank_ic in zip(groups, correlations, strict=True):
            count = math.ceil(group.height * 0.1)
            selected = group.sort([name, "symbol"], descending=[direction == 1, False]).head(count)
            benchmark_values = group["universe_forward_return"].unique()
            if len(benchmark_values) != 1:
                raise ValueError("inconsistent source-universe benchmark")
            selected_return = float(selected["forward_return"].mean())
            benchmark_return = float(benchmark_values[0])
            excess = selected_return - benchmark_return
            cumulative_selected += selected_return
            cumulative_benchmark += benchmark_return
            cumulative_excess += excess
            daily_rows.append({
                "factor_name": name, "decision_timestamp": timestamp, "exit_timestamp": timestamp + DAY_MS,
                "direction_sign": direction, "rank_ic": rank_ic, "eligible_count": group.height,
                "label_universe_count": int(group["label_universe_count"][0]),
                "selected_count": count, "selected_symbols": selected["symbol"].to_list(),
                "selected_return": selected_return, "universe_return": benchmark_return, "excess_return": excess,
                "selected_cumulative_simple_return": cumulative_selected,
                "universe_cumulative_simple_return": cumulative_benchmark,
                "cumulative_excess_simple_return": cumulative_excess,
            })
        summaries.append({
            "factor_name": name, "direction_sign": direction, "mean_rank_ic": mean_ic,
            "std_rank_ic": std_ic, "rank_icir": icir, "rank_ic_days": len(finite),
            "daily_period_count": len(groups), "selected_cumulative_simple_return": cumulative_selected,
            "universe_cumulative_simple_return": cumulative_benchmark,
            "cumulative_excess_simple_return": cumulative_excess,
        })
        if not np.isclose(cumulative_selected - cumulative_benchmark, cumulative_excess, atol=1e-12):
            raise ValueError("excess return reconciliation failed")
    return pl.DataFrame(summaries).sort("cumulative_excess_simple_return", descending=True), pl.DataFrame(daily_rows), incomplete


def plot_values(daily: pl.DataFrame, name: str, all_dates: list[int]):
    rows = {row["decision_timestamp"]: row for row in daily.filter(pl.col("factor_name") == name).iter_rows(named=True)}
    x = [datetime.fromtimestamp(all_dates[0] / 1000, timezone.utc).astimezone(SHANGHAI)]
    x += [datetime.fromtimestamp((stamp + DAY_MS) / 1000, timezone.utc).astimezone(SHANGHAI) for stamp in all_dates]
    columns = ["cumulative_excess_simple_return", "selected_cumulative_simple_return", "universe_cumulative_simple_return"]
    ys = [[0.0] + [rows[t][column] if t in rows else np.nan for t in all_dates] for column in columns]
    return x, ys


def draw_axis(ax, row, x, ys):
    for y, key, label, style in zip(ys, COLORS, ["Excess", "Selected top/bottom 10%", "Universe equal weight"], ["-", "--", ":"], strict=True):
        ax.plot(x, y, color=COLORS[key], linewidth=1.4 if key == "excess" else 1.0, linestyle=style, label=label)
    direction = "TOP 10%" if row["direction_sign"] == 1 else "BOTTOM 10%"
    ax.set_title(f"{row['factor_name']}  |  {direction}\nRankIC {row['mean_rank_ic']:+.4f}  |  RankICIR {row['rank_icir']:+.4f}\nExcess {row['cumulative_excess_simple_return']:+.1%}", fontsize=9, loc="left", pad=7)
    ax.axhline(0, color="#a6adb3", linewidth=0.6)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3, tz=SHANGHAI))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b", tz=SHANGHAI))
    ax.grid(alpha=0.16)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def render(output: Path, summary: pl.DataFrame, daily: pl.DataFrame, all_dates: list[int], metadata: dict):
    images = output / "individual"
    images.mkdir(exist_ok=True)
    rows = summary.to_dicts()
    sections = []
    excluded_days = ", ".join(local_time(t)[:10] for t in metadata["excluded_decision_timestamps"]) or "none"
    for page in range(4):
        subset = rows[page * 16:(page + 1) * 16]
        fig, axes = plt.subplots(4, 4, figsize=(18, 14), layout="constrained")
        fig.suptitle(f"2026 Rank64 factor diagnostics | page {page + 1}/4 | {metadata['valid_daily_windows']} daily windows\n{metadata['start_beijing'][:10]} to {metadata['end_beijing'][:10]}; gross, retrospective IC direction; excluded: {excluded_days}", fontsize=14)
        titles = []
        for row in subset:
            side = "最高10%" if row["direction_sign"] == 1 else "最低10%"
            titles.append(f"{row['factor_name']} · {side}<br>RankIC {row['mean_rank_ic']:+.4f} | RankICIR {row['rank_icir']:+.4f} | 超额 {row['cumulative_excess_simple_return']:+.1%}")
        interactive = make_subplots(rows=8, cols=2, subplot_titles=titles, vertical_spacing=0.045, horizontal_spacing=0.08)
        for index, row in enumerate(subset):
            x, ys = plot_values(daily, row["factor_name"], all_dates)
            draw_axis(axes.flat[index], row, x, ys)
            single, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
            draw_axis(ax, row, x, ys)
            ax.legend(loc="best", fontsize=9)
            single.savefig(images / f"{row['factor_name']}.png", dpi=160)
            plt.close(single)
            for y, key, label, dash in zip(ys, COLORS, ["累计超额单利", "选币组合累计单利", "币池等权累计单利"], ["solid", "dash", "dot"], strict=True):
                interactive.add_trace(go.Scatter(x=[v.strftime("%Y-%m-%d %H:%M") for v in x], y=y,
                    name=label, legendgroup=key, showlegend=index == 0, connectgaps=False,
                    line={"color": COLORS[key], "width": 2 if key == "excess" else 1.2, "dash": dash},
                    hovertemplate="%{x}<br>%{y:.2%}<extra>%{fullData.name}</extra>"), row=index // 2 + 1, col=index % 2 + 1)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=11)
        fig.savefig(output / f"overview_{page + 1}.png", dpi=140)
        plt.close(fig)
        interactive.update_yaxes(tickformat=".0%", zerolinecolor="#bcc3c7")
        interactive.update_xaxes(type="date", tickformat="%m-%d")
        interactive.update_layout(height=2680, template="plotly_white", hovermode="x unified",
            margin={"l": 65, "r": 35, "t": 100, "b": 30},
            legend={"orientation": "h", "y": 1.025, "x": 0}, font={"size": 12})
        sections.append(interactive.to_html(full_html=False, include_plotlyjs=True if page == 0 else False,
                                           config={"responsive": True, "displaylogo": False, "scrollZoom": True}))
    exclusions = ", ".join(local_time(t) for t in metadata["excluded_decision_timestamps"])
    description = (
        f"{metadata['start_beijing']} 至 {metadata['end_beijing']}（北京时间）；有效 {metadata['valid_daily_windows']} 天。"
        f" 排除缺失标签日：{exclusions}。每个因子按本期平均 RankIC 的正负，固定选最高或最低10%，等权只多。"
        " RankIC 为每日因子排名与未来24h超额收益的 Spearman 相关均值；RankICIR 为均值/样本标准差，未年化，保留原始符号。"
        " 截面值恒定导致相关系数无定义的日期不计入IC统计，实际IC天数见CSV；选币并列按symbol排序。"
        " 三条曲线均为日收益相加，不复利；基准沿用原始币池可计算收益成员的等权收益。"
        " 未计交易费用、滑点和资金费率，不设止损。方向使用2026年整段信息，属于回看诊断；64因子名单也曾参考2026年数据。"
    )
    table_rows = "".join(f"<tr><td>{html.escape(r['factor_name'])}</td><td>{'最高10%' if r['direction_sign']==1 else '最低10%'}</td><td>{r['mean_rank_ic']:+.5f}</td><td>{r['rank_icir']:+.5f}</td><td>{r['cumulative_excess_simple_return']:+.2%}</td></tr>" for r in rows)
    document = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026 年 64 因子单利收益</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#20272b;background:#fff;letter-spacing:0}}main{{max-width:1500px;margin:auto}}h1{{font-size:26px}}p{{line-height:1.8}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:7px;border-bottom:1px solid #e4e7e9;text-align:right}}td:first-child,th:first-child{{text-align:left}}.chart{{overflow:auto;min-width:0}}.chart>div{{min-width:800px}}a{{color:#007c78}}</style>
<main><h1>2026 年 64 因子单利收益</h1><p>{html.escape(description)}</p><p>按累计超额单利收益从高到低排列。<a href="summary.csv">指标明细 CSV</a> · <a href="report.md">报告</a></p>
{''.join(f'<section><h2>因子 {i*16+1}–{(i+1)*16}</h2><a href="overview_{i+1}.png">下载本组静态图</a><div class="chart">{section}</div></section>' for i,section in enumerate(sections))}
<h2>完整指标表</h2><table><thead><tr><th>因子</th><th>选币方向</th><th>RankIC</th><th>RankICIR</th><th>累计超额单利</th></tr></thead><tbody>{table_rows}</tbody></table></main></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    report = ["# 2026 年 64 因子单利收益诊断", "", description, "",
              "每个图先在收益起点补0，再将已兑现日收益画在对应结束时刻；缺失日断线，后续累计只包含有效窗口。", "",
              "| 因子 | 方向 | RankIC | RankICIR | 组合累计单利 | 累计超额单利 |", "|---|---|---:|---:|---:|---:|"]
    report += [f"| {r['factor_name']} | {'最高10%' if r['direction_sign']==1 else '最低10%'} | {r['mean_rank_ic']:+.5f} | {r['rank_icir']:+.5f} | {r['selected_cumulative_simple_return']:+.2%} | {r['cumulative_excess_simple_return']:+.2%} |" for r in rows]
    report += ["", f"源模型运行：`{metadata['source_run_id']}`。本报告不使用模型预测值，只复用同一批因子排名和收益标签。", "",
               "这是毛收益因子诊断；源策略回测计入交易成本且包含293天，不能直接把本表当作源策略的净收益比较。"]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.run:
        run = args.run.resolve()
    else:
        pointer = ROOT / "data/models/pit_rank64_xgboost_daily_excess_long_only/latest.json"
        run = Path(json.loads(pointer.read_text())["manifest"]).parent
    manifest = json.loads((run / "manifest.json").read_text())
    names = manifest["features"]
    portfolio = manifest["test_portfolio"]
    predictions_path = run / "predictions.parquet"
    frame = pl.read_parquet(predictions_path).filter(
        (pl.col("decision_timestamp") >= portfolio["backtest_start_ms"])
        & (pl.col("decision_timestamp") + DAY_MS <= portfolio["backtest_end_ms"])
    )
    all_dates = frame["decision_timestamp"].unique().sort().to_list()
    summary, daily, missing = calculate(frame, names)
    output = ROOT / "reports/factors" / f"rank64_2026_{run.name.rsplit('_', 1)[-1]}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_run_id": manifest["run_id"], "source_predictions": str(predictions_path),
        "source_predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "feature_count": len(names), "direction_source": "2026 full-period mean RankIC (retrospective)",
        "ic_sampling": "daily at 04:00 UTC; same complete cross sections as returns",
        "icir_annualized": False, "costs_included": False, "stop_loss": None,
        "start_beijing": local_time(all_dates[0]), "end_beijing": local_time(all_dates[-1] + DAY_MS),
        "valid_daily_windows": summary["daily_period_count"][0],
        "excluded_decision_timestamps": missing["decision_timestamp"].unique().sort().to_list(),
        "excluded_observations": missing.to_dicts(),
    }
    summary.write_csv(output / "summary.csv")
    daily.write_parquet(output / "daily.parquet")
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    render(output, summary, daily, all_dates, metadata)
    print(json.dumps({"output": str(output), "factor_count": len(names),
                      "days": metadata["valid_daily_windows"],
                      "positive_excess_count": summary.filter(pl.col("cumulative_excess_simple_return") > 0).height,
                      "top5": summary.head(5).to_dicts()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
