"""Ten equal-count factor portfolios and daily top-minus-bottom spreads."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
from matplotlib.ticker import PercentFormatter
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from build_factor_2026_curves import DAY_MS, ROOT, SHANGHAI, local_time


def compute(frame: pl.DataFrame, names: list[str], metrics: pl.DataFrame):
    rows = []
    for (stamp,), group in frame.sort(["decision_timestamp", "symbol"]).group_by("decision_timestamp", maintain_order=True):
        if group.height < 10 or group["forward_return"].null_count():
            raise ValueError("each included date needs ten assets and complete future returns")
        for name in names:
            ordered = group.sort([name, "symbol"])
            for number, indices in enumerate(np.array_split(np.arange(group.height), 10), 1):
                selected = ordered[indices.tolist()]
                rows.append({"factor_name": name, "decision_timestamp": stamp,
                             "exit_timestamp": stamp + DAY_MS, "decile": number,
                             "member_count": selected.height, "symbols": selected["symbol"].to_list(),
                             "return": float(selected["forward_return"].mean()),
                             "universe_return": float(group["universe_forward_return"][0])})
    daily = pl.DataFrame(rows).sort(["factor_name", "decile", "decision_timestamp"]).with_columns(
        pl.col("return").cum_sum().over("factor_name", "decile").alias("cumulative_simple_return"))
    spread = daily.filter(pl.col("decile").is_in([1, 10])).pivot(
        on="decile", index=["factor_name", "decision_timestamp", "exit_timestamp"], values="return",
    ).rename({"1": "g1_return", "10": "g10_return"}).join(
        metrics.select("factor_name", "mean_rank_ic", "rank_icir", "direction_sign"), on="factor_name",
    ).sort(["factor_name", "decision_timestamp"]).with_columns(
        (pl.col("g10_return") - pl.col("g1_return")).alias("spread"),
    ).with_columns((pl.col("spread") * pl.col("direction_sign")).alias("directed_spread")).with_columns(
        pl.col("spread").cum_sum().over("factor_name").alias("cumulative_spread"),
        pl.col("directed_spread").cum_sum().over("factor_name").alias("cumulative_directed_spread"),
    )
    summary = spread.group_by("factor_name").agg(
        pl.col("mean_rank_ic").first(), pl.col("rank_icir").first(), pl.col("direction_sign").first(),
        pl.col("spread").sum().alias("cumulative_spread"),
        pl.col("directed_spread").sum().alias("cumulative_directed_spread"),
        pl.col("spread").mean().alias("mean_daily_spread"), pl.len().alias("daily_windows"),
    ).sort("cumulative_directed_spread", descending=True)
    return daily, spread, summary


def series_for(frame, dates, column):
    values = dict(frame.select("decision_timestamp", column).iter_rows())
    return [0.0] + [values.get(t) for t in dates]


def render(output, daily, spread, summary, dates, metadata):
    import matplotlib.dates as mdates
    from datetime import datetime, timezone

    period_label = metadata.get("period_label", "2026")
    page_title = metadata.get("page_title", "2026 年 64 因子：十分组收益与 spread")
    valid_days = metadata["valid_daily_windows"]
    excluded = ", ".join(local_time(t)[:10] for t in metadata.get("excluded_decision_timestamps", [])) or "无"
    date_format = "%Y-%m" if dates[-1] - dates[0] > 365 * DAY_MS else "%m-%d"
    x = [local_time(dates[0])] + [local_time(t + DAY_MS) for t in dates]
    date_objects = [datetime.strptime(v, "%Y-%m-%d %H:%M").replace(tzinfo=SHANGHAI) for v in x]
    colors = ["#354f9b", "#3579ba", "#24a1bc", "#299c87", "#64a957",
              "#a3a83d", "#c79331", "#d17636", "#c85243", "#a52c48"]
    payload, sections = {}, []
    individual = output / "individual"
    individual.mkdir(exist_ok=True)
    for index, row in enumerate(summary.iter_rows(named=True)):
        name = row["factor_name"]
        factor = daily.filter(pl.col("factor_name") == name)
        sp = spread.filter(pl.col("factor_name") == name)
        figure = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{"colspan": 2}, None]],
                               column_widths=[0.72, 0.28], row_heights=[0.60, 0.40],
                               subplot_titles=["十组累计单利收益：G1最低 → G10最高", "各组期末累计单利", "累计收益 spread"],
                               vertical_spacing=0.20, horizontal_spacing=0.09)
        static = plt.figure(figsize=(13, 8), layout="constrained")
        grid = static.add_gridspec(2, 2, width_ratios=[2.6, 1], height_ratios=[1.4, 1])
        top, bars, bottom = static.add_subplot(grid[0, 0]), static.add_subplot(grid[0, 1]), static.add_subplot(grid[1, :])
        endpoints = []
        for number in range(1, 11):
            part = factor.filter(pl.col("decile") == number)
            y = series_for(part, dates, "cumulative_simple_return")
            endpoints.append(float(part["return"].sum()))
            figure.add_trace(go.Scatter(x=x, y=y, name=f"G{number}", legendgroup=f"G{number}",
                line={"color": colors[number - 1], "width": 2 if number in [1, 10] else 1.1},
                connectgaps=False, hovertemplate="%{x}<br>%{y:.2%}<extra>%{fullData.name}</extra>"), row=1, col=1)
            top.plot(date_objects, [np.nan if v is None else v for v in y], color=colors[number - 1],
                     linewidth=1.7 if number in [1, 10] else 0.9, label=f"G{number}")
        figure.add_trace(go.Bar(x=[f"G{i}" for i in range(1, 11)], y=endpoints,
                               marker_color=colors, showlegend=False,
                               hovertemplate="%{x}<br>%{y:.2%}<extra></extra>"), row=1, col=2)
        bars.bar(range(1, 11), endpoints, color=colors)
        bars.set_xticks(range(1, 11), [str(i) for i in range(1, 11)], fontsize=8)
        for column, label, color, dash in [
            ("cumulative_spread", "原始 G10 − G1", "#156c85", "solid"),
            ("cumulative_directed_spread", "按 RankIC 方向调整", "#b95820", "dash"),
        ]:
            y = series_for(sp, dates, column)
            figure.add_trace(go.Scatter(x=x, y=y, name=label, line={"color": color, "dash": dash},
                connectgaps=False, hovertemplate="%{x}<br>%{y:.2%}<extra>%{fullData.name}</extra>"), row=2, col=1)
            bottom.plot(date_objects, [np.nan if v is None else v for v in y], color=color,
                        linestyle="--" if dash == "dash" else "-",
                        label="Raw G10 - G1" if dash == "solid" else "IC-directed spread")
        annotation = f"RankIC {row['mean_rank_ic']:+.4f} | RankICIR {row['rank_icir']:+.4f} | G10−G1 {row['cumulative_spread']:+.2%} | 定向 {row['cumulative_directed_spread']:+.2%}"
        figure.update_layout(title={"text": f"{name}<br><sup>{annotation}</sup>", "font": {"size": 18}},
                             template="plotly_white", height=760, margin={"t": 145, "b": 40, "l": 65, "r": 30},
                             hovermode="x unified", legend={"orientation": "h", "y": 1.13, "font": {"size": 11}})
        figure.update_yaxes(tickformat=".0%", zerolinecolor="#a7afb6")
        figure.update_xaxes(type="date", tickformat=date_format, range=[x[0], x[-1]], row=1, col=1)
        figure.update_xaxes(type="date", tickformat=date_format, range=[x[0], x[-1]], row=2, col=1)
        coverage_label = "\n" + metadata["coverage_label"] if metadata.get("coverage_label") else ""
        static.suptitle(f"{name} | RankIC {row['mean_rank_ic']:+.4f} | RankICIR {row['rank_icir']:+.4f}\n{period_label} gross simple returns | G10-G1 {row['cumulative_spread']:+.2%} | IC-directed {row['cumulative_directed_spread']:+.2%}{coverage_label}", fontsize=13)
        top.set_title("Ten equal-count portfolios (G1 lowest; G10 highest)", fontsize=10)
        bars.set_title("Cumulative returns", fontsize=10)
        bottom.set_title("Top-bottom spread (100% long / 100% short convention)", fontsize=10)
        top.legend(ncol=5, fontsize=8, loc="lower left")
        bottom.legend(fontsize=9, loc="best")
        for ax in [top, bottom, bars]:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis="y", alpha=0.18)
            ax.axhline(0, color="#9aa3ab", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in [top, bottom]:
            ax.set_xlim(date_objects[0], date_objects[-1])
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3 if date_format == "%Y-%m" else 2, tz=SHANGHAI))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m" if date_format == "%Y-%m" else "%b", tz=SHANGHAI))
        static.savefig(individual / f"{name}.png", dpi=140)
        plt.close(static)
        key = f"factor-{index}"
        payload[key] = json.loads(figure.to_json())
        sections.append(f'<details id="{key}" {"open" if index == 0 else ""}><summary>{html.escape(name)} · {annotation}</summary><a href="individual/{name}.png">静态图</a><div class="plot" id="plot-{key}"></div></details>')
    # All-factor spread overview is small enough to scan without opening 64 panels.
    for page in range(4):
        fig, axes = plt.subplots(4, 4, figsize=(18, 13), layout="constrained")
        fig.suptitle(f"{period_label} factor spreads | page {page+1}/4 | {valid_days} valid daily windows\nBlue: raw G10-G1 | Orange dashed: IC-directed | gross simple returns", fontsize=14)
        for ax, row in zip(axes.flat, summary.slice(page * 16, 16).iter_rows(named=True), strict=True):
            part = spread.filter(pl.col("factor_name") == row["factor_name"])
            for col, color, style in [("cumulative_spread", "#156c85", "-"), ("cumulative_directed_spread", "#b95820", "--")]:
                ax.plot(date_objects, [np.nan if v is None else v for v in series_for(part, dates, col)], color=color, linestyle=style, linewidth=1.2)
            ax.set_title(f"{row['factor_name']}\nRankIC {row['mean_rank_ic']:+.4f} | ICIR {row['rank_icir']:+.4f}", fontsize=9, loc="left")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6 if date_format == "%Y-%m" else 3, tz=SHANGHAI))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m" if date_format == "%Y-%m" else "%b", tz=SHANGHAI))
            ax.axhline(0, color="#a7afb6", linewidth=0.5)
            ax.grid(alpha=0.15)
            ax.tick_params(labelsize=8)
        fig.savefig(output / f"spread_overview_{page+1}.png", dpi=140)
        plt.close(fig)
    explanation = (
        f"{metadata['start_beijing']} 至 {metadata['end_beijing']}（北京时间），{valid_days} 个有效日窗口，排除日期：{excluded}。"
        + metadata.get("coverage_note", "") +
        "每天按因子值升序分成10个互不重叠的组，G1最低、G10最高，组内等权；并列按symbol排序，每组约10%，人数最多差1。"
        "上左为十组累计单利，上右为期末分组收益，下方为累计spread。原始spread = G10日收益 − G1日收益；定向spread = 原始spread × 本期平均RankIC的符号。"
        "全部为日收益相加，不复利、不设止损，未扣手续费、滑点、资金费率。Spread采用多头100%减空头100%的差值口径，不是总敞口100%的组合净值；若多空各50%，收益差减半。"
        "IC定向使用整段报告区间信息，仅为回看诊断；64因子的选择历史也使用过2026年数据。RankIC与未年化RankICIR使用同区间每日截面计算。"
        "不能整除时按numpy.array_split分配余数，因此G10可能比此前单独向上取整选10%少1币。断线表示缺失日，后续累计仅包含有效窗口。"
    )
    (output / "plotly.min.js").write_text(get_plotlyjs())
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PAGE_TITLE</title>
<style>body{font-family:system-ui,sans-serif;color:#263138;margin:24px;letter-spacing:0}main{max-width:1450px;margin:auto}h1{font-size:26px}p{line-height:1.8}details{border-top:1px solid #dce2e6;padding:14px 0}summary{cursor:pointer;font-size:14px;font-weight:600;line-height:1.8;overflow-wrap:anywhere}a{color:#007c78}.plot{height:760px;min-width:800px}.frame{overflow:auto}input{font:inherit;padding:8px;width:280px;max-width:90%}</style>
<script src="plotly.min.js"></script><main><h1>PAGE_TITLE</h1><p>EXPLANATION</p><p>OVERVIEWS · <a href="summary.csv">全部指标</a> · <a href="report.md">口径说明</a> YEARLY_LINK</p><input id="search" type="search" placeholder="筛选因子名称" aria-label="筛选因子名称"><div class="frame">SECTIONS</div></main>
<script>const plots=PAYLOAD;
function draw(d){if(!d.open || d.dataset.ready)return;const f=plots[d.id];Plotly.newPlot('plot-'+d.id,f.data,f.layout,{responsive:true,displaylogo:false,scrollZoom:true});d.dataset.ready='1';}
document.querySelectorAll('details').forEach(d=>{d.addEventListener('toggle',()=>draw(d));draw(d)});
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('details').forEach(d=>{d.hidden=!d.querySelector('summary').textContent.toLowerCase().includes(q)})});</script></html>'''
    document = document.replace("EXPLANATION", html.escape(explanation)).replace("OVERVIEWS", " · ".join(f'<a href="spread_overview_{i}.png">总览{i}</a>' for i in range(1,5)))
    document = document.replace("PAGE_TITLE", html.escape(page_title)).replace("YEARLY_LINK", ' · <a href="yearly_summary.csv">2025 / 2026 分年指标</a>' if (output / "yearly_summary.csv").exists() else "")
    document = document.replace("SECTIONS", "".join(sections)).replace("PAYLOAD", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    (output / "index.html").write_text(document, encoding="utf-8")
    (output / "report.md").write_text("# " + page_title + "\n\n" + explanation + "\n\n来源：`" + metadata["source_run_id"] + "`。\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "reports/factors/rank64_2026_20261023T032315096488Z")
    args = parser.parse_args()
    metadata = json.loads((args.source / "metadata.json").read_text())
    source = Path(metadata["source_predictions"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["source_predictions_sha256"]:
        raise ValueError("source prediction snapshot changed")
    metrics = pl.read_csv(args.source / "summary.csv")
    prior = pl.read_parquet(args.source / "daily.parquet")
    dates = prior["decision_timestamp"].unique().sort().to_list()
    frame = pl.read_parquet(source).filter(pl.col("decision_timestamp").is_in(dates))
    daily, spread, summary = compute(frame, metrics["factor_name"].to_list(), metrics)
    out = args.source / "deciles"
    out.mkdir(exist_ok=True)
    daily.write_parquet(out / "decile_daily.parquet")
    spread.write_parquet(out / "spread_daily.parquet")
    summary.write_csv(out / "summary.csv")
    (out / "metadata.json").write_text(json.dumps({**metadata, "group_count": 10, "group_order": "ascending factor, symbol ties",
        "split_rule": "numpy.array_split", "raw_spread": "G10 - G1", "spread_gross_exposure": 2.0,
        "source_factor_report": str(args.source)}, ensure_ascii=False, indent=2))
    all_dates = list(range(dates[0], dates[-1]+DAY_MS, DAY_MS))
    render(out, daily, spread, summary, all_dates, metadata)
    print(json.dumps({"output": str(out), "factor_count": summary.height, "group_daily_rows": daily.height,
                      "spread_daily_rows": spread.height, "top5": summary.head(5).to_dicts()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
