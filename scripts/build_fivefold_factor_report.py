from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parents[1]
RUN = "walk_forward_4h_h6_forward_return_20260704T123039Z"
DAY = 86_400_000
BAR = 14_400_000
MODEL = "MODEL_SCORE"
KEYS = ["exchange", "market_type", "symbol", "timeframe", "timestamp"]
GROUPS = ["fold", "split", "timestamp"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ratio(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.std(values, ddof=1) <= 1e-14:
        return None
    return float(np.mean(values) / np.std(values, ddof=1))


def mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def tail_indices(values: np.ndarray, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = math.ceil(len(values) * 0.1)
    # Preserve historical top selection; remove its members before forming the short tail.
    high = np.lexsort((symbols, -values))[:n]
    ordered_low = np.lexsort((symbols, values))
    low = ordered_low[~np.isin(ordered_low, high)][:n]
    return low, high


def cross_sections(frame: pl.DataFrame, feature: str) -> pl.DataFrame:
    frame = frame.filter(pl.col(feature).is_finite() & pl.col("forward_return").is_finite())
    valid_corr = (pl.col(feature).n_unique() > 1) & (pl.col("forward_return").n_unique() > 1)
    return (
        frame.group_by(GROUPS)
        .agg(
            pl.len().alias("eligible_count"),
            pl.when(valid_corr).then(pl.corr(feature, "forward_return")).otherwise(None).alias("ic"),
            pl.when(valid_corr).then(pl.corr(feature, "forward_return", method="spearman")).otherwise(None).alias("rank_ic"),
        )
        .filter(pl.col("eligible_count") >= 20)
        .with_columns(pl.lit(feature).alias("factor_name"))
        .sort(GROUPS)
    )


def daily_sections(frame: pl.DataFrame, feature: str) -> list[dict]:
    daily = frame.filter((pl.col("timestamp") % DAY == 0))
    rows = []
    for (fold, split, timestamp), part in daily.partition_by(GROUPS, as_dict=True).items():
        total = part.height
        part = part.filter(pl.col(feature).is_finite() & pl.col("forward_return").is_finite()).sort("symbol")
        if part.height < 20:
            continue
        x = part[feature].to_numpy()
        y = part["forward_return"].to_numpy()
        symbols = part["symbol"].to_numpy()
        low, high = tail_indices(x, symbols)
        order = np.lexsort((symbols, x))
        groups = np.array_split(order, 10)
        row = {
            "fold": fold, "split": split, "timestamp": timestamp,
            "decision_timestamp": timestamp + BAR, "exit_timestamp": timestamp + BAR + DAY,
            "factor_name": feature, "eligible_count": part.height, "source_count": total,
            "selected_count": len(high), "tail_overlap_count": len(set(low) & set(high)),
            "unique_factor_values": int(len(np.unique(x))),
            "top_return": float(y[high].mean()), "bottom_return": float(y[low].mean()),
            "universe_return": float(part["universe_return"].first()),
            "top_symbols": symbols[high].tolist(), "bottom_symbols": symbols[low].tolist(),
        }
        row["raw_spread"] = row["top_return"] - row["bottom_return"]
        for number, indices in enumerate(groups, 1):
            row[f"g{number}_return"] = float(y[indices].mean())
            row[f"g{number}_count"] = len(indices)
        rows.append(row)
    return rows


def summarize(cs: pl.DataFrame, daily: pl.DataFrame, importance: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    directions = {}
    for (fold, feature), part in cs.filter(pl.col("split") == "train").partition_by(["fold", "factor_name"], as_dict=True).items():
        value = mean(part["rank_ic"].to_numpy())
        directions[fold, feature] = 0 if value is None or value == 0 else (1 if value > 0 else -1)
    records = daily.to_dicts()
    for row in records:
        direction = directions[row["fold"], row["factor_name"]]
        row["train_direction"] = direction
        row["directed_spread"] = direction * row["raw_spread"]
        row["directed_long_return"] = row["top_return"] if direction > 0 else row["bottom_return"] if direction < 0 else 0.0
        row["directed_long_excess"] = row["directed_long_return"] - row["universe_return"]
    daily = pl.DataFrame(records).sort(["factor_name", "fold", "split", "timestamp"])
    rows = []
    for (fold, split, feature), part in cs.partition_by(["fold", "split", "factor_name"], as_dict=True).items():
        d = daily.filter((pl.col("fold") == fold) & (pl.col("split") == split) & (pl.col("factor_name") == feature))
        ic, ric = part["ic"].to_numpy(), part["rank_ic"].to_numpy()
        direction = directions[fold, feature]
        row = {
            "fold": fold, "split": split, "factor_name": feature,
            "timestamp_count": part.height, "ic_count": int(np.isfinite(ic).sum()),
            "rank_ic_count": int(np.isfinite(ric).sum()), "mean_ic": mean(ic), "mean_rank_ic": mean(ric),
            "icir": ratio(ic), "rank_icir": ratio(ric), "train_direction": direction,
            "rank_ic_direction_win_rate": mean(np.where(np.isfinite(ric), ric * direction > 0, np.nan)),
            "daily_count": d.height, "mean_daily_spread": d["raw_spread"].mean(),
            "constant_daily_count": int((d["unique_factor_values"] == 1).sum()),
            "cumulative_spread": d["raw_spread"].sum(),
            "directed_cumulative_spread": d["directed_spread"].sum(),
            "directed_spread_win_rate": (d["directed_spread"] > 0).mean(),
            "top_cumulative_return": d["top_return"].sum(), "bottom_cumulative_return": d["bottom_return"].sum(),
            "universe_cumulative_return": d["universe_return"].sum(),
            "directed_long_cumulative_excess": d["directed_long_excess"].sum(),
            "min_eligible_count": part["eligible_count"].min(),
            "max_eligible_count": part["eligible_count"].max(),
        }
        rows.append(row)
    summary = pl.DataFrame(rows).join(importance.rename({"feature_name": "factor_name"}), on=["fold", "factor_name"], how="left")
    return summary.sort(["fold", "split", "factor_name"]), daily


def make_html(output: Path, summary: pl.DataFrame, cs: pl.DataFrame, daily: pl.DataFrame, names: list[str]) -> None:
    data = {}
    for row in summary.to_dicts():
        fold, split, name = row["fold"], row["split"], row["factor_name"]
        mask = (pl.col("fold") == fold) & (pl.col("split") == split) & (pl.col("factor_name") == name)
        d = daily.filter(mask).sort("timestamp")
        c = cs.filter(mask).sort("timestamp")
        dates = [datetime.fromtimestamp(t / 1000, timezone.utc).isoformat() for t in d["exit_timestamp"]]
        starts = [datetime.fromtimestamp(t / 1000, timezone.utc).isoformat() for t in c["timestamp"]]
        data[f"{fold}|{split}|{name}"] = {
            "metrics": row, "dates": dates, "icDates": starts,
            "ic": c["ic"].to_list(), "rankIC": c["rank_ic"].to_list(),
            "raw": d["raw_spread"].cum_sum().to_list(), "directed": d["directed_spread"].cum_sum().to_list(),
            "groups": [d[f"g{i}_return"].cum_sum().to_list() for i in range(1, 11)],
        }
    # pandas normalizes any remaining NaN to JSON null for strict browser parsing.
    payload = json.loads(pd.Series(data).to_json(force_ascii=False))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>5-fold 因子诊断</title><script src="plotly.min.js"></script>
<style>body{font:14px system-ui;margin:0;color:#202b29;background:#fff}header,main{padding:20px 28px}header{background:#edf3ef;border-bottom:1px solid #bfcfc7}h1{font-size:24px}p{line-height:1.7}select{padding:7px;margin:0 18px 8px 4px}section{margin:20px 0}.plot{height:390px;min-width:600px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#edf3ef}a{color:#2267a3}.note{color:#6b4527}#stats{line-height:1.9}</style>
<header><h1>5-fold / 30 因子诊断</h1><p>历史 +222.37% long-only 策略的研究数据；另附 MODEL_SCORE 作为组合模型对照。没有重训、重新回测或修改线上策略。</p>
<p class="note">IC = Pearson；RankIC = Spearman；两个 IR 均为时间序列均值 / 样本标准差，未年化。旧报告的 RankIC 列实际上是 Pearson，本页已重新计算。<br>相关性每 4h 采样，24h 标签重叠；累计 spread 每日一次。方向仅来自各 fold 训练集。因子筛选本身存在全样本选择偏差，不是完全隔离的样本外实验。</p>
<label>Fold <select id="fold">__FOLDS__</select></label><label>区间 <select id="split"><option value="test">测试集</option><option value="valid">验证集</option><option value="train">训练集</option></select></label><label>因子 <select id="factor">__FACTORS__</select></label>
<p><a href="report.md">完整说明</a> · <a href="factor_metrics.csv">全部指标 CSV</a> · <a href="test_factor_metrics.csv">测试集指标</a> · <a href="stability.csv">跨 fold 稳定性</a> · <a href="heatmaps.html">稳定性热力图</a></p></header>
<main><div id="stats"></div><section class="scroll"><div id="spread" class="plot"></div></section><section class="scroll"><div id="groups" class="plot"></div></section><section class="scroll"><div id="ic" class="plot"></div></section><h2>本 fold 指标</h2><div class="scroll"><table id="table"></table></div></main>
<script>const DATA=__DATA__;
const names=__NAMES__; const pct=v=>v==null?'NA':(100*v).toFixed(2)+'%';const num=v=>v==null?'NA':v.toFixed(4);
const layout=title=>({title:{text:title,font:{size:17}},margin:{l:64,r:24,t:50,b:45},paper_bgcolor:'#fff',plot_bgcolor:'#fff',legend:{orientation:'h',y:1.16},xaxis:{gridcolor:'#eee'},yaxis:{gridcolor:'#eee',zerolinecolor:'#888'},hovermode:'x unified'});
function draw(){const f=document.getElementById('fold').value,s=document.getElementById('split').value,n=document.getElementById('factor').value,d=DATA[`${f}|${s}|${n}`],m=d.metrics;
document.getElementById('stats').textContent=`${n} | IC ${num(m.mean_ic)} | RankIC ${num(m.mean_rank_ic)} | ICIR ${num(m.icir)} | RankICIR ${num(m.rank_icir)} | 训练方向 ${m.train_direction} | 4h截面 ${m.timestamp_count} | 日收益窗口 ${m.daily_count} | 原始spread ${pct(m.cumulative_spread)} | 训练定向spread ${pct(m.directed_cumulative_spread)}`;
Plotly.react('spread',[{x:d.dates,y:d.raw,name:'最高10% − 最低10%',line:{color:'#2166ac'}},{x:d.dates,y:d.directed,name:'训练集定向 spread',line:{color:'#d56a21',dash:'dash'}}],{...layout('累计单利 spread（100% 多 + 100% 空口径，未扣成本）'),yaxis:{tickformat:'.0%'},xaxis:{rangeslider:{visible:true,thickness:.10}}},{responsive:true});
Plotly.react('groups',d.groups.map((y,i)=>({x:d.dates,y,name:'G'+(i+1),line:{width:i===0||i===9?3:1}})),{...layout('十组累计单利收益（G1 低 → G10 高；组内等权）'),yaxis:{tickformat:'.0%'}},{responsive:true});
Plotly.react('ic',[{x:d.icDates,y:d.ic,name:'Pearson IC',line:{color:'#2166ac'}},{x:d.icDates,y:d.rankIC,name:'Spearman RankIC',line:{color:'#25835e'}}],layout('逐 4h 截面相关性（横轴为 K 线开始时间 UTC）'),{responsive:true});
const rows=names.map(name=>DATA[`${f}|${s}|${name}`].metrics).sort((a,b)=>(b.directed_cumulative_spread??0)-(a.directed_cumulative_spread??0));
document.getElementById('table').innerHTML='<tr>'+['因子','IC','RankIC','ICIR','RankICIR','原始spread','训练定向spread','定向多头超额','Gain'].map(x=>'<th>'+x+'</th>').join('')+'</tr>'+rows.map(r=>'<tr><td>'+r.factor_name+'</td>'+[num(r.mean_ic),num(r.mean_rank_ic),num(r.icir),num(r.rank_icir),pct(r.cumulative_spread),pct(r.directed_cumulative_spread),pct(r.directed_long_cumulative_excess),pct(r.importance_gain_pct)].map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('');}
document.querySelectorAll('select').forEach(e=>e.addEventListener('change',draw));draw();</script></html>'''
    page = page.replace("__FOLDS__", "".join(f'<option value="{i}">{i}</option>' for i in range(1, 6)))
    page = page.replace("__FACTORS__", "".join(f'<option>{name}</option>' for name in names))
    page = page.replace("__DATA__", json.dumps(payload, ensure_ascii=False, allow_nan=False))
    page = page.replace("__NAMES__", json.dumps(names))
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    panels = []
    for metric, title, pct in [("mean_rank_ic", "测试集 RankIC", False), ("rank_icir", "测试集 RankICIR", False), ("directed_cumulative_spread", "训练集定向的测试累计 spread", True), ("importance_gain_pct", "每 fold 模型 Gain 占比", True)]:
        matrix = summary.filter(pl.col("split") == "test").to_pandas().pivot(index="factor_name", columns="fold", values=metric).reindex(names)
        fig = go.Figure(go.Heatmap(z=matrix.values, x=[f"Fold {i}" for i in matrix.columns], y=names, colorscale="RdBu", reversescale=True, zmid=0, texttemplate="%{z:.1%}" if pct else "%{z:.3f}"))
        fig.update_layout(title=title, height=900, margin=dict(l=220, r=40, t=60, b=40), font=dict(size=12), yaxis=dict(autorange="reversed"))
        panels.append(fig.to_html(full_html=False, include_plotlyjs=False))
    (output / "heatmaps.html").write_text('<meta charset="utf-8"><script src="plotly.min.js"></script>'+"\n".join(panels), encoding="utf-8")


def make_png(output: Path, summary: pl.DataFrame, daily: pl.DataFrame, names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 3, figsize=(17, 12), constrained_layout=True)
    for ax, metric, title in zip(axes, ["mean_rank_ic", "rank_icir", "directed_cumulative_spread"], ["Test RankIC (Spearman)", "Test RankICIR (not annualized)", "Test cumulative spread (train direction)"]):
        matrix = summary.filter(pl.col("split") == "test").to_pandas().pivot(index="factor_name", columns="fold", values=metric).reindex(names)
        values = matrix.to_numpy()
        bound = np.nanmax(np.abs(values))
        ax.imshow(values, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
        ax.set_xticks(range(5), [f"Fold {i}" for i in range(1, 6)])
        ax.set_yticks(range(len(names)), names if ax is axes[0] else [""] * len(names), fontsize=9)
        ax.set_title(title, fontsize=11, pad=14)
        for i, j in np.ndindex(values.shape):
            ax.text(j, i, f"{values[i,j]:.1%}" if metric.endswith("spread") else f"{values[i,j]:.3f}", ha="center", va="center", fontsize=8, color="white" if abs(values[i,j]) > .6 * bound else "black")
    fig.savefig(output / "test_heatmaps.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(5, 1, figsize=(13, 14), constrained_layout=True)
    for fold, ax in enumerate(axes, 1):
        for name in ["range_to_quote_volume_12", "distance_to_high_24", "volatility_8", MODEL]:
            d = daily.filter((pl.col("split") == "test") & (pl.col("fold") == fold) & (pl.col("factor_name") == name)).sort("timestamp")
            ax.plot(range(15), np.r_[0, np.cumsum(d["directed_spread"].to_numpy())], label=name)
        ax.axhline(0, color="#777", linewidth=.7)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(f"Fold {fold} | Daily cumulative spread, direction fixed on training data")
        ax.set_xticks(range(0, 15, 2))
        ax.grid(alpha=.15)
    axes[0].legend(ncol=2, fontsize=9)
    axes[-1].set_xlabel("Non-overlapping 24h return windows")
    fig.savefig(output / "selected_spreads.png", dpi=140)
    plt.close(fig)


def main() -> None:
    run = ROOT / "data/models/walk_forward" / RUN
    output = ROOT / "reports/factors" / f"fivefold_{RUN}_20261027"
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "config.json").read_text())
    names = config["features"]
    predictions = pl.read_parquet(run / "predictions.parquet")
    assert len(names) == 30 and predictions["fold"].n_unique() == 5
    assert predictions.select(pl.struct(KEYS + ["fold", "split"]).n_unique()).item() == predictions.height
    start, end = predictions["timestamp"].min(), predictions["timestamp"].max()
    factors_path = ROOT / "data/factors/latest_factors.parquet"
    factors = (pl.scan_parquet(factors_path).filter((pl.col("timeframe") == "4h") & pl.col("timestamp").is_between(start, end) & pl.col("factor_name").is_in(names)).select(KEYS + ["factor_name", "factor_value"]).collect())
    assert factors.select(pl.struct(KEYS + ["factor_name"]).n_unique()).item() == factors.height
    wide = factors.pivot(index=KEYS, on="factor_name", values="factor_value")
    frame = predictions.join(wide, on=KEYS, how="left", validate="m:1").with_columns(pl.col("model_score").alias(MODEL))
    frame = frame.with_columns(pl.col("forward_return").mean().over(GROUPS).alias("universe_return"))
    assert frame.height == predictions.height
    parity = []
    for fold in range(1, 6):
        part = frame.filter(pl.col("fold") == fold)
        booster = lgb.Booster(model_file=str(run / f"fold_{fold:02d}_model.txt"))
        assert booster.feature_name() == names
        scores = booster.predict(part.select(names).to_numpy(), num_threads=2)
        error = float(np.max(np.abs(scores - part["model_score"].to_numpy())))
        assert error < 1e-12, (fold, error)
        parity.append({"fold": fold, "prediction_max_abs_error": error})
    print("Source feature/model parity verified for all 5 folds", flush=True)
    sections, days = [], []
    for name in names + [MODEL]:
        sections.append(cross_sections(frame, name))
        days.extend(daily_sections(frame, name))
        print(f"evaluated {name}", flush=True)
    cs = pl.concat(sections)
    importance = pl.read_parquet(run / "feature_importance.parquet")
    summary, daily = summarize(cs, pl.DataFrame(days), importance)
    assert summary.height == 31 * 5 * 3
    assert daily.filter(pl.col("split") == "test").height == 31 * 70
    assert np.allclose(daily["raw_spread"], daily["top_return"] - daily["bottom_return"])
    assert daily.filter(pl.col("eligible_count") != pl.col("source_count")).height == 0
    assert daily["tail_overlap_count"].max() == 0
    for _, part in daily.partition_by(["factor_name", "fold", "split"], as_dict=True).items():
        assert np.all(np.diff(part.sort("timestamp")["timestamp"].to_numpy()) == DAY)
    summary.write_csv(output / "factor_metrics.csv")
    summary.filter(pl.col("split") == "test").write_csv(output / "test_factor_metrics.csv")
    cs.write_parquet(output / "cross_section_metrics.parquet")
    daily.write_parquet(output / "daily_returns.parquet")
    stability = summary.filter(pl.col("split") == "test").group_by("factor_name").agg(
        pl.col("mean_ic").mean().alias("mean_test_ic"),
        pl.col("mean_rank_ic").mean().alias("mean_test_rank_ic"),
        pl.col("rank_icir").mean().alias("mean_fold_rank_icir"),
        ((pl.col("mean_rank_ic") * pl.col("train_direction")) > 0).sum().alias("aligned_rank_ic_folds"),
        (pl.col("directed_cumulative_spread") > 0).sum().alias("positive_directed_spread_folds"),
        pl.col("cumulative_spread").sum().alias("raw_cumulative_spread"),
        pl.col("directed_cumulative_spread").sum().alias("directed_cumulative_spread"),
        pl.col("directed_long_cumulative_excess").sum().alias("directed_long_cumulative_excess"),
    ).sort("directed_cumulative_spread", descending=True)
    stability.write_csv(output / "stability.csv")
    # Execution excluded ICP; keep this sensitivity separate from the 89-coin model universe.
    execution = frame.filter((pl.col("split") == "test") & (pl.col("symbol") != "ICP/USDT:USDT"))
    execution = execution.with_columns(pl.col("forward_return").mean().over(GROUPS).alias("universe_return"))
    sensitivity = pl.DataFrame([row for name in names + [MODEL] for row in daily_sections(execution, name)])
    sensitivity.write_parquet(output / "execution88_daily_sensitivity.parquet")
    sensitivity.group_by(["fold", "factor_name"]).agg(pl.col("raw_spread").sum().alias("raw_cumulative_spread"), pl.col("top_return").sum().alias("top_cumulative_return"), pl.col("universe_return").sum().alias("universe_cumulative_return")).write_csv(output / "execution88_summary.csv")
    make_html(output, summary, cs, daily, names + [MODEL])
    if "--png" in sys.argv:
        make_png(output, summary, daily, names + [MODEL])
    folds = pl.read_parquet(run / "folds.parquet")
    folds.write_csv(output / "fold_windows.csv")
    notes = [
        "# 原 +222.37% long-only 策略：逐 fold 因子诊断", "",
        f"源实验：`{RUN}`。本次未重训、未更改模型/策略、未更改东京服务。", "",
        "## 计算口径", "",
        "- 本次是原 30 个模型输入因子，另附 MODEL_SCORE 组合模型对照，不是后来去重的 64 个因子。",
        "- 5 次 rolling walk-forward：90天训练、14天验证、14天测试、14天步长，两处间隔各24小时；不是随机 K-fold。训练/验证窗口彼此重叠，仅测试窗口互不重叠。",
        "- 使用原始预测文件的 fold/split/样本和 forward_return 标签。冻结模型重算全部预测与档案完全一致（容差1e-12），确保因子文件对应原模型。",
        "- 主表恢复研究时89币截面。Freqtrade 排除 ICP 后为88币，另存 execution88 敏感性表；不能把两个币池混为一谈。",
        "- 收益 = close[t+6]/close[t]-1。t 为4h K线开始时间，因子在 t+4h 收盘后才可用；每日取 t=00:00 UTC（北京时间08:00开线、12:00收盘），收益在次日12:00实现。",
        "- IC：每个4h截面因子与未来24h收益的 Pearson；RankIC：Spearman（并列平均秩），再时间等权平均。常数截面的相关性记缺失。",
        "- ICIR = mean(IC)/std(IC,ddof=1)，RankICIR 同理；未年化。24h目标每4h采样存在重叠、自相关，不能把这些观测当独立样本计算显著性。",
        "- 原代码 training.py 的 target_rank_ic 使用 pl.corr 默认 Pearson；旧 folds 表所谓 test_rank_ic 实为 Pearson，不是 Spearman。本报告另算真正 RankIC，没有覆盖旧档案。",
        "- 10% spread = 因子最高 ceil(N*10%) 个币等权收益 − 最低 ceil(N*10%) 个币等权收益；89币时两端各9币。同值均按 symbol 升序，先选多头、排除后再选空头，确保两端不重合。",
        "- 常数截面没有排序信息：IC/RankIC记缺失。日收益表保留 unique_factor_values，汇总保留 constant_daily_count；常数日的组合由字典序打破并列，不代表因子预测能力。",
        "- 额外十组图按因子升序 array_split 成10组：89币时前9组各9币、末组8币。因此十等分 G10-G1 与对称9币尾部 spread 略有差异，不混用。",
        "- 累计 spread 仅使用每日一次、非重叠24h收益的简单累加。每fold测试14天，全5fold共70个窗口。图中时间为收益实现时间UTC。",
        "- 原始 spread 保留因子方向。定向 spread 的正负号只由该 fold 训练集平均 RankIC 确定，不用验证/测试集翻转。方向一致率亦基于训练方向。",
        "- 每个split所有记录按信号时间归属；最后一个标签允许在下一时间段实现，训练/验证间24h embargo隔开监督窗口。测试跨fold最后一期仍归信号所在fold。",
        "- 单因子与 MODEL_SCORE 的 spread 都是毛收益诊断，100%多+100%空（总敞口200%）；50%/50%口径数值减半。没有手续费、滑点、资金费率、10%止损或持仓延续，不是原Freqtrade复利净值。",
        "- 原222.37%是带10%止损、9槽满配、手续费的Freqtrade复利收益，71天报告窗口内有70次模型决策；因子spread不能直接相加来解释它。",
        "- 原30因子来自全历史单因子筛选/去相关，币池为2026-07-04快照，故仍有特征选择信息泄漏与幸存者偏差。重新按fold统计不自动修复这些历史设计问题。",
        "- 每fold只有14个不重叠日收益，不提供夸大的显著性结论；训练集指标也不是独立证据。", "",
        "## Fold 区间（K线开始时间 UTC）", "",
        "| Fold | Train | Valid | Test |", "|---|---|---|---|",
    ]
    for r in folds.to_dicts():
        notes.append(f"| {r['fold']} | {r['train_start']} ~ {r['train_end']} | {r['valid_start']} ~ {r['valid_end']} | {r['test_start']} ~ {r['test_end']} |")
    notes += ["", "## 组合模型对照（重新计算）", "", "| Fold | Pearson IC | Spearman RankIC | ICIR | RankICIR | 日序列累计原始spread |", "|---|---:|---:|---:|---:|---:|"]
    for r in summary.filter((pl.col("split") == "test") & (pl.col("factor_name") == MODEL)).to_dicts():
        notes.append(f"| {r['fold']} | {r['mean_ic']:.4f} | {r['mean_rank_ic']:.4f} | {r['icir']:.4f} | {r['rank_icir']:.4f} | {r['cumulative_spread']:+.2%} |")
    notes += ["", "## 因子跨 fold 稳定性", "", "排序仅用于本次回看；不应直接据此在同一测试集选因子后宣称样本外提升。mean_fold_rank_icir 是5个IR的均值，不是合并序列IR。", "", "| 因子 | 平均测试RankIC | RankIC训练方向一致fold | 定向spread正fold | 合计原始spread | 合计定向spread | 定向多头累计超额 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in stability.to_dicts():
        notes.append(f"| {r['factor_name']} | {r['mean_test_rank_ic']:.4f} | {r['aligned_rank_ic_folds']}/5 | {r['positive_directed_spread_folds']}/5 | {r['raw_cumulative_spread']:+.2%} | {r['directed_cumulative_spread']:+.2%} | {r['directed_long_cumulative_excess']:+.2%} |")
    factor_stability = stability.filter(pl.col("factor_name") != MODEL)
    model_metrics = stability.filter(pl.col("factor_name") == MODEL).row(0, named=True)
    notes += ["", "## 本次检查结论", "",
        f"- 组合模型真正的测试平均 RankIC 为 {model_metrics['mean_test_rank_ic']:.5f}；旧表约0.03236实际是 Pearson IC。5个fold中仅2个真正RankIC为正。",
        f"- 30个因子中，按训练RankIC定向后的70日累计spread为正的有 {factor_stability.filter(pl.col('directed_cumulative_spread') > 0).height} 个；5/5测试fold定向spread为正的有 {factor_stability.filter(pl.col('positive_directed_spread_folds') == 5).height} 个。",
        "- range_to_quote_volume_12：5/5 fold定向spread为正，4/5 fold RankIC沿用训练方向，是本区间较一致的尾部分组信号，但仍需核验低流动性交易成本与极端币贡献。",
        "- distance_to_high_24：4/5 fold定向spread为正，4/5 fold RankIC沿用训练方向，稳定性比只看累计收益更有说服力。",
        "- distance_to_high_2：5/5 fold定向spread为正，但只有1/5 fold RankIC与训练方向一致。这说明尾部收益差和整个截面的单调排序是两件事，不能仅看一个指标。",
        "- volatility_8：是模型高Gain因子，但按训练RankIC定向的单因子spread在4/5测试fold为负。模型重要性不等于独立因子盈利能力，也不能据此直接删除非线性模型特征。",
        f"- 组合模型每日原始累计spread {model_metrics['raw_cumulative_spread']:+.2%}，但多头相对均仓累计单利超额 {model_metrics['directed_long_cumulative_excess']:+.2%}。这与原long-only赚钱不矛盾：spread依赖两端，本报告也未模拟止损、费用和复投。",
        "- 正向极端收益可拉高Pearson IC/尾部均值，同时Spearman仍很弱或为负。不能由222.37%的历史复利收益推断模型具有稳定的全截面排序能力。",
    ]
    for fold in range(1, 6):
        notes += ["", f"## Fold {fold} 测试因子明细", "", "| 因子 | IC | RankIC | ICIR | RankICIR | 原始spread | 定向spread | Gain |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in summary.filter((pl.col("fold") == fold) & (pl.col("split") == "test") & (pl.col("factor_name") != MODEL)).sort("directed_cumulative_spread", descending=True).to_dicts():
            notes.append(f"| {r['factor_name']} | {r['mean_ic']:.4f} | {r['mean_rank_ic']:.4f} | {r['icir']:.4f} | {r['rank_icir']:.4f} | {r['cumulative_spread']:+.2%} | {r['directed_cumulative_spread']:+.2%} | {r['importance_gain_pct']:.2%} |")
    (output / "report.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    paths = [run / "config.json", run / "predictions.parquet", run / "feature_importance.parquet", run / "folds.parquet", factors_path] + [run / f"fold_{i:02d}_model.txt" for i in range(1, 6)]
    metadata = {"run_id": RUN, "generated_at": datetime.now(timezone.utc).isoformat(), "features": names, "model_parity": parity, "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths}, "summary_rows": summary.height, "daily_rows": daily.height, "cross_section_rows": cs.height, "tail_overlap_max": daily["tail_overlap_count"].max(), "feature_missing_daily_rows": daily.filter(pl.col("eligible_count") != pl.col("source_count")).height}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(stability.head(12))
    print(output)


if __name__ == "__main__":
    main()
