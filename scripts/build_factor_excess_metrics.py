from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "factor_metrics_4h_h6_top10_directional_20260704"
DAY_MS = 86_400_000
BAR_MS = 14_400_000


def retained_features() -> set[str]:
    matrix = pd.read_csv(
        ROOT / "reports/factors/factor_correlation_4h_20260704_spearman.csv",
        index_col=0,
    )
    names = list(matrix.columns)
    random.Random(42).shuffle(names)
    selected: list[str] = []
    for name in names:
        if not any(abs(matrix.at[name, peer]) > 0.99 for peer in selected):
            selected.append(name)
    assert len(selected) == 64
    return set(selected)


def local_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).astimezone(
        ZoneInfo("Asia/Shanghai")
    ).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    output_dir = ROOT / "reports/factors"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        pl.read_parquet(ROOT / "data/evaluation/single_factor_summary.parquet")
        .filter((pl.col("timeframe") == "4h") & (pl.col("horizon_bars") == 6))
        .select(
            "factor_name", "factor_family", "mean_rank_ic", "std_rank_ic",
            "rank_icir", "timestamp_count", "label_column",
        )
        .sort("factor_name")
    )
    assert summary.height == 83
    assert summary.get_column("label_column").unique().to_list() == ["forward_return"]
    directions = summary.select(
        "factor_name",
        pl.when(pl.col("mean_rank_ic") >= 0).then(1).otherwise(-1).alias("direction_sign"),
    )
    labels = (
        pl.scan_parquet(ROOT / "data/labels/latest_labels.parquet")
        .filter(
            (pl.col("timeframe") == "4h")
            & (pl.col("horizon_bars") == 6)
            & ((pl.col("timestamp") % DAY_MS) == 0)
            & pl.col("forward_return").is_finite()
        )
        .select("symbol", "timestamp", "close", "future_close", "forward_return")
        .collect()
    )
    assert labels.select(pl.struct("symbol", "timestamp").n_unique()).item() == labels.height
    benchmark = labels.group_by("timestamp").agg(
        pl.col("forward_return").mean().alias("universe_return"),
        pl.len().alias("universe_count"),
    )
    factors = (
        pl.scan_parquet(ROOT / "data/factors/latest_factors.parquet")
        .filter(
            (pl.col("timeframe") == "4h")
            & ((pl.col("timestamp") % DAY_MS) == 0)
            & pl.col("factor_value").is_finite()
        )
        .select("symbol", "timestamp", "factor_name", "factor_value")
        .collect()
        .join(labels.select("symbol", "timestamp", "forward_return"), on=["symbol", "timestamp"])
        .join(directions, on="factor_name")
        .with_columns((pl.col("factor_value") * pl.col("direction_sign")).alias("directed_value"))
    )
    groups = ["factor_name", "timestamp"]
    # Keep only dates on which every factor can select from the complete benchmark pool.
    full_groups = (
        factors.group_by(groups).len(name="eligible_count")
        .join(benchmark.select("timestamp", "universe_count"), on="timestamp")
        .filter(pl.col("eligible_count") == pl.col("universe_count"))
        .group_by("timestamp").len()
        .filter(pl.col("len") == 83)
        .select("timestamp")
    )
    factors = factors.join(full_groups, on="timestamp")
    selected = (
        factors.sort(
            ["factor_name", "timestamp", "directed_value", "symbol"],
            descending=[False, False, True, False],
        )
        .with_columns(
            pl.col("symbol").cum_count().over(groups).alias("selection_rank"),
            pl.len().over(groups).alias("eligible_count"),
        )
        .with_columns((pl.col("eligible_count") * 0.10).ceil().cast(pl.UInt32).alias("target_count"))
        .filter(pl.col("selection_rank") <= pl.col("target_count"))
    )
    daily = (
        selected.group_by(groups).agg(
            pl.col("forward_return").mean().alias("selected_return"),
            pl.col("symbol").alias("selected_symbols"),
            pl.len().alias("selected_count"),
        )
        .join(benchmark, on="timestamp")
        .with_columns((pl.col("selected_return") - pl.col("universe_return")).alias("excess_return"))
        .sort(groups)
        .with_columns(
            pl.col("excess_return").cum_sum().over("factor_name").alias("cumulative_excess_simple_return"),
            (pl.col("timestamp") + BAR_MS).alias("decision_timestamp"),
            (pl.col("timestamp") + BAR_MS + DAY_MS).alias("exit_timestamp"),
        )
    )
    dates = daily.get_column("timestamp").unique().sort().to_list()
    assert dates and all(b - a == DAY_MS for a, b in zip(dates, dates[1:])), "daily schedule has gaps"
    expected_days = len(dates)
    assert daily.height == 83 * expected_days
    assert (daily.get_column("selected_count") == (daily.get_column("universe_count") * 0.1).ceil()).all()
    assert np.allclose(
        daily.get_column("excess_return").to_numpy(),
        daily.get_column("selected_return").to_numpy() - daily.get_column("universe_return").to_numpy(),
    )
    aggregates = daily.group_by("factor_name").agg(
        pl.col("selected_return").sum().alias("selected_cumulative_simple_return"),
        pl.col("universe_return").sum().alias("universe_cumulative_simple_return"),
        pl.col("excess_return").sum().alias("cumulative_excess_simple_return"),
        pl.len().alias("daily_period_count"),
    )
    retained = retained_features()
    table = (
        summary.join(aggregates, on="factor_name")
        .join(directions, on="factor_name")
        .with_columns(pl.col("factor_name").is_in(retained).alias("retained_after_rank_corr_099"))
        .sort("cumulative_excess_simple_return", descending=True)
    )
    for row in table.iter_rows(named=True):
        assert math.isclose(row["rank_icir"], row["mean_rank_ic"] / row["std_rank_ic"], abs_tol=1e-12)
        assert math.isclose(
            row["cumulative_excess_simple_return"],
            row["selected_cumulative_simple_return"] - row["universe_cumulative_simple_return"],
            abs_tol=1e-10,
        )

    csv_path = output_dir / f"{PREFIX}.csv"
    selected_csv_path = output_dir / f"{PREFIX}_retained64.csv"
    report_path = output_dir / f"{PREFIX}.md"
    daily_path = output_dir / f"{PREFIX}_daily.parquet"
    json_path = output_dir / f"{PREFIX}_workbook.json"
    table.write_csv(csv_path)
    table.filter(pl.col("retained_after_rank_corr_099")).write_csv(selected_csv_path)
    daily.write_parquet(daily_path)

    lines = [
        "# 因子指标表：RankIC、RankICIR 与累计超额单利收益",
        "",
        "数据快照：2026-07-04。4h 因子预测未来 6 根 K 线（24h）原始收益。",
        "",
        "## 计算口径",
        "",
        "- RankIC：当时全样本各 4h 截面 Spearman 因子收益相关系数的时间等权平均，与上一份历史报告一致。",
        "- RankICIR：上述 RankIC 序列均值除以样本标准差（ddof=1），未年化。",
        "- RankIC、RankICIR 保留原始因子方向的符号；只在选币时按符号调整方向，不把报告指标统一翻成正值。",
        "- 方向：按全样本平均 RankIC 的正负号，正向取因子值最高 10%，负向取最低 10%。未使用当天未来 IC 切换方向。",
        "- 选币数：向上取整 10% × 有效币数；同值按 symbol 字典序打破并列，选中币种等权。",
        "- 基准：同一时点有完整 24h 标签的币池全币种等权收益。为保证可比性，仅保留全部 83 个因子都覆盖完整基准池的日期。",
        f"- 收益区间（北京时间）：{local_time(dates[0] + BAR_MS)} 至 {local_time(dates[-1] + BAR_MS + DAY_MS)}，共 {expected_days} 个不重叠日收益窗口。每天 12:00 换仓。",
        "- 单期超额 = 选中币种等权 24h 收益 − 同期币池等权 24h 收益。累计超额 = 各期超额相加，不复利。",
        "- RankIC 采用冻结评估的完整 4h 序列；累计收益采用统一的非重叠日序列。两列采样频率不同，收益区间因最长窗口预热而更短。",
        "- 全样本确定方向属于回看诊断，不能据此称为严格样本外策略收益。未计手续费、滑点、资金费率和信号生成后的成交价差；保留当前 universe 的幸存者偏差。",
        "",
        "## 随机去重后保留的 64 个因子",
        "",
        "按累计超额单利收益降序。负方向表示买入因子值最低的 10%。",
        "",
        "| 因子 | 方向 | RankIC | RankICIR | 累计超额单利收益 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in table.filter(pl.col("retained_after_rank_corr_099")).iter_rows(named=True):
        direction = "最高10%" if row["direction_sign"] == 1 else "最低10%"
        lines.append(
            f"| `{row['factor_name']}` | {direction} | {row['mean_rank_ic']:.5f} | "
            f"{row['rank_icir']:.5f} | {row['cumulative_excess_simple_return']:+.2%} |"
        )
    lines += [
        "", "## 全部 83 个因子", "",
        "| 因子 | 保留64集合 | 方向 | RankIC | RankICIR | 累计超额单利收益 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in table.iter_rows(named=True):
        direction = "最高10%" if row["direction_sign"] == 1 else "最低10%"
        kept = "是" if row["retained_after_rank_corr_099"] else "否"
        lines.append(
            f"| `{row['factor_name']}` | {kept} | {direction} | {row['mean_rank_ic']:.5f} | "
            f"{row['rank_icir']:.5f} | {row['cumulative_excess_simple_return']:+.2%} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "rows": table.to_dicts(),
        "return_start": local_time(dates[0] + BAR_MS),
        "return_end": local_time(dates[-1] + BAR_MS + DAY_MS),
        "daily_period_count": expected_days,
        "report_path": str(report_path),
        "source": "2026-07-04 冻结 single_factor_summary 与因子/标签 Parquet",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rows={table.height}; retained={len(retained)}; daily_windows={expected_days}")
    print(f"return_period={payload['return_start']} to {payload['return_end']} Asia/Shanghai")
    print(table.select("factor_name", "mean_rank_ic", "rank_icir", "cumulative_excess_simple_return").head(12))
    print(f"positive_excess_count={table.filter(pl.col('cumulative_excess_simple_return') > 0).height}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
