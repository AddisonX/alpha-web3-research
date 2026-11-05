from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


FAMILY_NAMES = {
    "momentum_reversal": "动量/反转",
    "volatility_range": "波动率/振幅",
    "volume": "成交量",
    "liquidity_proxy": "流动性代理",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the historical 83-factor correlation report.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--snapshot", default="20260704")
    return parser.parse_args()


def matrix_pairs(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    ic_corr: pd.DataFrame,
    counts: pd.DataFrame,
    ic_counts: pd.DataFrame,
    families: dict[str, str],
    selected: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    names = list(pearson.columns)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            rows.append(
                {
                    "factor_left": left,
                    "factor_right": right,
                    "family_left": families[left],
                    "family_right": families[right],
                    "same_family": families[left] == families[right],
                    "left_selected": left in selected,
                    "right_selected": right in selected,
                    "pairwise_observations": int(counts.at[left, right]),
                    "rank_ic_pairwise_timestamps": int(ic_counts.at[left, right]),
                    "pearson": pearson.at[left, right],
                    "abs_pearson": abs(pearson.at[left, right]),
                    "spearman": spearman.at[left, right],
                    "abs_spearman": abs(spearman.at[left, right]),
                    "rank_ic_corr": ic_corr.at[left, right],
                    "abs_rank_ic_corr": abs(ic_corr.at[left, right]),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_pearson", ascending=False, ignore_index=True)


def connected_components(corr: pd.DataFrame, threshold: float) -> list[list[str]]:
    names = list(corr.columns)
    neighbors = {name: set() for name in names}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            value = corr.at[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                neighbors[left].add(right)
                neighbors[right].add(left)

    components: list[list[str]] = []
    unseen = set(names)
    while unseen:
        root = unseen.pop()
        stack = [root]
        component = [root]
        while stack:
            current = stack.pop()
            linked = neighbors[current] & unseen
            unseen -= linked
            stack.extend(linked)
            component.extend(linked)
        if len(component) > 1:
            components.append(sorted(component))
    return sorted(components, key=lambda item: (-len(item), item))


def threshold_counts(pairs: pd.DataFrame, column: str) -> dict[float, int]:
    return {threshold: int((pairs[column] >= threshold).sum()) for threshold in (0.5, 0.7, 0.85, 0.95)}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def corr(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.4f}"


def top_pair_table(pairs: pd.DataFrame, metric: str, count: int = 15) -> list[str]:
    subset = pairs.sort_values(metric, ascending=False).head(count)
    signed = metric.removeprefix("abs_")
    lines = [
        "| 排名 | 因子 A | 因子 B | 类别 | 相关系数 | 共同样本 |",
        "|---:|---|---|---|---:|---:|",
    ]
    for rank, row in enumerate(subset.itertuples(index=False), start=1):
        family = "同族" if row.same_family else "跨族"
        observations = (
            row.rank_ic_pairwise_timestamps
            if metric == "abs_rank_ic_corr"
            else row.pairwise_observations
        )
        lines.append(
            f"| {rank} | `{row.factor_left}` | `{row.factor_right}` | {family} | "
            f"{corr(getattr(row, signed))} | {observations:,} |"
        )
    return lines


def build_heatmap(
    matrices: list[tuple[str, pd.DataFrame]],
    order: list[str],
    output_path: Path,
) -> None:
    ordered = [(name, matrix.loc[order, order]) for name, matrix in matrices]
    first_name, first_matrix = ordered[0]
    figure = go.Figure(
        data=[
            go.Heatmap(
                z=first_matrix.to_numpy(),
                x=order,
                y=order,
                zmin=-1,
                zmax=1,
                zmid=0,
                colorscale="RdBu",
                reversescale=True,
                colorbar={"title": "Correlation"},
                hovertemplate="%{y}<br>%{x}<br>corr=%{z:.4f}<extra></extra>",
            )
        ]
    )
    buttons = []
    for name, matrix in ordered:
        buttons.append(
            {
                "label": name,
                "method": "update",
                "args": [{"z": [matrix.to_numpy()]}, {"title": name}],
            }
        )
    figure.update_layout(
        title=first_name,
        template="plotly_white",
        width=1500,
        height=1350,
        margin={"l": 230, "r": 70, "t": 120, "b": 230},
        xaxis={"tickfont": {"size": 8}, "tickangle": -55},
        yaxis={"tickfont": {"size": 8}, "autorange": "reversed"},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
    )
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    factor_path = root / "data/factors/latest_factors.parquet"
    timeseries_path = root / "data/evaluation/single_factor_timeseries.parquet"
    decisions_path = root / "data/evaluation/decorrelation_decisions.parquet"
    production_manifest_path = (
        root
        / "data/models/production/walk_forward_4h_h6_forward_return_20260704T123039Z/manifest.json"
    )
    output_dir = root / "reports/factors"
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_long = (
        pl.scan_parquet(factor_path)
        .filter(pl.col("timeframe") == args.timeframe)
        .select(["symbol", "timestamp", "factor_name", "factor_family", "factor_value"])
        .collect()
    )
    metadata = (
        factor_long.select(["factor_name", "factor_family"])
        .unique()
        .sort(["factor_family", "factor_name"])
    )
    factor_names = metadata.get_column("factor_name").to_list()
    families = dict(metadata.iter_rows())
    if len(factor_names) != 83:
        raise ValueError(f"expected 83 factors, got {len(factor_names)}")

    wide = factor_long.pivot(
        index=["symbol", "timestamp"],
        on="factor_name",
        values="factor_value",
    ).select(["symbol", "timestamp", *factor_names])
    values = wide.select(factor_names).to_pandas().replace([np.inf, -np.inf], np.nan)
    pearson = values.corr(method="pearson", min_periods=100).loc[factor_names, factor_names]
    spearman = values.corr(method="spearman", min_periods=100).loc[factor_names, factor_names]
    valid = values.notna().to_numpy(dtype=np.int32)
    counts = pd.DataFrame(valid.T @ valid, index=factor_names, columns=factor_names)

    ic_long = (
        pl.scan_parquet(timeseries_path)
        .filter(
            (pl.col("timeframe") == args.timeframe)
            & (pl.col("horizon_bars") == args.horizon)
            & pl.col("factor_name").is_in(factor_names)
        )
        .select(["timestamp", "factor_name", "rank_ic"])
        .collect()
    )
    ic_wide = ic_long.pivot(index="timestamp", on="factor_name", values="rank_ic")
    ic_values = ic_wide.select(factor_names).to_pandas().replace([np.inf, -np.inf], np.nan)
    ic_corr = ic_values.corr(method="pearson", min_periods=20).loc[factor_names, factor_names]
    ic_valid = ic_values.notna().to_numpy(dtype=np.int32)
    ic_counts = pd.DataFrame(ic_valid.T @ ic_valid, index=factor_names, columns=factor_names)

    decisions = pl.read_parquet(decisions_path).filter(
        (pl.col("timeframe") == args.timeframe) & (pl.col("horizon_bars") == args.horizon)
    )
    selected = set(decisions.filter(pl.col("selected")).get_column("factor_name").to_list())
    pairs = matrix_pairs(pearson, spearman, ic_corr, counts, ic_counts, families, selected)

    distance = 1.0 - pearson.abs().fillna(0.0).to_numpy()
    distance = np.clip((distance + distance.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    cluster_order = [factor_names[index] for index in leaves_list(linkage(squareform(distance), method="average"))]

    prefix = f"factor_correlation_{args.timeframe}_{args.snapshot}"
    pearson_path = output_dir / f"{prefix}_pearson.csv"
    spearman_path = output_dir / f"{prefix}_spearman.csv"
    ic_path = output_dir / f"{prefix}_rank_ic.csv"
    pairs_path = output_dir / f"{prefix}_pairs.csv"
    heatmap_path = output_dir / f"{prefix}_heatmap.html"
    report_path = output_dir / f"{prefix}_report.md"
    pearson.to_csv(pearson_path, index_label="factor_name")
    spearman.to_csv(spearman_path, index_label="factor_name")
    ic_corr.to_csv(ic_path, index_label="factor_name")
    pairs.to_csv(pairs_path, index=False)
    build_heatmap(
        [
            ("因子值 Pearson", pearson),
            ("因子值 Spearman", spearman),
            (f"{args.timeframe} h{args.horizon} RankIC 序列 Pearson", ic_corr),
        ],
        cluster_order,
        heatmap_path,
    )

    pearson_counts = threshold_counts(pairs, "abs_pearson")
    spearman_counts = threshold_counts(pairs, "abs_spearman")
    ic_counts = threshold_counts(pairs, "abs_rank_ic_corr")
    components = connected_components(pearson, 0.85)
    same_family = pairs[pairs["same_family"]]
    cross_family = pairs[~pairs["same_family"]]
    selected_pairs = pairs[pairs["left_selected"] & pairs["right_selected"]]
    selected_count = int(decisions.get_column("selected").sum())
    value_excluded = decisions.filter(pl.col("exclusion_reason").str.starts_with("value_corr")).height
    ic_excluded = decisions.filter(pl.col("exclusion_reason").str.starts_with("ic_corr")).height

    factor_stats = []
    for name in factor_names:
        related = pairs[(pairs["factor_left"] == name) | (pairs["factor_right"] == name)]
        factor_stats.append(
            {
                "factor_name": name,
                "family": families[name],
                "mean_abs_pearson": related["abs_pearson"].mean(),
                "max_abs_pearson": related["abs_pearson"].max(),
                "peers_abs_pearson_ge_085": int((related["abs_pearson"] >= 0.85).sum()),
            }
        )
    factor_stats_frame = pd.DataFrame(factor_stats).sort_values("mean_abs_pearson", ascending=False)
    family_pairs = (
        pairs.assign(
            family_pair=pairs.apply(
                lambda row: " × ".join(
                    sorted(
                        [
                            FAMILY_NAMES.get(row["family_left"], row["family_left"]),
                            FAMILY_NAMES.get(row["family_right"], row["family_right"]),
                        ]
                    )
                ),
                axis=1,
            )
        )
        .groupby("family_pair", as_index=False)
        .agg(
            pair_count=("factor_left", "size"),
            mean_abs_pearson=("abs_pearson", "mean"),
            mean_abs_spearman=("abs_spearman", "mean"),
            pearson_ge_085=("abs_pearson", lambda values: int((values >= 0.85).sum())),
        )
        .sort_values("mean_abs_pearson", ascending=False)
    )
    divergence = pairs.assign(
        pearson_spearman_abs_gap=(pairs["abs_pearson"] - pairs["abs_spearman"]).abs()
    ).sort_values("pearson_spearman_abs_gap", ascending=False)

    factor_start = pd.to_datetime(factor_long.get_column("timestamp").min(), unit="ms", utc=True)
    factor_end = pd.to_datetime(factor_long.get_column("timestamp").max(), unit="ms", utc=True)
    ic_start = pd.to_datetime(ic_long.get_column("timestamp").min(), unit="ms", utc=True)
    ic_end = pd.to_datetime(ic_long.get_column("timestamp").max(), unit="ms", utc=True)
    symbols = factor_long.get_column("symbol").n_unique()
    observations = wide.height
    production_features = json.loads(production_manifest_path.read_text())["source_config"]["features"]

    report: list[str] = [
        "# 83 个候选因子相关性报告（2026-07-04 历史快照）",
        "",
        "> 本报告只使用 2026-07-04 当时冻结的因子、单因子 RankIC 和去相关决策产物，不加入后续实时数据。",
        "",
        "## 1. 计算口径",
        "",
        f"- 因子集合：`{len(factor_names)}` 个 4h 候选因子，覆盖 `{symbols}` 个 universe 币种。",
        f"- 因子值区间：`{factor_start.isoformat()}` 至 `{factor_end.isoformat()}`。",
        f"- 因子值观测单位：每个 `symbol × 4h timestamp` 一行，共 `{observations:,}` 行；缺失值按 pairwise complete observations 处理，最少共同样本为 100。",
        f"- RankIC 区间：`{ic_start.isoformat()}` 至 `{ic_end.isoformat()}`，目标为 4h `horizon={args.horizon}` 的 `forward_return`。",
        "- Pearson 因子值矩阵：与当时去相关代码的主矩阵口径一致，衡量原始因子值的线性冗余。",
        "- Spearman 因子值矩阵：衡量排序冗余，对极端值和不同量纲更稳健。",
        "- RankIC 相关矩阵：衡量两个因子的预测能力是否在相同时间增强或失效，不等同于因子值相关。",
        "- 当时的因子值 Pearson 使用了完整因子历史，而不是只截取有标签的 2026-01 至 2026-07 区间。这是对原去相关实现的复原。",
        "",
        "## 2. 总体结果",
        "",
        f"- 83 个因子共有 `{len(pairs):,}` 个不重复因子对。",
        f"- 因子值 Pearson：`|r|≥0.50` 有 `{pearson_counts[0.5]}` 对，`|r|≥0.70` 有 `{pearson_counts[0.7]}` 对，`|r|≥0.85` 有 `{pearson_counts[0.85]}` 对，`|r|≥0.95` 有 `{pearson_counts[0.95]}` 对。",
        f"- 因子值 Spearman：`|ρ|≥0.50` 有 `{spearman_counts[0.5]}` 对，`|ρ|≥0.70` 有 `{spearman_counts[0.7]}` 对，`|ρ|≥0.85` 有 `{spearman_counts[0.85]}` 对，`|ρ|≥0.95` 有 `{spearman_counts[0.95]}` 对。",
        f"- RankIC 序列：`|r|≥0.50` 有 `{ic_counts[0.5]}` 对，`|r|≥0.70` 有 `{ic_counts[0.7]}` 对，`|r|≥0.85` 有 `{ic_counts[0.85]}` 对，`|r|≥0.95` 有 `{ic_counts[0.95]}` 对。",
        f"- 同族因子平均绝对 Pearson 为 `{same_family['abs_pearson'].mean():.3f}`，跨族因子为 `{cross_family['abs_pearson'].mean():.3f}`。",
        f"- 按 `|Pearson|≥0.85` 连边后形成 `{len(components)}` 个包含至少两个因子的相关簇，最大簇包含 `{max(map(len, components), default=1)}` 个因子。",
        f"- 当时贪心去相关最终保留 `{selected_count}/83` 个因子；`{value_excluded}` 个因值相关超阈值被排除，另有 `{ic_excluded}` 个因 RankIC 序列相关超阈值被排除。",
        f"- 保留下来的 30 个因子内部，最大绝对因子值 Pearson 为 `{selected_pairs['abs_pearson'].max():.4f}`，最大绝对 RankIC 相关为 `{selected_pairs['abs_rank_ic_corr'].max():.4f}`。",
        "",
        "### 因子族之间的相关结构",
        "",
        "| 因子族组合 | 因子对数 | 平均绝对 Pearson | 平均绝对 Spearman | `|Pearson|≥0.85` |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in family_pairs.itertuples(index=False):
        report.append(
            f"| {row.family_pair} | {row.pair_count} | {row.mean_abs_pearson:.3f} | "
            f"{row.mean_abs_spearman:.3f} | {row.pearson_ge_085} |"
        )

    report.extend([
        "",
        "## 3. 最高因子值 Pearson 相关对",
        "",
        *top_pair_table(pairs, "abs_pearson", 20),
        "",
        "## 4. 最高因子值 Spearman 相关对",
        "",
        *top_pair_table(pairs, "abs_spearman", 15),
        "",
        "## 5. 最高 RankIC 时间序列相关对",
        "",
        *top_pair_table(pairs, "abs_rank_ic_corr", 15),
        "",
        "### Pearson 与 Spearman 差异最大的因子对",
        "",
        "这类组合通常存在明显非线性、长尾或异常值影响，不能只依赖 Pearson 判断冗余。",
        "",
        "| 因子 A | 因子 B | Pearson | Spearman | 绝对相关差 |",
        "|---|---|---:|---:|---:|",
    ])
    for row in divergence.head(15).itertuples(index=False):
        report.append(
            f"| `{row.factor_left}` | `{row.factor_right}` | {row.pearson:.4f} | "
            f"{row.spearman:.4f} | {row.pearson_spearman_abs_gap:.4f} |"
        )

    report.extend([
        "",
        "## 6. 冗余程度最高的因子",
        "",
        "| 因子 | 因子族 | 平均绝对 Pearson | 最大绝对 Pearson | `|r|≥0.85` 邻居数 |",
        "|---|---|---:|---:|---:|",
    ])
    for row in factor_stats_frame.head(20).itertuples(index=False):
        report.append(
            f"| `{row.factor_name}` | {FAMILY_NAMES.get(row.family, row.family)} | "
            f"{row.mean_abs_pearson:.3f} | {row.max_abs_pearson:.3f} | {row.peers_abs_pearson_ge_085} |"
        )

    report.extend(["", "## 7. 高相关簇", ""])
    for index, component in enumerate(components, start=1):
        report.append(f"- 簇 {index}（{len(component)} 个）：" + "、".join(f"`{name}`" for name in component))

    report.extend(
        [
            "",
            "## 8. 相关性分析",
            "",
            "1. **窗口变体是主要冗余来源。** 同一种经济含义在 2、3、4、6、8、12、24 根窗口上展开，邻近窗口通常高度相关。更多窗口不等于更多独立信息，主要是在重复描述相同的趋势、波动或成交量状态。",
            "2. **价格位置、均线偏离和累计收益存在结构性重叠。** `return_*`、`ma_gap_*`、`distance_to_high_*`、`distance_to_low_*` 都由同一段 close 路径变换而来。它们可能方向不同，但绝对相关很高，适合按相关簇选代表变量，而不是全部送入模型。",
            "3. **Pearson 与 Spearman 的差距能暴露异常值影响。** volume change 和流动性代理容易出现长尾。若 Pearson 高而 Spearman 明显较低，线性相关可能被少数极端样本推动；若 Spearman 仍高，则排序信息也高度重复。",
            "4. **因子值不相关不代表预测信息独立。** 当时还有 10 个候选因子不是因子值相关超限，而是 RankIC 时间序列相关超限，说明它们数值形态不同，但在相同市场阶段一起有效或失效。",
            "5. **30 因子集合通过了当时阈值，但不是统计独立。** 阈值 0.85 只去掉极强冗余，中等相关仍大量存在；LightGBM 可以处理一部分冗余，但重要性会在替代变量之间分摊，解释单个 gain 时要谨慎。",
            "6. **原矩阵混合了时间序列和横截面相关。** 把不同币种、不同时点池化后计算，会同时包含市场共同趋势、币种固定差异和横截面排序关系。若目标是横截面选币，下一版更严格的主指标应是每个时点先做截面 rank 或 z-score，再汇总相关性。",
            "7. **原去相关存在研究期信息泄漏。** 这 83 个因子的完整历史矩阵和完整 RankIC 序列先被用于筛选，再运行早期 walk-forward fold。相关矩阵本身计算无误，但用于严格样本外选择时，应在每个训练 fold 内单独重算。",
            "",
            "## 9. 建议",
            "",
            "- 保留 Pearson 作为与历史流程兼容的基准，同时把截面 Spearman 相关作为主要去冗余指标。",
            "- 每个 walk-forward fold 只使用训练窗口计算相关矩阵和相关簇，验证期与测试期不得参与选因子。",
            "- 对每个高相关簇优先保留经济含义清楚、RankIC 稳定、换手较低、缺失率较低的代表因子。",
            "- 对生产 30 因子做滚动相关监控；当相关结构显著抬升时，降低模型有效特征数预期并重新校准风险。",
            "",
            "## 10. 产物",
            "",
            f"- Pearson 矩阵：`{pearson_path.relative_to(root)}`",
            f"- Spearman 矩阵：`{spearman_path.relative_to(root)}`",
            f"- RankIC 相关矩阵：`{ic_path.relative_to(root)}`",
            f"- 全部因子对：`{pairs_path.relative_to(root)}`",
            f"- 交互热力图：`{heatmap_path.relative_to(root)}`",
            "",
            f"生产 manifest 中的 30 个特征与去相关保留集合重合 `{len(set(production_features) & selected)}/30`。",
        ]
    )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"report={report_path}")
    print(f"heatmap={heatmap_path}")
    print(f"pairs={pairs_path}")
    print(f"pearson_ge_085={pearson_counts[0.85]}")
    print(f"spearman_ge_085={spearman_counts[0.85]}")
    print(f"rank_ic_ge_085={ic_counts[0.85]}")


if __name__ == "__main__":
    main()
