# 策略模块与历史注册表

## 1. 目标

策略不再等同于某个模型文件或某个 Freqtrade 类。项目将职责拆成五层：

```text
Data -> Model -> Strategy policy -> Execution/backtest -> Run artifacts
```

- Data：生成严格 as-of 的行情、特征和标签。
- Model：只负责从特征生成预测，不决定买卖。
- Strategy policy：把预测转换成离散 target position 或连续 target weight。
- Execution/backtest：处理调仓差额、成本、换手、成交和权益。
- Run artifacts：保存模型、预测、组合轨迹、指标与报告。

## 2. 稳定标识

每个策略使用：

```text
strategy_id@vN
```

修改以下任一行为时必须创建新版本，不能覆盖旧定义：

- Score 公式或预测输入契约。
- 选币规则。
- 方向 long-only / long-short。
- 目标仓位算法。
- 决策频率。
- 风险约束属于策略本身的部分。

只修改模型参数或重新训练时，保留 strategy key，创建新的 run。

## 3. 当前目录

| strategy key | 状态 | 说明 |
|---|---|---|
| `cross_sectional_4h_h6_daily_long_short@v1` | archived | 旧 4h/h6 每日前10%做多、后10%做空 |
| `cross_sectional_4h_h6_daily_long_only@v1` | deployed_legacy | 当前服务器兼容 dry-run 的旧 long-only 策略 |
| `hourly_multihorizon_absolute_score_long_only@v1` | research_active | 当前小时多期限绝对 Score 连续权重策略 |
| `pit_rank64_daily_excess_long_only@v1` | research_active | 2025起点币池、64因子排名、单模型未来日超额收益、前10%等仓只多 |
| `pit_rank64_xgboost_daily_excess_long_only@v1` | research_active | 2025年固定训练、2026年预测，XGBoost固定300树、不早停、Rank64超额收益 |

规范定义保存在：

```text
strategies/catalog/*.json
```

策略实现保存在：

```text
src/alpha_web3_research/strategies/
```

## 4. 历史运行索引

执行：

```bash
uv run alpha-web3-research strategies sync
```

注册表会扫描现有 manifest、walk-forward config 和 Freqtrade meta，不移动、不改写历史产物，并生成：

```text
data/strategies/strategy_registry.json
data/strategies/strategy_definitions.parquet
data/strategies/strategy_runs.json
data/strategies/strategy_runs.parquet
```

DuckDB 视图：

```text
strategy_definitions
strategy_runs
```

历史 artifact 是事实来源；运行索引可以随时重新生成。

## 5. 查询命令

```bash
uv run alpha-web3-research strategies list

uv run alpha-web3-research strategies show \
  hourly_multihorizon_absolute_score_long_only

uv run alpha-web3-research strategies runs

uv run alpha-web3-research strategies runs \
  --strategy cross_sectional_4h_h6_daily_long_only
```

## 6. 新增策略

1. 在 `src/alpha_web3_research/strategies/` 实现独立 policy。
2. 为策略定义稳定的模型输入和目标仓位输出契约。
3. 在 `strategies/catalog/` 新增版本化 JSON，不修改旧版本语义。
4. 训练/回测 manifest 写入 `strategy_key`。
5. 为历史运行添加 discovery rule。
6. 增加策略行为、tie-breaking、仓位上限和注册表发现测试。
7. 运行 `strategies sync` 更新索引。

## 7. 当前兼容边界

旧 Freqtrade 文件继续保留，因为东京服务器仍在运行旧 long-only dry-run。其选币规则的规范实现已经抽到 `DailyCrossSectionalQuantileStrategy`，Freqtrade exporter 通过该 policy 生成目标方向。

小时策略的 Score 与目标权重规则位于 `HourlyMultiHorizonAbsoluteScoreLongOnlyStrategy`；LightGBM 训练器不再拥有 Score 公式，小时回测器也不再拥有选币/权重公式。
