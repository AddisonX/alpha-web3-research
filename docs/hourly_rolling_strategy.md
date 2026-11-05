# 小时滚动多期限策略

## 1. 决策定义

策略每小时重新回答一次：在当前已知信息下，未来一小时应该持有哪些币、各持有多少。1h、4h、12h 和 24h 是预测期限，不是固定持仓周期。

每小时执行顺序：

1. 读取刚刚完全收盘的 1h K 线。
2. 使用最近四根 1h K 线重构当前滚动 4h K 线。
3. 生成 `1h__*` 和 `4h__*` 因子。
4. 四个期限分别预测累计收益和未来路径波动。
5. 合成绝对 Score。
6. 根据 Score 和风险计算目标权重。
7. 与当前漂移后权重比较，只交易必要差额。

## 2. 时间与 4h 重构

原始 OHLCV 的 `timestamp` 是 K 线开始时间。模型使用收盘后时点：

```text
decision_timestamp = timestamp + 1h
```

在决策时点 t，滚动 4h K 线由 `[t-4h, t)` 内四根已收盘 1h K 线构成：open 取第一根、high 取最大值、low 取最小值、close 取最后一根、volume 和 quote volume 求和。

为了让 4h 因子窗口仍代表低频尺度，历史滚动 4h 快照按 `decision_timestamp mod 4h` 分成四个相位。每个相位的相邻样本相隔四小时，例如 `4h__return_6` 比较当前与 24 小时前的滚动 4h close。

## 3. 标签与模型

四个 horizon bundle：

```text
h1, h4, h12, h24
```

每个 bundle 包含：

- LightGBM return model：预测从 t 到 t+h 的累计收益。
- LightGBM volatility model：预测未来 h 小时 log return 路径的 root-sum-square 风险。

因此实现上是四个期限、每个期限两棵模型，共八个 LightGBM booster。收益和风险预测在 validation 上进行线性/比例校准。

数据只切分一次：70% train、15% validation、15% test。train/validation 和 validation/test 边界前 purge 24 小时，避免最长标签跨越边界。不生成 fold，不拼接 fold 预测。

## 4. Score

每个期限：

```text
Z_h = (predicted_return_h - 0.14% round-trip cost) / predicted_volatility_h
```

综合分数：

```text
Score = mean(Z_1, Z_4, Z_12, Z_24) - 0.25 * std(Z_1, Z_4, Z_12, Z_24)
```

当前不包含模型不确定性 U。四期限暂时等权；后续只有在新增样本外证据后才调整期限权重。

选币不使用横截面排名：新仓只有 `Score >= entry_threshold` 才可进入；已有仓位在 `Score >= 0.7 * entry_threshold` 时继续持有。

## 5. 目标权重

符合阈值的币种计算：

```text
raw_weight_i = max(Score_i - exit_threshold, 0) / predicted_volatility_h24_i
```

原始权重归一化到最多 90% 总仓位，并进行 12% 单币上限的迭代再分配。没有最大持仓数量；当很多币同时超过阈值时都可以获得非零权重，当信号稀少时剩余资金保持现金。

## 6. 换手控制

- 进入阈值和较低退出阈值构成迟滞区间。
- 普通目标权重变化不足 2 个百分点不交易。
- 普通单币每小时最多变化 4 个百分点。
- 整个组合每小时单边换手不超过 25%。
- 减仓优先于加仓。
- 清仓和修复超过 12% 的漂移仓位不受 2% 最小交易死区限制。
- 每次实际交易按 0.07% 单边成本扣减。

## 7. 回测语义

test 中每个小时严格按顺序模拟：

1. 使用 t 时刻预测与当前漂移仓位生成目标。
2. 执行受约束的仓位差额并扣除成本。
3. 用 t 到 t+1 的真实收益计算组合收益。
4. 持仓随市场收益漂移到下一小时。
5. 在 t+1 使用新预测重复。

validation 只选择绝对 Score 阈值，test 不参与参数选择。报告应同时查看收益、回撤、换手、平均仓位、持仓数量和币种贡献集中度。

## 8. 当前限制

- latest universe 存在 survivorship bias。
- 当前 test 只有约 27 天，证据不足以支持实盘结论。
- 资金费率和市场冲击尚未建模，0.07% 单边成本只是统一近似。
- 四期限预测高度相关，等权并不是最终答案。
- U 尚未进入 Score，模型不确定性仍未显式控制。
- 新策略尚未接入 Freqtrade position adjustment；当前结果来自仓位级研究回测器。
