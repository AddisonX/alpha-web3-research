# 2025年训练、2026年预测：XGBoost Rank64策略

策略标识：`pit_rank64_xgboost_daily_excess_long_only@v1`。LightGBM研究策略和历史结果继续保留，这次使用独立模型、回测、报告目录。

## 数据与固定划分

- 复用已采集的 `data/research/pit_rank64_202501_202610/` 点时数据快照及2025-01起点固定120成员币池，不使用2026存续情况反筛。
- 训练集合：2025-01-01至2026-01-01，右端不包含。清除24h标签延伸至2026年的训练样本。
- 预测集合：2026-01-01至2027-01-01，右端不包含。仅评估当前已获得数据中完整兑现的24h窗口，不虚构尚未发生的月份。
- 不滚动、不划fold、不单独留验证集、不早停、不通过2026结果选择参数。

## 因子和标签

沿用前次固定的64因子清单，按每个日决策时点的币种截面计算平均秩百分比，输入范围0～1。缺失和缺口处理复用PIT Rank64数据构建器，排名不使用未来数据。

```text
rank_feature_i = (average_rank(factor_value_i) - 1) / (eligible_count - 1)
forward_return_i = open_i(t + 24h) / open_i(t) - 1
excess_forward_return_i = forward_return_i - mean(forward_return_universe)
```

symbol用于分组、排名和成交识别，不作为模型特征。预测阶段不因为未来标签缺失而删掉特征记录。

## 模型和策略

XGBoost `reg:squarederror`，`gbtree`，固定300棵树，一个模型一次拟合。预设参数为学习率0.03、最大深度4、min_child_weight100、L1=0.1、L2=10、行/列采样0.8、seed42、CPU histogram训练。模型以UBJSON保存，保存后重新加载并验证轮数和树数均为300。

每天北京时间12:00生成决策，选预测超额收益最高10%，向上取整，等仓只多。总目标仓位100%，手续费/滑点会留下少量现金。不设止损，不因跳空或盘中低点触发平仓，仅在每天调仓时卖出落选币或调整仓位。4h收盘价用于权益估值。

单边手续费5bp、滑点2bp，资金费率暂按0。原生4h执行模拟不等同于逐笔订单簿撮合。

## 命令与结果

```bash
uv run alpha-web3-research research run-pit-xgboost
uv run alpha-web3-research strategies sync
```

配置：`strategies/configs/pit_rank64_xgboost_2025_train_2026_test.v1.json`。`--research-config`可指定其他配置，300棵树作为本策略固定约束。

模型、预测与split manifest：`data/models/pit_rank64_xgboost_daily_excess_long_only/<run_id>/`。

回测账本、权益、月度结果：`data/backtests/pit_rank64_xgboost_daily_excess_long_only/<run_id>/`。

中文报告与可缩放权益比较图：`reports/strategies/pit_rank64_xgboost_daily_excess_long_only/<run_id>.md/.html`。

策略净值与币池基准收益按各日收益复利计算；累计超额单利为每日策略净收益减同期币池等权收益再相加，两种口径分开列示。

64因子名单曾通过2026-07数据筛选。虽然本次只拟合2025标签，2026依然存在历史特征选择带来的研究偏差，报告与manifest保留该信息。

模型参数参考：[XGBoost官方参数文档](https://xgboost.readthedocs.io/en/stable/parameter.html)。
