# 2025 起点币池 Rank64 单模型只多策略

策略标识：`pit_rank64_daily_excess_long_only@v1`。

## 与旧策略的关系

这是一套独立研究流程。原有三种策略的类、配置、生产模型、最新 universe、因子与标签文件继续保留。新数据位于 `data/research/pit_rank64_202501_202610/`，模型与回测按 strategy id/run id 分目录，不写入旧 `latest_*` 或生产模型指针，不更新东京运行环境。

## 数据与起点币池

- 研究区间：2025-01-01 至 2026-10-31，结束边界为 2026-11-01。数据日期采用 UTC，未到来的日期不会生成行情。
- 额外预热历史：2024-07-01 至 2024-12-31，用于站在 2025-01-01 构建币池和初期因子。
- 来源：Binance 官方 USD-M USDT 合约 4h 历史归档。历史目录发现候选，不依赖今天仍活跃的 exchange markets。每个 ZIP 根据官方 `.CHECKSUM` 验证 SHA256，来源保存在 acquisition/data manifest 中。
- 入池时点：2025-01-01 00:00 UTC。只使用 `bar_open + 4h <= inception` 的已收盘数据。
- 门槛：至少 180 天历史，近180天4h覆盖率不少于98%，入池前最后一根4h仍有行情，近30个完整UTC自然日日成交额中位数不少于500万USDT。
- 按真实 `quote_volume` 排流动性，最多120个币，打平时按symbol排序。排除稳定币、杠杆代币和已知非加密标的；保留1000类crypto multiplier。
- 成员在起点冻结。后来上市的币不补入，后来下架或数据停止的币不从历史成员中删除。
- `survival_audit.parquet` 单独检查每个原始成员能否覆盖至采集末端。该结果不参与入池筛选，也不等于官方下架公告。
- 归档日文件通常比实时行情滞后。数据清单分别记录请求结束时间、已收盘截止时间与实际最后bar收盘时间，报告不将尚未采集的10月末段算入测试。

## 因子与标签

因子清单冻结在 `strategies/features/pit_rank64.v1.json`，精确对应前次随机种子42、绝对Spearman大于0.99贪心去重后保留的64个因子。公式复用旧因子库，不修改旧公式。遇到4h行情缺口，分段重新预热至少25根bar，不跨缺口拼接滚动窗口。

每天北京时间12:00，即UTC04:00，使用刚收盘的UTC00:00～04:00这根bar及其过去数据。对每一个原始因子，在当时有完整64维特征的币种截面内计算：

```text
rank_feature = (average_rank(factor_value) - 1) / (cross_section_count - 1)
```

输入只有64个0～1排名，没有symbol、未来收益或全样本标准化参数。并列值使用平均秩，截面至少20个币。排名在每个决策时点独立计算。

目标是**决策后一天**的超额简单收益：

```text
forward_return_i = open_i(t + 24h) / open_i(t) - 1
universe_return = mean(forward_return_j of fixed inception members with complete observations)
excess_forward_return_i = forward_return_i - universe_return
```

`t` 为北京时间12:00决策与模拟执行时间。因子使用上一根已收盘bar，模拟成交用下一根bar开盘价。未来标签通过确切24小时时间戳对齐，同时检查中间6根bar连续。训练阶段未来标签缺失不纳入监督样本，但预测阶段不因标签缺失删除该币。标签是单日收益差，不是累计收益，亦未加入止损或交易费用。

## 单模型与一次固定训练/验证/测试切分

1. 从原2025-01～2026-04的训练期尾部划出2026年4月作为固定验证集。实际拟合集为 `[2025-01-01, 2026-04-01)`，使用整个15个月的可用样本。
2. 固定验证集为 `[2026-04-01, 2026-05-01)`，测试集为 `[2026-05-01, 2026-11-01)`。10月只使用实际已获得的数据。
3. 训练样本清除 `label_end_timestamp >= validation_start` 的记录，验证样本清除 `label_end_timestamp >= test_start` 的记录，保证标签不跨入下一集合。训练并不缩短为365天。
4. 预先固定参数：树数上限 `n_estimators=500`、`num_leaves=15`、`learning_rate=0.03`、`min_child_samples=100`、`reg_lambda=10`、`reg_alpha=0.1`、seed42。
5. 仅调用一次 `fit`，只把4月验证集放进 `eval_set`。验证RMSE连续40轮不改善时停止，保存最佳迭代的一个 `model.txt`。不搜索参数、不滚动、不划fold。
6. 验证后不将4月数据重新并入训练。全部测试预测均由保存的同一个冻结模型产生，测试标签只用于事后评分。

`split.json` 保存实际训练/验证/测试行数和时间边界，`validation_predictions.parquet` 保存验证预测。旧运行的manifest和报告留作历史，新运行不生成滚动验证或参数搜索产物。最新运行指针带有 `training_protocol=fixed_train_valid_test`。

## 仓位与回测

- 只多，每天12:00按模型预测超额收益从高到低取前10%，数量向上取整，同分按symbol排序。
- 等仓，目标总仓位100%，实际现金受手续费/滑点占用影响。不附加正预测阈值。
- 单币固定10%止损，以平均实际入场价计算。持续持有时不把止损每天重置到最新价。新增份额按成本更新平均入场价。
- 4h检查：开盘已跳空跌破止损，则按更差的开盘价卖出；bar内low触及则按止损价卖出，再扣滑点与费用。
- 当天止损后，最早下一次日调仓重新参选。离开前10%时退出，继续入选时按等仓目标调整差额。
- 单边手续费5bp、滑点2bp，资金费率暂按0。回测有完整交易账本和4h权益曲线。
- 若持仓bar缺失，不预知未来下架时间提前平仓，也不将异常静默计为正常零收益。保留上次估值并写入 `data_issues.parquet`，相关运行 `execution_data_complete=false`。
- 基准为同一期固定币池可计算完整24h收益的成员等权收益。每日净收益减基准后相加，形成累计超额单利收益；权益曲线则反映资金滚动变化。最后持仓按市值计价，不假装已全部清仓。

## 研究限制

64因子的历史筛选使用了2026-07的相关性快照，含有本次部分测试区间。用户要求复用此清单，因此该选择历史会写入manifest；不能把新测试结果称为从未参与过特征选择的最终留出证据。

2025起点固定币池避免用2026存续结果反选成员，但不代表每月动态更新币池。原始合约符号迁移也不自动拼接，数据归档是否完整、合约是否正式下架需要分开解释。

## 命令

```bash
uv run alpha-web3-research research prepare-pit-rank
uv run alpha-web3-research research run-pit-rank
uv run alpha-web3-research strategies sync
uv run alpha-web3-research strategies show pit_rank64_daily_excess_long_only
```

配置文件：`strategies/configs/pit_rank64_202501_202610.v1.json`。可用 `--research-config` 提供另一份配置；研究数据目录、成员、时间与模型均在manifest中固定。改变币池或采集范围时应重新准备数据；只修改训练参数可复用同一份已采集数据，创建新的run。

产物位置：

```text
data/research/pit_rank64_202501_202610/
  archive_index.json
  inception_bars/
  universe.parquet
  bars/
  survival_audit.parquet
  data_manifest.json
data/models/pit_rank64_daily_excess_long_only/<run_id>/
  dataset.parquet
  labels.parquet
  split.json
  model.txt
  validation_predictions.parquet
  predictions.parquet
  feature_importance.parquet
  manifest.json
data/backtests/pit_rank64_daily_excess_long_only/<run_id>/
  portfolio.parquet
  daily.parquet
  trades.parquet
  data_issues.parquet
  metrics.json
reports/strategies/pit_rank64_daily_excess_long_only/<run_id>.md
```

官方数据说明：[Binance Public Data](https://github.com/binance/binance-public-data)。
