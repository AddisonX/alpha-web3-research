# 2026-11-11 本机同步及启动条件验收

## 已启动独立dry-run（北京时间12:28）

按用户明确要求，STOPPED联网启动检查通过后，已启动本地`xgb15-paper`，不是真钱交易或东京部署。
检查数据库中交易和订单均为0；检查worker已正常退出，未把检查库混入正式模拟库。
运行配置：主网公共行情、dry_run=true、10,000 USDT模拟资金、逐仓、1倍杠杆、独立交易库，无交易所密钥和RPC交易接口。

北京时间12:28:30至12:28:41，51笔模拟入场订单全部成交：25个多头、26个空头。
12:30独立只读核查：审计healthy、entry_allowed=true、缺失目标0、意外持仓0、未完成订单0。
目标多空各50%；初始实际stake合计约4,998.56 USDT多头、4,941.91 USDT空头，精度、价格与可用保证金使实际分配有偏差。
不能将此启动/成交验收视为跨日再平衡、资金费率或滑点的完整验收。

实例目录和监督状态：

```text
freqtrade/instances/xgb15-paper/config.json
freqtrade/instances/xgb15-paper/runtime/runtime_status.json
freqtrade/instances/xgb15-paper/signals/alpha_live_safety_audit.json
freqtrade/instances/xgb15-paper/database/trades.dryrun.sqlite
```

监控API已加载多空版本，原两个交易进程未重启。前端使用`?instance=xgb15-paper`直接选择新实例。
修正了硬编码LONG ONLY标签；未交付的可选历史回测文件不再在实时页面显示404告警，没有伪造历史回放。

每15分钟检查信号、每分钟审计；实例未设置开机自启，本机休眠或断网会影响运行。
**市值自动刷新仍未启用。** 当前实时副本覆盖至11月10日，按现有滞后规则还能覆盖后续两轮；
若此前不刷新，11月14日所需的11月11日市值将缺失，新信号会失败关闭。需单独补上定时刷新及失败告警，不能认为已具备无人值守持续运行条件。

以下保留启动前的检查记录。

## 后续更新与复检（北京时间11:43）

已按用户要求完成全部301个CMC资产ID请求，实时市值更新到2026-11-10。
模型仍严格使用滞后规则要求的截至11月8日的40日窗口：完整覆盖291/297，离线资产ready=true。
未填补缺失的6个成员为1000CHEEMS、1000RATS、BTCDOM、DOGE、OM、OXT。
发布快照不变，`scripts.verify_xgb15_bundle`再次通过，全部历史复现差异仍为0。

完整297币最新主网行情和真实模型的隔离信号预检已通过：

- 258个币具备完整可预测特征；25多、26空，共51个目标，两侧目标权重分别+50%和-50%。
- 当轮决策bar为11月10日12:00 UTC，信号bar收盘为北京时间11月11日00:00。
- 信号健康检查、行校验及目标可交易性检查通过，入选的非活跃交易对数量为0。
- 297次OHLCV请求无请求异常；并非所有有行情的币都具备完整因子。
- 未启动worker、未下单。信号仅在临时目录校验后清理，没有接入实例的正式信号路径。

当前结论：**市值过期阻塞已解除，数据与信号条件通过；尚未进行STOPPED联网启动或模拟成交验收，也未启用市值定时刷新。**
实时副本及机器可读结果：

```text
data/imports/xgb15/runtime/caps.parquet
data/imports/xgb15/runtime/asset_check.json
data/imports/xgb15/runtime/signal_preflight.json
```

以下保留更新前的验收记录，不能视为最新状态。

## 结论

已将PR #1提交`3d4465fd33eb3d97dbf4f91ad722bbe31edef448`合入本地dev，并保留原资产工具。
**历史复现通过；当前不满足开启dry-run交易的条件。** 未推送远端、未启动新worker、未部署服务器。

## 已通过

- 发布清单引用的14个文件的大小和SHA-256均一致。
- 2022至2026五个年度模型加载通过，有序特征和120轮模型校验通过。
- 2026年度池297币，运行pairs与manifest一致。
- 2026-10-27北京时间00:00的独立复现：260行特征与预测，26多/26空；特征、分数和权重最大差异均为0。
- 年度池296币有代币CMC ID；BTCDOM是指数，原始空ID映射已正确保留，不伪造代币市值。
- 临时目录中的Freqtrade配置schema校验通过：dry_run=true，多空执行类，297币白名单，不启动worker。
- Binance主网时间与BTC 4h K线样本接口HTTP 200。
- CMC BTC历史样本接口HTTP 200、提供方状态SUCCESS，返回2026-11-06至11-10的5条日观察。未做全池更新。
- 后端回归108项通过；前端22项通过；生产构建通过。

复现命令：

```bash
uv run --no-sync python -m scripts.verify_xgb15_bundle
uv run --no-sync python -m scripts.xgb15_assets check
```

## 明确阻塞项：市值时效

发布caps快照截至2026-11-05。
当前决策bar为2026-11-10 12:00 UTC，收盘/信号时刻为11-10 16:00 UTC，即北京时间11-11 00:00。
按现有研究滞后公式，需要截至11月8日的40日市值窗口，目前缺11月6、7、8日。
因此当前完整窗口覆盖为0/297，低于至少60币的运行要求。
“截至快照最后日期291币完整”和“当前日期0币完整”不矛盾，二者窗口结束日期不同。

在临时输出目录中实际调用信号生产链，得到：

```text
market cap source date 2026-11-08 is missing; OHLCV fetch not started
status=failed, signal_written=false, worker_started=false
```

说明系统会在过期市值输入处失败关闭，没有假信号或模拟开仓。

## 接下来需要做

1. 显式运行市值刷新，确认至少60个年度池成员具备当前所需的完整40日历史及稳定CMC身份。
2. check通过后，验证完整币池最新主网OHLCV及当轮信号；目前只测了BTC行情连通，不等于全池已验收。
3. 配置独立xgb15-paper实例，做STOPPED启动检查，核对实际可交易币对、最小订单额、保证金及目标方向。
4. 由用户明确允许后开始dry-run，观察多空订单和至少一次跨日再平衡。调仓/资金费率/滑点还未完成实时成交验收。
5. 持续运行前安排并监控市值定时刷新；本次没有启用定时任务。

刷新命令已保留，但本次验证未批量下载或覆盖数据：

```bash
uv run --no-sync python -m scripts.xgb15_assets refresh-caps
uv run --no-sync python -m scripts.xgb15_assets check
```

为避免实时更新破坏发布快照的摘要，后续更新仅写入Git忽略的`data/imports/xgb15/runtime/caps.parquet`。
行情层和缺件检查优先读取实时副本，历史验收仍读取发布快照。数据源拒绝访问时停止，不绕过限制。
以上结论只针对模拟交易准备，不是实盘交易或收益有效性认证。
