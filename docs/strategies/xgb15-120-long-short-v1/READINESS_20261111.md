# 2026-11-11 本机同步及启动条件验收

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
