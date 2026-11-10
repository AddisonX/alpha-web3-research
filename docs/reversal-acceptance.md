# 第二策略兼容性验收

日期：2026-11-10。目的为验证解耦，不是验证收益。

## 新增策略

- ID：`return-1-reversal-long-only-v1`。
- 按过去一根已收盘4h bar的close/前一根close−1升序，选最低ceil(有效币数×10%)只多。
- 并列按symbol升序；所有币都上涨时仍选涨幅最小组，不要求必须为负收益。
- 非有限因子不入选；全部无效时失败冻结，不发布清仓目标。
- 每日北京时间12点后决策，复用既有执行：最多9仓、1倍杠杆、10%止损。
- 无模型、不加载LightGBM；旧东京策略仍为默认。
- 只新增策略及注册项，核心signals/market/signal_store与Freqtrade执行类没有修改。

## 离线验证

按本地当前85个币对下载官方2026年10月4h归档，85个归档全部获得并通过官方SHA256校验。
这里使用当前币对名单仅作兼容性演示，不能作为历史点时币池或无幸存者偏差的收益研究。

2026-10-08、09、10连续3个决策日：每天85个有效币、9个入选，全部通过原健康状态和信号时间线校验。
使用回放时钟及当时已收盘bar，未来K线不输入决策。

```bash
# 使用已保存快照，完全离线运行
uv run --no-sync python scripts/replay_reversal.py
# 如需重新下载，访问官方历史归档，不绕过受限主网接口
uv run --no-sync python scripts/replay_reversal.py --download
```

产物在`data/acceptance/reversal/`：`bars.parquet`及`.manifest.json`、`replay/replay_report.json`。
回放信号已过期且位于验收目录，没有接入实时交易进程。

自动测试覆盖并列、输入乱序、非有限值、不导入模型、修改未来行情不改变当时结果。
使用实际Freqtrade策略和临时SQLite验证入场、目标退出、10%止损、同轮禁止重入，以及关闭数据库并清空缓存后的恢复。
这属于离线回调/持久化集成测试，不能代替联网worker成交验证。

## 独立dry-run实例

实例目录：`freqtrade/instances/return-1-reversal/`。
独立pairs、配置、signals、审计、SQLite和日志，不覆盖旧策略文件。
仅主网公共行情、无交易密钥，命令强制`--dry-run`，禁用API及Telegram。
启动脚本负责首次生成信号、15分钟刷新、每分钟审计和退出清理。

```bash
uv run --no-sync python scripts/run_reversal_dryrun.py --prepare-only
# 限时启动检查，初始状态STOPPED，不启用交易；结束自动停止检查进程
uv run --no-sync python scripts/run_reversal_dryrun.py --check-startup
# 在符合交易所访问规则且网络可用的环境中运行独立模拟盘
uv run --no-sync python scripts/run_reversal_dryrun.py
```

首次尝试时，本机HTTP探针返回451，Freqtrade加载exchangeInfo超时；该次检查已停止。
历史证据在`logs/startup-check.log`及`startup_report.json`，不能将它们当作最新运行状态。

## 联网复测：已启动

2026-11-10北京时间15:29，独立实例已进入RUNNING，9笔模拟买单全部成交，数据库中有9个只多持仓。
使用主网公共行情、无交易密钥、Freqtrade dry-run；并非向交易所发送真实订单。

本次主网HTTP接口恢复200；CCXT原先未读取用户已配置的网络环境。
新增默认关闭的`ALPHA_CCXT_TRUST_ENV`开关，独立实例显式启用同步/异步环境支持。
只使用本机已有的环境设置，没有新增代理地址或更改东京网络配置。
修复了禁用Telegram/API时不完整的Freqtrade配置，并添加配置schema测试。

信号覆盖85个币并选出9个；执行框架自动剔除非活跃EPIC，实际监控84个交易对。本轮9个目标全部可执行。
真实成交价格、费用、订单及持仓见独立SQLite；此处的“成交”始终是Freqtrade模拟成交，不代表实盘成交质量。
审计每分钟运行，信号每15分钟检查，策略仍然每日调仓。

当前后台监督进程PID见`runtime/supervisor.pid`；worker PID、审计和运行状态见`runtime/runtime_status.json`。
日志：`logs/supervisor.log`及`logs/freqtrade.log`。没有部署为开机自启服务，本机休眠、关机或网络中断会影响持续运行。

```bash
# 停止新实例，监督进程会清理自己的Freqtrade子进程
kill -INT "$(cat freqtrade/instances/return-1-reversal/runtime/supervisor.pid)"
```

东京原策略和服务未修改；当前dashboard仍读取东京旧策略，并未自动切换到这个本地实例。

原安全层仍要求至少60币、至少1个入选，不支持主动全部空仓。
不支持多策略共享账户，dashboard也未新增多实例视图。
