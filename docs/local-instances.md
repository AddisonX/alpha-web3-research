# 本地策略实例

`tokyo-lightgbm`是本地独立模拟盘，使用东京同款固定5模型，不接管东京账户或进程。
它从10,000 USDT模拟初始资金开始；没有复制东京交易历史，因此持仓和收益不保证与东京相同。
当前本地币对列表也不保证与服务器名单一致。

2026-11-10北京时间15:38已启动，85币信号选出9个目标；9笔模拟买单全部成交。
核查时运行状态RUNNING、审计healthy。最新状态以实例文件及进程为准，不把此启动记录当成实时状态。

```text
freqtrade/instances/tokyo-lightgbm/
  instance.json         策略ID和目录布局标识
  config.json           独立dry-run配置
  pairs.json            独立币对列表
  signals/              交易信号、健康状态、安全审计
  database/             本地模拟交易SQLite及WAL文件
  logs/                 supervisor.log、freqtrade.log
  runtime/              supervisor.pid、supervisor.lock、runtime_status.json
  checks/               限时启动检查的配置、数据库和报告
  user_data/            Freqtrade自动生成的数据、回测等框架目录
```

生产模型只读共享`data/models/production/`，执行适配类共享`freqtrade/strategies/`。
两个实例不共享信号、状态或交易数据库。

```bash
# 前台运行同一个本地实例；文件锁拒绝重复启动
uv run --no-sync python scripts/run_dryrun.py

# 仅准备目录，或做STOPPED状态的限时联网检查
uv run --no-sync python scripts/run_dryrun.py --prepare-only
uv run --no-sync python scripts/run_dryrun.py --check-startup

# 停止这个本地实例，不影响反转策略或东京服务器
kill -INT "$(cat freqtrade/instances/tokyo-lightgbm/runtime/supervisor.pid)"
```

后台运行状态以`runtime/runtime_status.json`为准，必须结合PID是否存活与日志判断，旧状态文件不是永久运行保证。
15分钟检查新信号，每分钟审计。只使用主网公共行情，无交易密钥，始终强制dry-run。
进程未设置开机自启，本机休眠或网络中断会影响运行。

反转策略已统一到`freqtrade/instances/return-1-reversal/`，与tokyo-lightgbm使用相同目录层级。
迁移前先停止本地进程并备份SQLite，迁移后核对18个持仓ID、币对和开仓状态均保留，再恢复两个实例。
两个入口共享`src/live/instances.py`中的准备与监督函数，不复制监督逻辑。
反转策略原启动命令`python scripts/run_reversal_dryrun.py`保持兼容，改为委托统一入口并使用进程锁。

`freqtrade/`顶层只保留`strategies/`和`instances/`。
旧的闲置`user_data/`和根配置已转存`data/archive/freqtrade-layout-20261110/`，含数据库备份和迁移核对记录。
SQLite的wal/shm是正常辅助文件，保留在各实例database/中，不能为了美观删除活动文件。
本地部署模板及CLI默认路径已更新，东京服务器没有应用本次路径变更。

## 多实例监控

本地前端：http://127.0.0.1:4173/，只读监控API：http://127.0.0.1:8080/。
页面顶部选择`tokyo-lightgbm`或`return-1-reversal`，切换持仓、订单、信号和权益曲线。
当前前端已指向本机，不是东京服务器；两个交易实例自身不需要开放HTTP端口。

监控API扫描`freqtrade/instances/`，不加载策略模型，也不启动或控制交易进程。
前端每10秒刷新实例列表和所选实例快照；API每60秒独立采集所有实例的权益。
交易数据库以SQLite只读方式打开，监控曲线单独保存到各实例`database/equity_history.sqlite`。
初始资金读取该实例`config.json`的`dry_run_wallet`，不同实例的曲线与前端缓存完全隔离。
运行状态同时检查最近3分钟心跳和监督/交易PID，过期或进程不存在显示unknown，而不是继续显示运行中。
行情不可用时标记降级，不把开仓价替代的估值写入权益历史。

本地权益历史从本次监控启用时开始采集，不伪造此前的逐分钟曲线，也没有导入东京历史。
历史交易仍从各自数据库读取，曲线和最大回撤不代表缺失监控时段的完整历史。

### 后续增加实例

通过统一启动入口生成符合上述目录结构的实例，例如：

```bash
uv run --no-sync python scripts/run_dryrun.py --strategy return-1-reversal-long-only-v1 --instance reversal-paper-2
```

目录需包含`instance.json`（`layout: structured`及`strategy_id`）和`config.json`。
当前监控支持主网公共行情的Binance USDT永续dry-run、数值型初始资金，交易数据库必须位于本实例`database/trades.dryrun.sqlite`。
有效实例会在10秒内出现在选择器，无需修改React页面或重启API。
不符合约定的实例单独报告，不影响其他实例；尚未产生数据库的实例明确显示未就绪。
全新策略仍需实现并注册策略接口，目录自动发现不等于自动支持任意交易所或执行框架。

接口：`GET /api/instances`、`GET /api/instances/{id}/live`、`GET /api/instances/{id}/equity?range=all&interval=auto`。
原`/api/live`和`/api/live/equity`保留为默认实例别名，供旧客户端使用。
通过`?instance=return-1-reversal`可以直接打开指定实例。
