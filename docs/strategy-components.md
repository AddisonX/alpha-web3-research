# 在线策略组件

本地已经拆分运行组件，东京服务器尚未部署。默认模型、因子、排名规则及Freqtrade输入格式不变。

```text
定时任务 / CLI
    → live/signals.py（时钟、缓存、校验及编排）
    → live/market.py（行情与特征）
    → strategies/<策略>.py（decide，只决定持有什么）
    → live/signal_store.py（转换为现有信号文件）
    → 原安全层 / Freqtrade / 监控API / React
```

## 谁负责什么

| 文件 | 职责 |
|---|---|
| `src/strategies/base.py` | 数据需求、策略上下文、完整0/1目标及协议 |
| `src/strategies/lightgbm.py` | 当前生产模型加载、均值预测、前10%选币 |
| `src/strategies/registry.py` | 显式工厂，只允许已登记策略，不动态执行任意导入 |
| `src/live/market.py` | 按策略声明的特征、周期及预热长度准备矩阵 |
| `src/live/signal_store.py` | 旧Parquet/JSON协议、时间线、历史合并及原子写入 |
| `src/live/signals.py` | 调用以上组件，验证输出后发布；异常标记failed |

模型属于策略的内部实现；规则策略无需模型。策略的`decide(context)`不应联网、写文件、读取交易库或下单。
模型文件可以在策略构造阶段加载，不在数据层或执行层加载。

## 第一版契约

- `DataRequirements`：特征名称/顺序、周期、回看长度、最少已收盘bar数。
- `StrategyContext`：决策bar开始时间，以及按symbol标识的特征矩阵。该bar收盘后才调用策略。
- `StrategyDecision`：相同决策时间，以及`pair/target_position`表。每个输入币必须出现一次；0退出、1持有，不是增量指令。
- `model_score`、`model_score_rank_pct`、`score_position`为可选诊断。规则策略省略分数时，保存层保留兼容列并写null，不伪造预测分数。
- `StrategyMetadata`：来源标识及可选模型数、horizon、标签信息；规则策略模型数为0。

当前执行和安全适配器仅支持4h、只多、二元目标、至少60个币且至少一个入选。
不支持空仓目标、做空或连续权重调仓；空目标会失败关闭，不能暗中转成旧安全层不认可的信号。
这里未新增持仓/账户输入，因为现有决策不使用它们；需要状态型策略时再扩展上下文并增加独立持仓读取接口。
数据准备仅能提供现有FactorBuilder会计算的特征；自定义数据源或因子尚需新增数据提供组件。

## 运行与扩展

现有命令不变，默认目录与当前Freqtrade配置一致：

```bash
uv run --no-sync alpha-web3-research live generate-signals --mainnet
uv run --no-sync alpha-web3-research live generate-signals --mainnet \
  --strategy tokyo-lightgbm-long-only-v1
```

也可在环境配置中设置`ALPHA_LIVE_STRATEGY`。目前注册了上述默认策略及`return-1-reversal-long-only-v1`兼容性验收策略，未知ID会报错；没有新增真实资金交易策略。

新增策略需要：实现协议，在`registry.py`显式登记，添加独立测试；使用统一实例入口时，还需将新ID加入`scripts/run_dryrun.py`的允许列表。无需修改signals.py、保存组件、Freqtrade或前端。
外部交付可使用[策略导入模板](../templates/strategy-import/README.md)，其中包含策略骨架、Codex提示词和离线发布验收。
`src/strategies/reversal.py`实现无模型反转策略，独立运行及已完成验收见[第二策略兼容性验收](reversal-acceptance.md)。不能将兼容性结果作为生产收益证据。

非默认策略的CLI默认将信号和状态写到`freqtrade/instances/<instance>/signals/`，不能覆盖默认生产文件。
每次写状态新增`strategy_id`用于所有权检查；旧状态没有该字段时仅视为默认东京策略。
发现目录属于其他策略时直接拒绝，连失败状态也不覆盖对方文件。

独立执行仍需单独Freqtrade配置、数据库和环境变量，不能仅生成另一路信号就认为已开始交易：

```text
ALPHA_FREQTRADE_LIVE_SIGNAL_PATH=<该策略Parquet>
ALPHA_FREQTRADE_LIVE_STATUS_PATH=<该策略JSON>
ALPHA_LIVE_AUDIT_PATH=<该策略审计JSON，若启用>
```

运行审计时显式传该策略的`--signal-path/--status-path/--database-path/--report-path`。
当前本地dashboard自动发现`freqtrade/instances/`中的规范实例，可切换持仓、订单和独立权益曲线；东京未部署新版多实例接口。
每个实例仍是独立模拟账户，没有共享账户资金分配器，自动监控发现也不会自动注册或启动新策略。

## 兼容和失败行为

模型、特征、排名与时间线算法保持原样。Parquet字段不变，健康状态保留全部旧字段，仅新增strategy_id。
同轮健康信号继续跳过；重复强制生成仍按date/pair去重，保留约45天历史。
新策略异常、重复/遗漏币、错误时间、空目标或不支持的目标值，只写failed状态并抛错；之前的Parquet保持原样，不发布“清仓”。
两份文件各自原子替换，不是跨文件事务；原有安全层继续负责检查它们的一致性。
这不是通用插件平台，也未重新添加模型训练或历史回测功能。
