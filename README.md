# Alpha Tokyo Runtime

本目录仅保留东京服务器现有策略的运行链路，不再提供通用研究及模型训练。

**协作开发以`dev`分支为准。接入外部模型与策略请先阅读[策略导入README](docs/STRATEGY_IMPORT_README.md)。**
`main`保留旧开源研究版本；当前分支使用扁平化`src/`、策略接口和独立实例。
仓库不包含生产模型、运行币池、交易数据库或密钥，克隆后不能直接运行原作者的东京模型。

## 当前策略

- Binance USD-M主网公共行情，Freqtrade dry-run，无真实下单。
- 4h因子、30个输入特征、5个LightGBM模型取平均，预测未来6根bar收益。
- 每日使用00:00 UTC开始、04:00 UTC收盘的bar，即北京时间12点后生成信号。
- 截面前10%只多，最多9个持仓、1倍杠杆、10%止损。
- 固定模型：`data/models/production/walk_forward_4h_h6_forward_return_20260704T123039Z/`。
- 不改变已有预测、选币、持仓对账和安全检查规则。

## 安装

需要Python 3.12、uv、Node.js 22+；macOS运行LightGBM需要libomp。

```bash
uv sync --frozen --group freqtrade
uv run --no-sync alpha-web3-research live --help
uv run --no-sync pytest -q
```

公共行情不需要API密钥。不要将密钥写入代码或上传GitHub。
公共回归测试使用临时合成模型，不依赖私有权重；真实生产模型测试在没有权重时跳过。

## 运行

以下默认模型命令仅适用于已经通过私有渠道准备好生产权重和币池的本机。
新协作者请按[导入README](docs/STRATEGY_IMPORT_README.md)创建自己的策略和实例。

```bash
# 联网生成信号，默认主网公共行情
uv run --no-sync alpha-web3-research live generate-signals --mainnet
uv run --no-sync alpha-web3-research live audit-dry-run

# 统一启动模拟交易、信号检查和安全审计；文件锁拒绝重复启动
uv run --no-sync python scripts/run_dryrun.py

# 只读监控API
uv run --no-sync uvicorn src.live.dashboard:app --host 127.0.0.1 --port 8080
```

Freqtrade使用实例的`pairs.json`。本地模拟盘与东京独立，不共享交易历史。

```bash
cd dashboard
npm ci
npm run dev -- --port 4173
```

前端：http://127.0.0.1:4173，默认代理本机8080接口。
顶部选择本地实例，自动发现`freqtrade/instances/`中的规范实例，持仓、订单、权益历史互相隔离。
API后台采集所有实例权益，不依赖页面当前选择。扩展约定见[本地实例](docs/local-instances.md)。
东京尚未部署新版多实例API，不能仅将新前端代理切回旧东京接口。
不要公开暴露无保护API、交易数据库或密钥。

## 目录

| 目录 | 用途 |
|---|---|
| `src/data/` | 交易所连接、公共行情获取和时间处理 |
| `src/factors/` | 现有在线因子计算，公式不变 |
| `src/universe/rules.py` | 行情模块使用的基础资产规则 |
| `src/live/` | 推理、信号、监控、配置、安全与对账 |
| `src/strategies/` | 独立决策接口、默认LightGBM策略及显式注册表 |
| `data/models/production/` | 当前5模型、manifest与指针 |
| `freqtrade/` | 唯一运行策略、配置、币对、信号、数据库及日志 |
| `dashboard/` | 实时持仓、订单及可缩放历史权益前端 |
| `deploy/` | Nginx和systemd部署模板 |
| `scripts/` | 实例启动入口、离线兼容重放、安全故障演练 |
| `tests/` | 运行链路回归测试 |
| `templates/strategy-import/` | 策略交付骨架、Codex提示词和离线验收测试 |
| `docs/` | 精简记录 |

## 精简边界

其他策略、训练/标签/评估模块、研究行情、分层报告、历史回测页面及测试网真实交易辅助模块均已移除。
此分支只发布精简后的代码；不包含本地私人备份、删除日志、研究数据或运行资产。
部署目录是示例，需自行填写域名及安装路径；提交代码不会更新或重启服务器。

## 策略解耦

`live/signals.py`仅编排；`live/market.py`准备数据，`strategies/lightgbm.py`预测选币，
`live/signal_store.py`保存兼容的Freqtrade信号。默认运行方式和模型不变。
策略通过`decide(context)`返回完整的只多0/1目标，不联网、不下单。
新增策略通过显式注册及独立信号目录接入，目前另注册了兼容性验收策略`return-1-reversal-long-only-v1`，东京部署未变。
详细契约、支持边界和隔离运行方法见[策略组件说明](docs/strategy-components.md)。

让其他Codex或协作者生成兼容策略时，使用[策略导入模板](templates/strategy-import/README.md)，
并发送其中的[Codex提示词](templates/strategy-import/CODEX_PROMPT.md)。模板未注册，不会启动交易。

第二策略的历史重放、独立dry-run启动方式和当前网络阻塞，见[兼容性验收](docs/reversal-acceptance.md)。

本地东京同款模型实例位于`freqtrade/instances/tokyo-lightgbm/`，独立目录及运行方式见[本地实例说明](docs/local-instances.md)。
