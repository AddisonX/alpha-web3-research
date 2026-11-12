# Alpha Tokyo Runtime

本目录保留三种策略的信号、模拟执行和监控链路，不再提供通用研究及模型训练。

**协作开发以`dev`分支为准。接入外部模型与策略请先阅读[策略导入README](docs/STRATEGY_IMPORT_README.md)。**
`main`保留旧开源研究版本；当前分支使用扁平化`src/`、策略接口和独立实例。
`dev`已包含东京5个LightGBM权重、XGB15年度权重、三个策略的币池及无密钥模拟配置。
也包含XGB15历史复现输入和最近60天市值启动快照；数据时效和交易所连通性仍须检查。
交易数据库、实时持仓、日志、API密钥、看板登录密码和证书私钥不上传。

## 当前策略

| 实例 | 策略 | 币池 | 模型资产 |
|---|---|---|---|
| `tokyo-lightgbm` | 每日横截面只多 | 85币 | 5个LightGBM模型已发布 |
| `return-1-reversal` | 每日反转只多 | 85币 | 规则策略，不需要模型 |
| `xgb15-paper` | 每日多空及缓冲调仓 | 297币年度池 | 2022～2026年度XGBoost模型已发布 |

下面是默认东京LightGBM策略的参数，不代表另外两个策略的规则：

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
回归测试包含合成模型及已发布真实权重的加载、哈希和配置校验，不将加载成功等同于收益验证。

## 运行

新机器先初始化已发布的三个模拟实例。该命令只做离线哈希校验、生成本机路径配置和非覆盖式市值快照安装，不启动交易。

```bash
uv run --no-sync python -m scripts.prepare_paper
uv run --no-sync python -m scripts.verify_xgb15_bundle
uv run --no-sync python -m scripts.xgb15_assets check

# XGB15持续运行前更新市值并检查时效；不修改历史复现快照
uv run --no-sync python -m scripts.xgb15_assets refresh-caps
uv run --no-sync python -m scripts.xgb15_assets check
```

完整说明及每个实例的启动命令见[模拟盘资产与配置](docs/published-paper.md)。
初始化不会覆盖已有模拟账户或已有实时市值，也不提供历史训练OHLCV；在线运行会获取所需的近期主网行情。
新策略接入仍按[导入README](docs/STRATEGY_IMPORT_README.md)操作。

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
东京新版多实例看板已独立部署；本地前端默认仍指向本机，不接管服务器账户。
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
| `configs/paper/` | 三个可移植模拟配置、币池、市值种子及资产哈希清单 |
| `freqtrade/` | 唯一运行策略、配置、币对、信号、数据库及日志 |
| `dashboard/` | 实时持仓、订单及可缩放历史权益前端 |
| `deploy/` | Nginx和systemd部署模板 |
| `scripts/` | 实例启动入口、离线兼容重放、安全故障演练 |
| `tests/` | 运行链路回归测试 |
| `templates/strategy-import/` | 策略交付骨架、Codex提示词和离线验收测试 |
| `docs/` | 精简记录 |

## 精简边界

其他策略、训练/标签/评估模块、研究行情、分层报告、历史回测页面及测试网真实交易辅助模块均已移除。
此分支发布精简代码和明确列入清单的策略运行资产，不包含私人备份、删除日志、研究训练数据或账户运行状态。
部署目录是示例，需自行填写域名及安装路径；提交代码不会更新或重启服务器。

## 策略解耦

`live/signals.py`仅编排；`live/market.py`准备数据，`strategies/lightgbm.py`预测选币，
`live/signal_store.py`保存兼容的Freqtrade信号。默认运行方式和模型不变。
策略通过`decide(context)`返回完整的只多0/1目标，不联网、不下单。
新增策略通过显式注册及独立信号目录接入，目前另注册了`return-1-reversal-long-only-v1`和独立多空策略`xgb15-120-long-short-v1`。
详细契约、支持边界和隔离运行方法见[策略组件说明](docs/strategy-components.md)。

让其他Codex或协作者生成兼容策略时，使用[策略导入模板](templates/strategy-import/README.md)，
并发送其中的[Codex提示词](templates/strategy-import/CODEX_PROMPT.md)。模板未注册，不会启动交易。

已接入的`xgb15-120-long-short-v1`使用独立多空执行器，真实资产准备及缺件检查见[XGB15本机接入](docs/strategies/xgb15-120-long-short-v1/LOCAL_SETUP.md)。

第二策略的历史重放、独立dry-run启动方式和当前网络阻塞，见[兼容性验收](docs/reversal-acceptance.md)。

本地东京同款模型实例位于`freqtrade/instances/tokyo-lightgbm/`，独立目录及运行方式见[本地实例说明](docs/local-instances.md)。
