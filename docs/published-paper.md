# 已发布的模拟盘资产与配置

`dev`的运行资产已按项目所有者要求发布。不是自动实盘交易，也没有复制原账户历史。

## 资产位置

| 路径 | 内容 |
|---|---|
| `data/models/production/latest.json`及指向目录 | 东京5个真实LightGBM权重、30个特征顺序、参数和SHA-256 |
| `data/models/strategies/xgb15-120-long-short-v1/v1/` | 2022～2026年度真实XGBoost权重及年度池manifest |
| `data/imports/xgb15/` | 原始市值、映射、297币pairs和独立预测复现输入，不改原快照 |
| `configs/paper/<实例>/config.json` | 无密钥Freqtrade配置，数据库路径为相对路径 |
| `configs/paper/<实例>/pairs.json` | 85/85/297币池；这是固定运行池，不是动态历史universe算法 |
| `configs/paper/<实例>/instance.json` | 策略ID和结构化实例标识 |
| `configs/paper/xgb15-paper/caps_seed.parquet` | 2026-09-12至2026-11-10的60天已观察市值，共17,760行 |
| `configs/paper/bundle_manifest.json` | 初始化所需32个资产的字节数和SHA-256清单 |

策略权重和配置的发布不构成收益承诺。第三方行情/市值仍受原供应商条款约束，见`THIRD_PARTY_NOTICES.md`。
市值快照不是永久有效数据，不补填缺失币种；更晚日期必须刷新。

## 新机器初始化

需要Python 3.12、uv和Node.js 22+；macOS的LightGBM另需libomp，网络需允许主网公共行情访问。

```bash
git clone --branch dev https://github.com/AddisonX/alpha-web3-research.git
cd alpha-web3-research
uv sync --frozen --group freqtrade
uv run --no-sync python -m scripts.prepare_paper
```

可用`--instance tokyo-lightgbm`等参数只初始化单个实例。
初始化先验证资产清单，将配置数据库路径改为当前机器的绝对路径，写入各自`freqtrade/instances/<实例>/`。
未存在实时市值时原子安装种子到`data/imports/xgb15/runtime/caps.parquet`，绝不覆盖已有观察。
已有实例只沿用既有准备流程，不用发布配置替换其资金或经济参数。
初始化不启动worker，不创建成交或交易数据库，不获取行情，不开启任何RPC交易接口。

```bash
# 历史复现与当前数据条件分开检查
uv run --no-sync python -m scripts.verify_xgb15_bundle
uv run --no-sync python -m scripts.xgb15_assets refresh-caps
uv run --no-sync python -m scripts.xgb15_assets check

# STOPPED联网检查，不允许模拟下单；需要约45秒
uv run --no-sync python scripts/run_dryrun.py --instance tokyo-lightgbm --check-startup
```

数据不足、地区限制、行情抓取失败都可能阻止启动。不要绕过失败检查强行交易。
XGB15有相应年度权重才可跨年，发布2026模型不代表2027年自动可用。

## 启动与看板

每个交易命令分别在独立终端运行；不要为同一实例重复启动监督进程。

```bash
uv run --no-sync python scripts/run_dryrun.py --strategy tokyo-lightgbm-long-only-v1 --instance tokyo-lightgbm
uv run --no-sync python scripts/run_dryrun.py --strategy return-1-reversal-long-only-v1 --instance return-1-reversal
uv run --no-sync python scripts/run_dryrun.py --strategy xgb15-120-long-short-v1 --instance xgb15-paper
```

另启动API与前端：

```bash
uv run --no-sync uvicorn src.live.dashboard:app --host 127.0.0.1 --port 8080
# 另一个终端
cd dashboard
npm ci
npm run dev -- --host 127.0.0.1 --port 4173
```

打开`http://127.0.0.1:4173/`。每个新实例从独立10,000 USDT模拟资金开始，不复用东京或本机其他实例的账户。
公开的是启动配置，不是运行中数据库、实时信号、当前持仓、日志、PID、密码或私钥。
本地默认不自启、不定时刷新市值；持续运行要安排每日刷新并检查失败，Linux部署模板见`deploy/MULTI_INSTANCE.md`。
对外部署必须使用`deploy/DASHBOARD_ACCESS.md`中的认证和限流，不要公开无保护的API。
