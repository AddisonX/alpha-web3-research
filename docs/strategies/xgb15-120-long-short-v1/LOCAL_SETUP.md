# XGB15：research到dev的本机接入

## 已完成什么

`dev`已登记`xgb15-120-long-short-v1`，无需重复注册。
策略实现、15因子计算、多空签名目标、安全校验、执行适配器和前端支持来自PR #1。
本次补充研究资产导出/导入、离线缺件检查和显式市值刷新命令，默认实例名为`xgb15-paper`。
没有将research的全部训练实验合并到dev，没有重新训练、启动新worker或部署服务器。

更新：已同步PR #1提交`3d4465f`，该提交带有指定年度权重及复现输入。
现在无需再找对方导出这些已有文件，也不要对已有目录重复执行import。
先运行`python -m scripts.verify_xgb15_bundle`复现历史，再执行check检查当前时效。
本机实际验收及实时阻塞项见[2026-11-11验收报告](READINESS_20261111.md)。

| research来源（6f7b0c7） | dev中的对应部分 |
|---|---|
| `analysis/gpu_xgb15_expanding_120_lr005.py` | 原120树年度训练产物的导出适配，不在dev重新训练 |
| `analysis/buffer30/15_gpu_hysteresis_backtest.py` | 已有`src/strategies/xgb15.py`的30%保留缓冲决策 |
| `analysis/collect_market_cap_2026.py`、`collect_market_cap_supply.py` | `src/data/xgb15_caps.py`：显式映射、日观察、身份校验和原时间滞后规则 |
| 研究私有模型/预测表/市值产物 | `src/strategies/xgb15_assets.py`：只读导出、校验、拒绝覆盖的导入 |

不能直接复制research的采集脚本运行：它们含研究机路径、研究币池依赖和顶层执行逻辑。
移植后只在显式调用命令时联网，没有自动下载、自动训练或自动调仓副作用。

## 1. 检查当前缺什么

项目根目录运行：

```bash
uv run --no-sync python -m scripts.xgb15_assets check
```

返回JSON，并以退出码2表示缺件或未通过。检查真实模型的SHA-256、120轮、特征顺序、年度池及训练截止时间，
还会检查市值40天完整性、CMC身份稳定性和至少60币覆盖。它不联网，不创建实例。
`ready=true`只表示离线资产条件通过，不代表已验证最新OHLCV、预测复现、成交或实盘安全。
`cap_refresh_mapping_ready`单独表示是否具有刷新所需的显式映射；未配置自动定时刷新。

2026-11-11本机初次检查：缺运行manifest、真实模型、年度pairs、市值历史和CMC映射。
本轮决策bar为2026-11-10 12:00 UTC，按现有公式使用截至11月8日的市值，最长窗口为40天。
后续以命令实时计算的日期为准，不能把这段初次记录视为永久运行状态。
初次缺件记录已被本次PR同步解决；当前阻塞项为市值快照过期，而不是模型缺失。

## 2. 从实际research产物导出

仅克隆research代码不能完成导出，必须提供对方真实研究目录，其相关产物结构为：

```text
reports/xgb15_gpu_expanding_120_depth4_lr005_2022_2026/
  manifest.json
  2026/model.ubj
data/model_tables/factors15_2021_2026/predict_table_2022-2026/predict_table_2026.parquet
data/fundamentals/cmc_2021_2025/asset_mapping.json
data/fundamentals/cmc_2021_2025/contract_daily.parquet
data/fundamentals/cmc_2026_supplement/contract_market_cap_2026.parquet
```

市值产物可以只提供实际所需区间，但列名和语义必须与research一致。
目前导出器按上述原私有产物结构读取，GPU训练脚本若将产物保存在/tmp，需先由对方整理到对应目录。

```bash
uv run --no-sync python -m scripts.xgb15_assets export \
  --research-root /path/to/actual-private-research \
  --output data/exports/xgb15-2026 --year 2026
```

这个命令不训练、不联网、不改输入文件、不启动交易。输出可移交的目录：

```text
manifest.json          私有运行清单，含models.2026.pairs等实际年度池信息
model_2026.ubj         真实120轮模型，校验实际特征名、轮数及SHA-256
pairs.json            年度完整币池
caps.parquet          统一的五列日市值观察，可另保留原历史
asset_mapping.json    有原映射时一并交付；持续刷新需要它
```

导出器读取实际120树文件，不会自动截取任意200树模型冒充原模型。
原120树文件缺失时，让对方先用原恢复/验收流程提供经过核对的模型；旧`import_xgb15_research.py`仍保留原历史恢复流程。
不要把docs里的公开manifest直接替代运行manifest，因为公开版没有完整年度池。
包里的市值可能仍过期；导出/导入成功不等于市值时效验收通过。
权重、池和数据通过双方约定的私有渠道传递，不推送Git。

## 3. 导入到本机dev

```bash
uv run --no-sync python -m scripts.xgb15_assets import \
  --source /path/to/xgb15-2026 --year 2026
uv run --no-sync python -m scripts.xgb15_assets check
```

导入目录：

```text
data/models/strategies/xgb15-120-long-short-v1/v1/
  manifest.json
  model_2026.ubj
data/imports/xgb15/
  pairs.json
  caps.parquet
  asset_mapping.json
```

已有资产目录时拒绝覆盖，避免替换运行中的模型或市值。更新已安装资产前需要明确的离线迁移/备份步骤。
仅2026模型即可做2026推理；历史年度回放需要相应年度模型，跨到2027也不会自动获得新年度模型。

## 4. 更新市值

```bash
uv run --no-sync python -m scripts.xgb15_assets refresh-caps
uv run --no-sync python -m scripts.xgb15_assets check
```

需要原research的`asset_mapping.json`，记录格式为`[{"contract":"BTC/USDT:USDT","cmc_ids":[1]}, ...]`。
映射必须覆盖完整年度池；版本顺序沿用研究定义，不用ticker猜测CMC ID，也不把1000倍合约的市值再乘1000。
对明确标记为`instrument_type: index`的BTCDOM等指数允许空ID列表，保留其市值缺失；普通代币不允许空ID。
采集沿用research里的CMC公开历史响应接口，只读取已完成日，不用现时市值倒填历史。
接口访问、服务条款和响应稳定性需在实际运行环境确认；本次只做了模拟响应测试，没有完成真实CMC联网验收。
遇到401/403等拒绝访问直接失败，不换入口绕过；任意请求失败时保留原caps.parquet。
保存原始响应、获取时间及摘要；已有历史观察不被后续版本静默覆盖，不插值或前后填充。
发布的`data/imports/xgb15/caps.parquet`保持不变，以保留历史复现摘要。
刷新结果写到`data/imports/xgb15/runtime/caps.parquet`；实时行情与check优先读取该文件，历史验收仍使用发布快照。
缺失身份误差值沿用原研究处理，不伪造为0；它不代表已完成价格×供给的一致性验证。
如需修正已有观察或更换数据源，先做单独审核，不能直接覆盖后声称与旧研究严格一致。

这只是显式刷新入口，**尚未启用定时任务或监督器自动刷新**。
持续dry-run前需在每日决策前安排刷新，并检查覆盖；没有新数据就应失败关闭，不沿用过期输入。

## 5. 最后再创建实例和联网验收

拿到同一历史时点的15因子、上一轮目标和预期分数/权重，先独立复现，尤其检查30%缓冲、年度池及排名分母。
本次自动测试用临时训练的合成模型，不是对方私有生产模型的收益或预测复现证据。

离线资产通过后，准备独立实例：

```bash
uv run --no-sync python scripts/run_dryrun.py \
  --strategy xgb15-120-long-short-v1 --instance xgb15-paper \
  --pairs-path data/imports/xgb15/pairs.json --prepare-only
uv run --no-sync python scripts/run_dryrun.py \
  --strategy xgb15-120-long-short-v1 --instance xgb15-paper --check-startup
```

此时还需检验完整年度池的最新主网OHLCV、至少60个完整特征截面、真实模型推理、
多空订单/保证金、调仓和费用，再由用户明确启动dry-run。这里不自动执行启动步骤。
旧策略仍用各自原目录；新策略50/50多空、总名义100%、每日等权、框架-99%紧急止损，
不是旧的9仓只多/10%止损，也不是无强平风险或已经实现资金费率/滑点一致性的保证。
