# 数据接入规范 v1

状态：第一阶段已实现历史 OHLCV 接口、Binance USD-M 适配器、按市场查询与兼容验证。
Funding/OI、原始归档、持久化合约注册表、版本化研究快照和实时恢复属于后续阶段，本文明确其约束，不代表已上线。

## 目标与边界

数据供应商负责请求与解析，研究代码读取统一数据。历史与实时最终复用聚合和因子逻辑；交易执行独立于行情采集。
第一阶段保留 CCXT、Polars、Parquet、DuckDB 和现有文件格式，不迁移生产数据，不启动下载或交易。

## 身份规范

| 字段 | 定义 | 当前约定 |
| --- | --- | --- |
| venue | 数据发生的交易市场，不是供应商 | v1 保留既有 `binanceusdm`；与磁盘 `exchange` 列对应 |
| provider_id | 数据获取来源 | `binance_official_ccxt` |
| market_type | 合约类型 | 当前适配器仅 `swap` |
| symbol | 项目内部统一交易对符号 | 如 `BTC/USDT:USDT` |
| native_symbol | 上游原始交易对 ID | 如 `BTCUSDT` |
| instrument_id | 含市场、类型、符号的完整身份 | `v1:` + 三段分别 URL 编码后用 `\|` 连接 |

InstrumentKey 的 ID 不包含 provider，所以同一市场切换供应商不会改变身份；不同 venue 的同名 symbol 不会碰撞。
Instrument 保存 base、quote、settle、contract_size、active。list_instruments 返回元数据，不做研究 universe 筛选，并保留上游返回的非活跃合约。
当前是确定性身份映射，不是历史注册表；不声称当前 load_markets 能发现所有退市市场。
后续增加环境/链/DEX 命名空间、合约更名别名和有效期注册表，必须显式迁移身份版本。主网和测试网不得写入同一数据根目录。

## 数据契约

所有时间使用 UTC。BarRequest 和 BarQuery 的整数时间严格为 epoch 毫秒，无秒/毫秒自动猜测；旧 CLI 的字符串日期解析保持原样。
请求按 K 线开始时间筛选，区间为 `[start_ms, end_ms)`；end=None 表示读取可得尾部，不代表无限历史。

### OHLCV v1（已实现）

BarBatch 包含 request、provider_id、schema_version=1、frame。
frame 保留原字段：exchange、market_type、symbol、timeframe、timestamp、open、high、low、close、volume、fetched_at、source。

- timestamp 是开盘时间；小时决策时间仍为 timestamp + 1h。
- 默认排除尚未收盘的 K 线；include_unclosed 是显式调试选项，不用于训练发布。
- 本适配器 volume 按既有 Binance USD-M CCXT 基础币量语义使用。未来合约量来源必须结合 contract_size 转换，禁止假定所有供应商一致。
- 价格为成交价 K 线，不能混入 mark/oracle K 线。
- close × volume 是估算成交额。未来真实成交额另设字段，并在统一升级 schema 后使用。
- 原始数据层将保留原始精度；v1 数值保持旧 Float64 路径，以兼容现有研究。
- 主键是 exchange、market_type、symbol、timeframe、timestamp。
- fetched_at 是本次采集时间，不是历史可用时间。source 保留 `ccxt.fetch_ohlcv` 以兼容旧产物；供应商明确标识在 BarBatch 中。
- 当前存储仍是单源标准表，不会持久化 BarBatch envelope。严禁把多供应商同主键数据直接写入旧 upsert：其 keep-last 不能决定来源权威性。第二个历史供应商落库前必须实现来源分区和发布规则。

### Funding（后续实现）

至少保存 instrument_id、event_time、settlement_time、rate、interval_seconds、rate_kind（estimated/settled）、provider_id。
rate 为小数费率（0.01 = 1%），不得把年化率、8 小时率、1 小时率混存。缺失不等于零。若有用户资金费用，其金额/币种另建账户事件表，不混入市场费率表。

### OI / asset_context（后续实现）

保存 instrument_id、event_time、open_interest、oi_unit、mark_price、oracle_price、premium、provider_id。
原始单位明确标记基础币/合约/名义价值；派生名义 OI 保存换算价格和时刻。各字段非原子快照时记录各自时间，不捏造成同步快照。

### 成交、盘口（后续实现）

交易级成交和账户级 fills 分开；主键必须依据源语义，不能仅靠时间戳去重。盘口声明深度、快照/增量、序列号、采样频率及重建规则。供应商更改采样/深度时分版本，不能静默拼接。

## 接口与配置（已实现）

- HistoricalBarProvider：capabilities()、list_instruments()、fetch_bars(BarRequest)。
- capabilities 声明支持的数据集、周期和实时能力；未知历史起点使用 None，不保证请求范围完整。
- BinanceProvider 包装既有 OHLCVFetcher，通过继承复用列表/排名方法，fetch_bars 调用旧分页逻辑。错误市场、未知合约、不支持的周期在请求行情前拒绝。
- create_ohlcv_fetcher 是旧 CLI 兼容工厂。ALPHA_DATA_PROVIDER=auto 默认选择 Binance USD-M 适配器；其他交易所暂保留原 CCXT 路径，该路径不代表已通过新规范验收。
- 可显式配置 ALPHA_DATA_PROVIDER=binance_official_ccxt；与非 binanceusdm/swap 配置组合时报错。未知 provider 名称直接报错。
- MarketDataReader.read_bars(BarQuery) 是统一研究读取边界。必须指定 venue、market_type、timeframe；symbols=None 为该市场全部合约，空 tuple 为不选任何合约。
- 存储先选择匹配分区，再下推时间/符号过滤，同时核对行内身份。旧 read_ohlcv() 保留用于全库目录和质量报告。
- 小时因子与标签使用 MarketDataReader，按 Settings 指定市场。滚动算法、标签公式、原有输出路径保留。
- fetch-universe-ohlcv 仅刷新当前配置市场中的 included 合约。

运行旧命令无需新增参数。显式选择适配器可使用：

```bash
ALPHA_DATA_PROVIDER=binance_official_ccxt uv run alpha-web3-research data list-symbols --limit 5
```

该命令会访问网络。离线验证使用文末 pytest。

## 存储与版本演进

第一阶段继续现有 exchange/market_type/timeframe/symbol/data.parquet 布局。
不把全量历史拆分迁移混入接口试点。逐笔及高频快照接入时按 dataset/provider/venue/instrument/date 分区；路径使用无碰撞身份编码，替换旧 symbol 字符替换方案。

后续采用三层：

1. 原始层：原始响应/文件、请求、校验和、下载与解析器版本；可重放，不原地覆盖。
2. 标准层：按供应商保留标准化数据，记录修订。原子提交、单写者或写入锁，成功提交后再推进 checkpoint。
3. 研究层：显式来源优先级、口径一致性与质量检查后发布不可变 snapshot；latest 仅是入口指针。

多源冲突：记录差异，按 dataset/instrument/time-window 的发布规则选用；不平均价格、不随到随覆盖。补源必须记录边界，验证重叠段，不用别的交易场所行情替代缺口。
实验固定 dataset_snapshot_id、universe_snapshot_id、feature_version、配置与代码版本/源码哈希。

当前 latest_universe、因子和标签输出仍共享默认路径。因此多市场独立运行时必须配置不同 ALPHA_DATA_PATH、ALPHA_REPORTS_PATH；尚不支持混合市场组合回测，回测仓位 symbol 索引迁移另行实施。

## 时间可知性与采集质量

未来记录 event_time、received_at、ingested_at；available_at 只在有证据或明确延迟假设时设置，注明来源。
历史今天入库不能被当成历史今天才可知；也不能把结算后发布的数据放进结算前决策。历史接收时间缺失应声明假设。
保留实盘决策所用版本与事后校正版，分别用于回放和研究。

采集任务保存真实覆盖区间、缺口、重复数、异常数、请求/文件标识和最后连续完整位置。HTTP 200、非空响应、最大时间戳都不是完整性证明。
K 线连续性结合交易日历、上市/退市及零成交语义检查；成交和快照使用各自规则。缺失、零成交、暂停不得混同。
历史回填与实时采集共享限流预算并优先实时，使用重叠区间幂等补拉；异常重试有界，未完成区间可恢复。以上调度能力尚未实现。

## 新数据源准入

1. 登记业务用途、范围、额度与本地保存权限。
2. 声明能力、定义市场映射，保存小样本。
3. 通过字段、单位、时间、分页边界、去重与异常响应验证。
4. 检查目标市场/日期真实覆盖，与独立数据源交叉核对。
5. 验证重复运行幂等；实现 checkpoint 后补充崩溃恢复验证。
6. 验证统一输入下批处理/回放因子一致；实时上线前验证同段录制回放。
7. 生成质量报告、发布研究快照，再扩大下载范围。

第一阶段验收：

```bash
uv run pytest tests/test_data_providers.py -q
uv run pytest -q
```

覆盖分页与旧入口一致、未收盘过滤、左闭右开区间、拒绝错误市场/周期、重复写入幂等、同名跨市场隔离，以及相同数据经旧读取路径和新入口生成一致因子/标签。
测试不访问交易所、不使用真实凭证、不修改项目研究数据。

## 本次验收记录（2026-10-22）

- 新增测试文件：9 项通过；全项目：64 项通过。
- 旧 `data fetch-ohlcv --help` 正常，参数保持兼容。
- 读取本地 BTC/USDT:USDT 最近 600 根已有 1h K 线，复制到临时目录，分别经过旧全库读取路径和新 MarketDataReader。
- 因子输出 597 行、171 列；标签输出 600 行、15 列。排除本次生成时间 generated_at 后逐列完全一致。
- 原数据、已有模型和研究产物未改写。未访问交易所接口，未下载付费数据，未启动实盘。
- 尚未验收网络异常、实时重连、崩溃恢复、供应商历史覆盖和多源发布；这些属于后续实现范围。
