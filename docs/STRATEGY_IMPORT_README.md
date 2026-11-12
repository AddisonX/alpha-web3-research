# 将你的模型和策略接入Alpha（dev分支）

新增显式多空扩展见[`xgb15-120-long-short-v1`](strategies/xgb15-120-long-short-v1/README.md)。下文原有0/1限制仍适用于只多策略；多空策略必须使用独立的带签名权重契约和执行类，不能直接向旧适配器写负数。

现有东京LightGBM和XGB15的指定权重、币池、模拟配置及数据快照已按用户明确授权发布，见[已发布模拟盘资产](published-paper.md)；这不改变新接入策略默认不公开私有文件的规则。

本文面向协作者和协作者的Codex。目标是把你已经研究/训练好的策略接入现有运行链路，
不是复制一套行情服务、Freqtrade worker或前端。

`dev`包含当前精简后的运行系统和多实例监控；`main`保留旧开源研究版本，两者目录和接口不同。
以本分支真实代码为准，不使用旧`src/alpha_web3_research/`导入路径。
仓库不分发交易历史或密钥。已发布策略可离线初始化，但联网、输入时效和模拟执行仍需单独验收。

## 1. 获取代码和安装环境

```bash
git clone --branch dev https://github.com/AddisonX/alpha-web3-research.git
cd alpha-web3-research
git switch -c feature/my-alpha-v1
uv sync --frozen --group freqtrade
uv run --no-sync pytest -q
```

需要Python 3.12、uv；前端需要Node.js 22+。macOS的LightGBM需要libomp，Linux需要libgomp。
测试使用合成模型验证通用链路，也验证已发布的真实权重及其哈希和配置。
测试通过不等于你的模型有效、完成历史回测或可以使用真实资金。
没有仓库写权限时先fork，PR仍以原仓库`dev`为目标；不要提交到`main`。

## 2. 先理解接入位置

```text
已收盘主网OHLCV → 因子矩阵 → 你的策略.decide(context)
                                   ↓
                            完整目标持仓0/1表
                                   ↓
               信号保存 → 安全检查 → 共用Freqtrade执行适配器
                                   ↓
                        独立实例数据库 → 多实例前端
```

先让Codex读取以下文件，再写适配代码：

| 文件 | 阅读目的 |
|---|---|
| `src/strategies/base.py` | 唯一的接口、上下文及输出校验 |
| `src/strategies/reversal.py` | 最简单的无模型策略参考 |
| `src/strategies/lightgbm.py` | 现有模型加载、预测与决策参考 |
| `src/factors/builder.py` | 平台已有因子的名称和公式 |
| `src/live/market.py` | 实际给策略准备什么输入 |
| `src/live/signal_store.py` | 如何转成执行信号，不要重复实现 |
| `src/strategies/registry.py` | 显式注册工厂，不动态执行任意包 |
| `scripts/run_dryrun.py` | 实例入口和策略允许列表 |
| `templates/strategy-import/` | 可选骨架、约束和离线验收示例 |

## 3. 你需要增加的文件

以下以`my-alpha-long-only-v1`为例，换成自己的唯一ID：

```text
src/strategies/my_alpha.py                 必需：协议适配、模型加载、推理、选币
tests/test_my_alpha.py                     必需：策略测试和发布兼容测试
docs/strategies/my-alpha-v1/README.md       必需：策略逻辑、参数、边界和测试结果
docs/strategies/my-alpha-v1/manifest.json   模型/参数/特征及权重摘要清单
data/models/strategies/my-alpha-long-only-v1/v1/   本机权重目录，Git忽略
data/imports/my-alpha/pairs.json           本机实际币池，Git忽略
```

代码、测试和非敏感说明通过Git交付。模型文件通过双方约定的私有文件渠道交付，
相对文件名和SHA-256写入manifest，接收方在加载前核对。未经明确发布授权，不要强行`git add -f`上传权重或数据。
不加载未经审核的Python代码或可执行反序列化文件；目前没有自动安装器、签名检查或模型下载服务。
若希望公开发布模型，另行确认授权和托管方式，本流程不默认公开。

模型清单必须包含：模型类型/依赖版本、权重文件及摘要、有序特征、原始值还是排名、
归一化/缺失处理、训练/验证/测试区间、标签的收益起止时刻与基准、策略参数和模型数量。
研究环境的指标、排名和预处理必须与线上完全一致，不能只导出权重就认为完成接入。

## 4. 实现策略协议

类名可以自定，也可从模板复制`ImportedLongOnlyStrategy`再修改。
必须实现`strategy_id`、`requirements: DataRequirements`、`metadata: StrategyMetadata`和`decide(context)`。

输入`StrategyContext`包含：

- `decision_timestamp`：UTC毫秒，表示已收盘决策bar的开始时刻，不是实际下单时刻。
- `features`：Polars表，包含`symbol`及requirements声明的因子列；没有持仓、账户状态或原始历史K线。

返回`StrategyDecision(context.decision_timestamp, targets)`，targets示例：

```text
pair                  target_position
BTC/USDT:USDT         1
ETH/USDT:USDT         0
```

必须覆盖全部输入币且各出现一次，1代表目标持有、0代表不持有，不是增量指令或仓位权重。
`model_score`、`model_score_rank_pct`、`score_position`是可选诊断；不使用排名选币也能接入。
调用`decision.validate(context)`检查结果。模型可在构造时加载，`decide()`不联网、不写文件、不读交易库、不下单。
新增依赖使用`uv add`或`uv add --group ...`并更新锁文件；不要替换项目的整个依赖清单。

当前支持范围：4h、只多、完整二元目标、至少60个有效币、至少一个入选；默认每日调仓、最多9仓、1倍杠杆、10%止损。
`metadata.horizon_bars`只描述预测跨度，不会改变调仓时钟。
若策略需要1h调仓、全部空仓、做空、连续权重、持仓状态或不同止损，不要伪装成兼容，先提出接口扩展。
数据层只计算现有FactorBuilder的因子；新增特征或数据源需要另行实现并测试平台组件。

## 5. 两个显式注册点

在`src/strategies/registry.py`增加独立分支，不替换默认分支。例如你的类接收`model_dir`：

```python
if strategy_id == "my-alpha-long-only-v1":
    from src.strategies.my_alpha import MyAlphaStrategy

    return MyAlphaStrategy(
        model_dir=production_root.parent / "strategies" / strategy_id / "v1",
    )
```

模型路径基于宿主data/models根目录，不能使用你电脑上的绝对路径，也不能覆盖`production/latest.json`。
规则策略直接返回无模型实例即可。
然后将新ID加入`scripts/run_dryrun.py`中`--strategy`的choices。
目前不是把文件放进目录就自动注册；前端自动发现的是实例，不是外部策略代码。

## 6. 必须通过的离线验收

```bash
uv run --no-sync pytest -q tests/test_my_alpha.py
uv run --no-sync pytest -q
```

至少覆盖：正常预测及目标完整性、确定性和输入乱序、重复/缺失币、缺失列、NaN/Inf、特征顺序、
训练/推理预处理一致性、模型缺失或摘要错误、同轮幂等，以及策略失败后保留旧Parquet并写failed状态。
模型摘要验证需要你的加载实现和测试，清单本身不会自动提供验证。
用临时目录、合成或脱敏样本运行，不联网、不修改真实实例。
可以参考`tests/test_strategy_components.py`和模板中的离线发布测试。
用未来数据扰动和时间对齐检查证明研究链路的因果性；只测试`decide()`并不能证明训练过程无泄露。

## 7. 创建自己的实例（审核后才联网）

先把双方约定的真实权重和币池放到本机忽略目录。
实际币池是去重的JSON数组，如`["BTC/USDT:USDT", ...]`，必须有足够可获取有效行情的币。
首次准备时显式传入它，避免新克隆依赖原作者私有tokyo-lightgbm/pairs.json：

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
from src.live.instances import prepare_instance

prepare_instance(
    root=Path.cwd(),
    source_pairs=Path("data/imports/my-alpha/pairs.json"),
    strategy_id="my-alpha-long-only-v1",
    instance_name="my-alpha-paper",
)
PY
```

这会同时生成pairs和Freqtrade白名单；后续不能只修改其中一份。
新实例的配置、信号、审计、SQLite、日志和心跳都在`freqtrade/instances/my-alpha-paper/`，不得借用旧实例目录。

```bash
# 限时联网启动检查，worker初始STOPPED，不启用模拟交易
uv run --no-sync python scripts/run_dryrun.py \
  --strategy my-alpha-long-only-v1 --instance my-alpha-paper --check-startup

# 以下会正式开始dry-run模拟，必须在审核、网络与权重核对完成后单独执行
uv run --no-sync python scripts/run_dryrun.py \
  --strategy my-alpha-long-only-v1 --instance my-alpha-paper
```

只使用交易所允许访问的网络；遇到区域限制不通过本流程绕过。
入口强制dry-run、主网公共行情和无交易凭证，不支持实盘真钱下单。
`--check-startup`只是框架/网络检查，不代替模型推理和完整发布测试。

## 8. 查看与提交

```bash
# 项目根目录另开终端
uv run --no-sync uvicorn src.live.dashboard:app --host 127.0.0.1 --port 8080
# 再另开终端
cd dashboard
npm ci
npm run dev -- --port 4173
```

本机4173的实例选择器会自动发现规范实例，权益历史从监控开始采集，不会补造过去曲线。
没有实例时显示空列表，不自动下载生产模型或启动交易。

提交前运行`git status`和`git diff --cached`确认没有权重、数据、密钥、IP或本机路径。
推送你的feature分支并创建PR，**base选择dev**；无写权限则从fork发PR。
PR说明：策略ID、修改文件、真实测试结果、依赖、兼容边界、权重私下交付方式和仍未验证的研究假设。
新增策略不应要求修改共用安全/执行/监控模块，确有需求时拆成单独讨论的接口扩展PR。

## 给协作者Codex的一段任务

```text
以当前dev分支及docs/STRATEGY_IMPORT_README.md为准，将我已有的模型和策略接入Alpha。
先阅读文档列出的接口源码；保留现有两个策略和公共执行链路。
新增独立策略模块、模型加载/预处理、测试、manifest和策略说明，并完成两个显式注册点。
权重放到专属data/models/strategies目录，币池放data/imports，不提交Git。
若有1h、全空仓、权重调仓、新因子或其他不兼容需求，先列出，不暗改平台约束。
运行离线验收及全量回归；未经我明确要求，不启动交易、不操作服务器。
将代码提交到feature分支并向dev提PR，不改main，不覆盖原生产模型和运行实例。
```
