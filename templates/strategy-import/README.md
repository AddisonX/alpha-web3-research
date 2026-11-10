# 策略导入模板

这是可交给另一边Codex修改的策略交付包，不是自动安装器。
示例选择`return_1`最高的前10%（最多9币），仅演示接口，不代表有效alpha。
模板没有注册、没有启动交易，也不会覆盖已有实例。

## 一、给对方什么

最好提供同一版本的项目代码及本目录，然后发送`CODEX_PROMPT.md`里的提示词。
`AGENTS.md`约束本目录的生成工作，不是可替代测试的安全隔离机制。
若不能提供整个项目，至少提供下面这些真实源码，避免对方重建不兼容的接口：

```text
src/strategies/base.py          唯一接口定义
src/strategies/reversal.py      规则策略参考
src/live/market.py             实际输入与特征准备方式
src/factors/builder.py          已支持的因子名称和公式
src/live/signal_store.py        输出转换约定
```

在其他项目训练模型没关系，但最终必须写一个符合这里接口的推理适配类。
只给一个Freqtrade策略类或模型权重，无法直接接入当前决策链路。

## 二、交付内容

```text
strategy.py             策略代码，入口ImportedLongOnlyStrategy
manifest.json           与代码一致的审核清单
test_strategy.py        离线契约、策略逻辑及发布兼容测试
README.md               替换为真实策略说明和测试结果
AGENTS.md               生成约束
CODEX_PROMPT.md          可直接发送给Codex的任务
artifacts/              模型策略放真实权重；规则策略可无模型
```

自定义币池可额外交付`pairs.json`，格式是去重的`["BTC/USDT:USDT", ...]`。
默认运行需要至少60个有效币，不要把只有两三个币的格式示例当成运行币池。
新依赖写进manifest的`additional_dependencies`，接收方审核后通过uv加入宿主环境。
不要交付数据库、密钥、`.env`、日志、缓存、虚拟环境或已生成的实时信号。

`manifest.json`不是动态插件入口，参数也不会自动注入策略或改变执行配置。
当前示例在Python中定义参数，测试检查清单与代码一致。
模型包的额外说明见`artifacts/README.md`，不要覆盖既有生产模型指针。

## 三、当前接口边界

- 输入：`StrategyContext(decision_timestamp, features)`。features只有`symbol`及声明的因子列，不是原始完整OHLCV。
- 时间：UTC毫秒的决策bar开始时刻；只有该bar收盘后才可用于决策，不能理解为在开盘时已经知道其完整因子。
- 输出：同一时间戳及完整`pair/target_position`表，每个输入币恰好一次。1是目标持有，0是目标不持有，不是资金比例。
- 分数/排序为可选诊断；规则示例只提供排序，不伪造模型分数。
- 当前只支持4h、只多、二元目标，至少一个入选；不支持全空仓、做空、连续权重或持仓状态输入。
- 当前默认每天调仓、最多9仓、1倍杠杆、10%止损。预测horizon不改变调仓频率；新选币逻辑不能绕过公共风险约束。
- 行情层仅提供FactorBuilder已有特征，新增因子或其他数据源需要明确申请独立的平台扩展。
- 异常应抛错；发布层标记failed并保留之前的信号文件，不能把异常偷偷变成随机选币或清仓指令。

示例的`horizon_bars=6`只是24小时参考跨度，不是训练出的收益预测，也不会强制持有满24小时。

## 四、在宿主项目离线验收

在项目根目录运行，要求宿主依赖已经通过uv准备好：

```bash
uv run --no-sync pytest -q templates/strategy-import/test_strategy.py
uv run --no-sync pytest -q
```

测试只使用临时目录和合成特征，不联网、不下单、不修改现有实例。
将模板用于新策略时，保留通用契约测试，并把示例策略断言、合成特征和模型测试改成真实策略对应内容。
这些测试验证接入兼容性，不等于历史回测、样本外有效性或无未来函数的完整证明。
研究交付还应说明训练/测试切分，并做未来数据扰动、时间对齐和训练/推理一致性检查。

## 五、接收方接入步骤

1. 人工检查代码、依赖和模型，先运行离线测试；不要自动执行未经审核的外部包。
2. 给策略唯一ID，例如`my-alpha-long-only-v1`，同步更新代码、manifest及说明。
3. 将`strategy.py`放到`src/strategies/my_alpha.py`，整理测试到`tests/test_my_alpha.py`；调整测试导入为宿主模块。
4. 模型放入`data/models/strategies/<strategy_id>/<version>/`，构造函数显式接收该目录，保留特征顺序和预处理。
5. 在`src/strategies/registry.py`增加显式工厂分支；如有模型，在该分支传入专属模型目录。不要替换默认策略分支。
6. 将新策略ID加入`scripts/run_dryrun.py`的`--strategy`允许列表。目前需要这一步，不是放文件就能自动运行。
7. 准备独立实例，核查币池、信号路径和交易库。自定义币池必须在准备配置时使用同一份源文件生成白名单，不能只改pairs.json。
8. 审核后做联网启动检查，再由用户决定开始dry-run。前端会自动发现规范实例，无需新增页面。

以下命令仅在完成注册且替换真实ID以后可用；当前模板ID会被拒绝：

```bash
uv run --no-sync python scripts/run_dryrun.py \
  --strategy my-alpha-long-only-v1 --instance my-alpha-paper --prepare-only
uv run --no-sync python scripts/run_dryrun.py \
  --strategy my-alpha-long-only-v1 --instance my-alpha-paper --check-startup
# 以下命令会启动模拟交易，须审核后单独执行
uv run --no-sync python scripts/run_dryrun.py \
  --strategy my-alpha-long-only-v1 --instance my-alpha-paper
```

原始包留作交付记录，不把模板整个目录复制进freqtrade/strategies。
复用共用执行适配类，不能为了新策略复制监督、安全检查、交易API或前端。

## 六、真实策略README必须补齐

- 策略ID/版本、作者、变更说明、选币/退出逻辑和全部参数。
- 币池来源和时点规则、特征公式、频率、历史长度与缺失处理。
- 模型训练/验证/测试区间、特征顺序、标签公式、推理预处理与权重摘要。
- 费用、滑点、资金费率及止损的研究假设；哪些属于研究假设，哪些当前执行确实支持。
- 测试命令、实际结果、所需依赖和不兼容项。没有跑过的实验明确标为未验证。
