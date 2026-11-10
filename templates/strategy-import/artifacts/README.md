# 模型文件放置约定

当前示例是规则策略，不需要任何模型。不要放置假的模型文件。

模型策略在此目录交付真实权重，例如`model.txt`或`model.ubj`，并更新上层`manifest.json`：

- `model.kind`：模型类型及版本；`metadata.model_count`：实际模型数量。
- `model.files`：每个模型的相对路径和SHA-256，例如`{"file": "artifacts/model.txt", "sha256": "实际64位摘要"}`。
- `model.training_start/end`、`validation_start/end`：实际训练及验证区间，标明UTC。
- `model.label_definition`：收益起止价格、预测跨度、超额基准及是否排名。
- `preprocessing`：有序特征、原始值或截面排名、排名方向、并列排名、归一化、缺失处理。
- `additional_dependencies`：模型加载所需的额外依赖及版本约束。

构造策略时接收本地模型目录并加载权重；`decide()`只在内存中推理。
不使用对方电脑的绝对路径，不覆盖本项目`data/models/production/latest.json`。
接收方审核后放入独立的`data/models/strategies/<strategy_id>/<version>/`，并显式传入路径。

上述清单是交付规范，当前平台不会自动验证摘要或按manifest动态加载模型。
未审核的Python代码、pickle/joblib等可执行反序列化内容不能直接运行。
