# 协作开发

策略和模型接入先阅读[导入README](docs/STRATEGY_IMPORT_README.md)。

1. 以`dev`为基线创建`feature/<strategy-name>`分支，PR的目标分支选择`dev`，不直接修改`main`。
2. 使用Python 3.12及`uv sync --frozen --group freqtrade`安装环境，不存在旧版research依赖组。
3. 运行`uv run --no-sync pytest -q`。公共测试不需要生产权重，真实模型测试必须另行补充。
4. 前端变更在dashboard目录运行`npm ci`、`npm run build`和`npm run test:e2e`。
5. 策略、模型和实例分别版本化；模型权重通过约定的私有渠道交付，PR中只提交代码、manifest及摘要。
6. 不提交密钥、行情转储、交易数据库、日志、模型权重或私人部署地址。

原始贡献按本仓库MIT许可提交；第三方代码保留原有许可，并遵守THIRD_PARTY_NOTICES.md。
