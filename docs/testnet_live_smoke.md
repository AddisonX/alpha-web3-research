# Binance Futures Testnet 小额真实测试订单

> 历史部署记录，非当前公开服务。地址为文档保留地址，不可用于连接真实服务器。

## 当前状态

示例服务器 `203.0.113.10` 曾部署独立的测试网执行目录 `/home/ubuntu/alpha-testnet` 和
`alpha-freqtrade-testnet-live.service`，但服务保持 **disabled / inactive**，没有
`credentials.json`，也没有 `ARMED` 文件，因此不会发送订单。原有
`alpha-freqtrade-dryrun.service` 保持运行，数据库和 Dashboard 不变。

已通过的检查：东京 Testnet 公共 REST 可达；当前模型目标币在 Testnet 上市；
Freqtrade 2026.6 能解析配置和加载策略；隔离的 Testnet 公共行情 dry-run 能启动并
正常退出。尚未通过的检查：测试网签名账户、真实测试订单、交易所侧止损单与持仓
对账。**不要把当前状态当成已经跑通测试网真实下单。**

Freqtrade 当前 FAQ 不把沙盒账户列为正式支持路径。本实例使用 CCXT Testnet 地址
固定映射，仅用于小额集成测试，不用于评价策略收益。信号读取现有主网公共行情
生产器；订单路由指向 Binance USD-M Futures Testnet。

## 隔离与限额

- 测试网专用配置、策略副本、用户目录、SQLite、服务名和 API 凭据文件与现有
  dry-run 完全隔离。
- 最多 1 仓、每笔固定 20 测试 USDT、1 倍杠杆、单币 10% 止损。
- 入场和出场使用市价单；配置要求交易所侧市价止损。首次成交后必须实际核查
  Testnet 上是否存在对应止损单，不能只凭配置推断保护已生效。
- 不启用 API Server 或 Telegram；策略回调每次入场前重新校验信号、测试网
  路由和 `ARMED` 文件。
- 旧 `AlphaCrossSectionalLiveLongOnlyStrategy` 仍硬性拒绝 `dry_run=false`。

## 启动前必须完成

1. 在 Binance Futures Testnet 创建**专用**交易 API Key。不要使用主网 API Key，
   不要把 Key 或 Secret 粘贴到聊天或提交到仓库。测试网账户应为 One-way Mode、
   Single-Asset Mode，至少有 25 测试 USDT 可用，且没有既有仓位或挂单。
2. 由账户持有人把凭据放到东京服务器
   `/home/ubuntu/alpha-testnet/credentials.json`，文件权限设为 `0600`，内容仅为：

   ```json
   {"exchange": {"key": "<testnet-key>", "secret": "<testnet-secret>"}}
   ```

3. 运行只读签名预检：

   ```bash
   cd /home/ubuntu/alpha-web3-research
   .venv/bin/python scripts/testnet_preflight.py \
     --config /home/ubuntu/alpha-testnet/config.json \
     --credentials /home/ubuntu/alpha-testnet/credentials.json --account
   ```

   预检验证地址映射、凭据权限与格式、持仓模式、资产模式、可用余额、空仓和无
   挂单。不通过时不要创建 `ARMED`，不要启动服务。

4. 核查模型信号健康且目标币在 Testnet 可交易，确认已有 dry-run 正常，然后
   才创建 `/home/ubuntu/alpha-testnet/ARMED` 并启用独立服务。首次订单后检查
   Binance Testnet 的实际成交、仓位、止损单、独立 SQLite 和日志；发现偏差
   立即停止服务并在 Testnet 页面人工核查残留仓位/挂单。

停机命令：

```bash
sudo systemctl stop alpha-freqtrade-testnet-live.service
```

停机不会自动保证测试网仓位已经平掉或条件单已撤销，必须以 Testnet 账户实际
状态为准。现有 Dashboard 不展示这个独立测试网账户。
