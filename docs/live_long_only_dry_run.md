# Live Long-only Dry-run 运维说明

> 历史部署示例，非当前公开服务。地址已替换为文档保留地址；请先阅读 public_deployment.md。

## 部署状态

- 主服务器：`203.0.113.10`（示例）
- 旧服务器：`203.0.113.20`（历史示例）
- 用户：`ubuntu`
- 项目目录：`/home/ubuntu/alpha-web3-research`
- Python：3.12，由 uv 管理
- Freqtrade：2026.6
- 行情：Binance USDⓈ-M Futures 主网公开行情
- 订单：Freqtrade 本地 dry-run，不发送交易所订单
- 策略：`AlphaCrossSectionalLiveLongOnlyStrategy`
- 模型：4h / horizon 6 / `forward_return`，5 个 walk-forward 模型等权集成
- 调仓：每日 `00:00 UTC` K 线收盘后生成信号

东京服务器已验证 Binance Futures 主网 REST 与 WebSocket 连通。模型信号、
Freqtrade dry-run 价格和 Dashboard 持仓估值均使用主网公开行情，不需要交易
API Key，也不会向 Binance 发送真实订单。

Dashboard：

```text
http://203.0.113.10
```

HTTPS IP 证书已经签发并配置自动续期；腾讯云防火墙放行 TCP 443 后可直接使用
`https://203.0.113.10`。历史配置使用 Nginx Basic Auth；公开模板需要自行配置认证。

## 服务

```bash
sudo systemctl status alpha-freqtrade-dryrun.service
sudo systemctl status alpha-web3-signal.timer
sudo systemctl status alpha-web3-signal.service
```

Freqtrade 日志：

```bash
journalctl -u alpha-freqtrade-dryrun.service -f
```

信号生产日志：

```bash
journalctl -u alpha-web3-signal.service -n 100 --no-pager
```

重启：

```bash
sudo systemctl restart alpha-freqtrade-dryrun.service
```

暂停模拟盘：

```bash
sudo systemctl stop alpha-freqtrade-dryrun.service
```

## 实时产物

```text
freqtrade/user_data/pairs_live.json
freqtrade/user_data/signals/alpha_live_long_only_signals.parquet
freqtrade/user_data/signals/alpha_live_long_only_status.json
freqtrade/user_data/tradesv3.live-long-only.dryrun.sqlite
```

健康状态：

```bash
cat freqtrade/user_data/signals/alpha_live_long_only_status.json
```

状态必须满足：

- `state` 为 `healthy`
- `cross_section_count` 不低于 60
- `selected_count` 为 top 10% 的精确数量
- `fetch_error_count` 为 0 或处于可接受范围
- `grace_valid_until_time` 尚未过期

## 信号调度

systemd timer 每 15 分钟触发一次。当天决策已经健康生成时，任务会快速幂等退出，不重复抓取 85 个交易对。

新的 `00:00 UTC` 4h K 线在 `04:00 UTC` 收盘后才可用于推理。旧目标保留到该时刻，避免新信号生成前提前平仓。新信号会覆盖宽限行；若生产器持续失败，策略冻结模型驱动的开仓与换仓，不再将“缺失信号”误判为全数平仓。已有仓位继续受 Freqtrade 自身止损约束，但 dry-run 并没有交易所侧的保护单。

## 实盘安全层：第一阶段（本地实现，尚未部署）

本地实时策略增加了以下控制：

- 每次生成交易信号和最终开仓回调都复核状态 JSON：必须健康、未过期、时间戳有效，且选币数量与名单一致。
- 状态 JSON 与 Parquet 的决策时间、选币名单、有效期和横截面数量必须一致；缺失或损坏时停止模型驱动的进出场。
- 按最新目标状态持续对账。重启后即使错过目标变化的首根 K 线，也能在健康信号有效期内补开目标仓位、退出不再入选的仓位；仅使用最近收盘的 4h K 线。
- 同一轮决策中已平仓的币不会再次入场，避免止损后被持续的目标信号立刻买回。
- `confirm_trade_entry` 在 `dry_run` 不为 `true` 时无条件拒绝开仓。不要把这个版本当作已可交易的实盘策略。

这些改动**尚未部署到东京**，当前东京 dry-run 行为保持不变。先在隔离的 dry-run 实例上验证重启、信号超时、文件损坏、止损重入与名单变化，再考虑替换当前实例。

## 实盘安全层：第二阶段（本地实现，尚未部署）

`live audit-dry-run` 每次只读检查三类本地数据：生产信号状态、信号 Parquet 和 Freqtrade dry-run SQLite。它不调用交易所、不下单、不修改持仓或订单。报告原子写入 `freqtrade/user_data/signals/alpha_live_safety_audit.json`，包含目标名单、持仓、未完成订单、异常代码和 `entry_allowed`。

```bash
uv run alpha-web3-research live audit-dry-run
```

当前仓库中的本地信号可能早已过期，因此直接运行返回 `blocked` 和退出码 2 是预期行为；应在使用新生成信号的隔离 dry-run 实例上验收。规则如下：

- 信号过期、文件不一致、数据库不可读、重复/空头持仓、多余的非目标持仓或超过 15 分钟未完成的订单：`blocked`，拒绝**新开仓**。
- 目标仓位缺失超过 15 分钟：`degraded`，告警但允许策略继续补仓。同一轮已平仓的币不计作待补仓位。
- 新信号生成后的 15 分钟内允许正常调仓过渡；异常空头、重复仓位和超时订单不受此宽限。
- 报告停更超过 180 秒或与当前决策不一致时，开仓闸门拒绝新开仓。已有仓位的退出和 Freqtrade 止损不受该闸门阻挡。

异常变化及恢复会写入日志；设置 `ALPHA_SAFETY_WEBHOOK_URL` 后，也可向接收通用 JSON POST 的告警端点发送事件。Webhook 失败时报告变为 `blocked`，下一轮重试。不要将含令牌的 URL 写入仓库或命令行参数。

部署文件 `deploy/systemd/alpha-live-safety-audit.service` 和 `.timer` 每分钟运行一次审计。建议先只启用定时审计，检查报告和 `journalctl -u alpha-live-safety-audit.service`；确认稳定后，才将 `deploy/systemd/alpha-freqtrade-audit-gate.conf` 安装为 `/etc/systemd/system/alpha-freqtrade-dryrun.service.d/audit-gate.conf`，执行 `systemctl daemon-reload`，再重启 dry-run。未设置 `ALPHA_LIVE_AUDIT_PATH` 时，策略沿用第一阶段行为，不读取审计报告。启用闸门前，必须保证审计报告持续更新且决策时间、目标名单正确匹配；否则策略会拒绝新开仓。这些文件只是部署模板，**本次没有在东京安装或启用**。

目前的对账范围仅是本地 dry-run 数据库，不代表 Binance 实际持仓或真实保护单。进入真金白银交易前仍缺少：交易所持仓和挂单对账、交易所侧止损单及其重建/撤销流程、按账户权益计算的单币/总敞口上限、连续亏损和异常行情熔断，以及可回滚的发布与人工批准。当前真实开仓硬锁继续保留。

本机隔离故障演练的实际结果见 `docs/live_safety_rehearsal_20260922.md`。该演练已完成重启和故障注入，但尚未覆盖自然发生的完整调仓周期。

## 重新部署

本地测试：

```bash
uv run pytest -q
uv run alpha-web3-research live prepare-pairs --mainnet
uv run alpha-web3-research live write-dry-run-config --mainnet-market-data
uv run alpha-web3-research live generate-signals --mainnet --force
```

部署后必须在服务器重新运行信号生成，不能只复制本地信号文件。
