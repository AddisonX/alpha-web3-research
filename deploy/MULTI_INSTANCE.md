# 独立三实例服务器部署

部署根目录`/home/ubuntu/alpha-web3-runtime`，与旧`alpha-web3-research`目录隔离。
新前端Nginx端口8443，API仅监听127.0.0.1:18081；旧443/8080服务不修改。
Nginx示例需用实际证书域名或IP替换SERVER_NAME，安全组与主机防火墙只放行8443，不对外开放18081。
沿用已有证书续期及Nginx reload机制，不复制私钥到项目或部署包。
新旧两个HTTPS入口均使用服务端访问认证，部署前需安装共享限流配置、认证片段和密码哈希文件，见[DASHBOARD_ACCESS.md](DASHBOARD_ACCESS.md)。不要在后续发布时覆盖掉认证。

三个实例是新建的独立10,000 USDT模拟账户，不导入本地持仓、成交或旧服务器历史。
只传输代码、模型、年度池及市值，不传递数据库、PID、信号、凭证或开发机配置。

| 实例 | service.env中的ALPHA_STRATEGY_ID |
|---|---|
| tokyo-lightgbm | tokyo-lightgbm-long-only-v1 |
| return-1-reversal | return-1-reversal-long-only-v1 |
| xgb15-paper | xgb15-120-long-short-v1 |

先通过uv在新目录创建独立环境，运行回归、模型/快照复现和资产检查。
分别用`prepare_instance(root, source_pairs, strategy_id=..., instance_name=...)`生成服务器本机路径的配置。
每个实例启动前做STOPPED检查，确认没有订单后才启用对应的`alpha-multi-instance@<name>.service`。
所有实例始终dry-run、无交易密钥；禁止复用旧数据库及覆盖旧服务单元。

`alpha-multi.slice`限制新环境资源，不给旧服务设置资源或依赖限制。启动时逐个验收，观察内存与API限流。
`alpha-multi-dashboard.service`负责独立权益采集；Nginx只允许读请求。
每日UTC06:20（北京时间14:20，最多延迟5分钟）通过timer刷新市值，随后检查资产；错过计划会补执行。
刷新只写`data/imports/xgb15/runtime/caps.parquet`，保留发布快照，数据不足或拒绝访问时失败关闭。
失败记录在systemd journal；这里不宣称已接入外部告警渠道，应定期查看timer/service状态。

```bash
systemctl status alpha-multi-instance@tokyo-lightgbm
systemctl status alpha-multi-instance@return-1-reversal
systemctl status alpha-multi-instance@xgb15-paper
systemctl status alpha-multi-dashboard
systemctl list-timers alpha-multi-caps-refresh.timer
journalctl -u alpha-multi-caps-refresh.service --since today
```

部署验收包括：旧服务PID/配置未变、新实例审计与成交、对外TLS和API、三实例切换及手机布局。
换用新目录/端口的部署不代表迁移或停止旧系统，也不等于生产交易认证。
