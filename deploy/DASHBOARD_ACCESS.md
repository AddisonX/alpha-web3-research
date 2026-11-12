# 看板访问保护

443 旧站和 8443 多实例站使用 HTTPS Basic Auth，网页、静态资源及全部 API 均需认证。
80 端口只保留 HTTPS 跳转和证书续期的 ACME 验证目录，不提供看板 API。

## 安装约定

- `alpha-dashboard-limits.conf` 安装至 `/etc/nginx/conf.d/`，定义共享限流区。
- `alpha-dashboard-access.conf` 安装至 `/etc/nginx/snippets/`，由两个 HTTPS server 引入。
- 用 `htpasswd -cB /etc/nginx/auth/alpha-dashboard.htpasswd alpha` 交互创建凭证，不在命令行传递密码。
- 密码目录权限 `root:www-data 0750`，密码哈希文件权限 `root:www-data 0640`；不要放在网站根目录。
- 使用高熵随机密码；明文不放入仓库、部署包、URL或访问日志。修改现有用户时不要使用 `-c`，以免覆盖其他用户。
- 先 `sudo nginx -t`，成功后 `sudo systemctl reload nginx`；不重启 API 或交易进程。

## 限制

两个端口共享每 IP 每秒 10 次请求、40 次突发，以及最多 20 个并发活动请求。
API 另限每 IP 每秒 3 次、20 次突发；超限返回 429。
因为 location 自定义限流不继承 server 的限流项，API 显式配置两项。
网站及 API 只允许 GET/HEAD；代理转发时清除浏览器的 Authorization，不向应用传递看板密码。
接口 8080、18081 仍只监听本机，不能通过公网绕过认证。

浏览器进入时会弹出账号密码框，通常会在本次浏览器会话中记住凭证。
应用自身只调用本机接口，不依赖外部 Nginx 认证，因此不影响交易与权益采集。

验收应包含两个端口的匿名/错误密码 401、正确密码网页及 API 200、写请求 403、受控限流 429，以及证书验证目录不要求登录。
保留现有配置备份和只读健康检查，不以绕过 TLS 验证来代替证书检查。

此保护用于限制访问和降低请求滥用，并不能代替补丁管理、SSH 安全或云侧大流量攻击防护。
