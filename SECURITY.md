# Security

Do not include API keys, passwords, SSH keys, wallet secrets, account databases or
private deployment addresses in issues, pull requests, screenshots or logs.

Use GitHub private vulnerability reporting for security issues. If private reporting
is unavailable, do not publish exploit details or credentials in a public issue.

Keep APIs bound to loopback by default. Add authentication, TLS, access controls and
appropriate firewall rules before exposing a dashboard or RPC endpoint. A read-only
dashboard can still expose sensitive balances, positions and operational information.

Use separate credentials for testnet and live environments. Never infer live-trading
readiness from a passing backtest or testnet smoke test. Do not grant withdrawal
permissions to research bots. Rotate any accidentally disclosed credential immediately.

The repository does not ship a funded account, production secrets or a managed service.
