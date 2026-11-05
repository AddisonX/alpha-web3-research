# Alpha Web3 Research

A reproducible crypto factor-research toolkit with versioned strategies, historical
backtesting, Freqtrade integration, and a React monitoring dashboard.

面向 Binance USDT 本位永续合约的研究系统：行情采集、币池构建、因子与标签、
LightGBM/XGBoost、单因子分层诊断、组合回测和模拟盘监控。

**Research software, not investment advice or a production trading guarantee.**
Historical results are diagnostic artifacts, not evidence of repeatable returns.

## Scope

- CCXT and checksum-verified Binance public archives; Parquet storage and DuckDB queries.
- Momentum/reversal, volatility, volume and liquidity-proxy factors on 1h/4h data.
- IC/RankIC diagnostics, decile returns, correlation analysis and feature selection.
- Versioned daily cross-sectional, hourly multi-horizon and point-in-time-universe strategies.
- LightGBM and XGBoost experiments, chronological splits and label-boundary purging.
- Freqtrade backtest adapters, dry-run monitoring, reconciliation and testnet safety checks.
- React dashboard for strategy comparison, trades and live simulated equity.

不同研究策略独立保存，不能把它们的训练区间、币池或收益口径混用。
2023起点的动态月度币池目前是研究设计方向，尚未实现。

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), and Node.js 22+ for the dashboard.
LightGBM on macOS may require `brew install libomp`.

```bash
git clone https://github.com/AddisonX/alpha-web3-research.git
cd alpha-web3-research
uv sync --frozen --group freqtrade --group research
cp .env.example .env
uv run --no-sync alpha-web3-research --help
uv run --no-sync pytest -q
```

Public OHLCV downloads do not require a trading API key. Access and archive coverage
depend on the exchange and your jurisdiction; do not bypass access restrictions.
No data download, model training or trading process starts merely by installing.

## Research workflows

Inspect the available commands before launching large downloads:

```bash
uv run --no-sync alpha-web3-research data --help
uv run --no-sync alpha-web3-research research --help
uv run --no-sync alpha-web3-research strategies list
```

The fixed-inception-universe workflow downloads historical data and builds its own
artifacts. It may take time and disk space:

```bash
uv run --no-sync alpha-web3-research research prepare-pit-rank
uv run --no-sync alpha-web3-research research run-pit-rank
```

Other workflows and their assumptions:

| Workflow | Documentation |
| --- | --- |
| Data adapters and storage | [Data architecture](docs/data_architecture.md) |
| Strategy definitions and archived runs | [Strategy registry](docs/strategy_registry.md) |
| Hourly multi-horizon allocation | [Hourly rolling strategy](docs/hourly_rolling_strategy.md) |
| Fixed 2025 inception universe / LightGBM | [PIT Rank64](docs/pit_rank64_strategy.md) |
| Fixed-year, 300-tree XGBoost experiment | [PIT XGBoost](docs/pit_xgboost_2026.md) |
| Historical Freqtrade integration | [Freqtrade guide](docs/freqtrade_strategy_guide.md) |

Date-stamped scripts in `scripts/build_factor_*` reproduce specific historical
experiments and expect the corresponding local data/model artifacts. Those artifacts
are intentionally not included; these scripts are not a one-command fresh-clone demo.

## Dashboard

The repository includes a small **synthetic** backtest example, not live account data:

```bash
cp dashboard/public/data/backtests.example.json dashboard/public/data/backtests.json
cd dashboard
npm ci
npm run dev
```

Open the URL printed by Vite (default `http://127.0.0.1:4173`) and choose
**策略总览** to inspect the synthetic example. The initial live tab requires a configured
dry-run backend; an unavailable live endpoint is expected on a fresh clone.

For a configured local simulation, the read-only API can be started from the project root:

```bash
uv run --no-sync uvicorn alpha_web3_research.live.dashboard:app --host 127.0.0.1 --port 8080
```

The API requires locally generated signals/trade data; it does not create a simulation.
See [deployment boundaries](docs/public_deployment.md) before exposing any endpoint.

## Verification

```bash
uv run --no-sync pytest -q
cd dashboard
npm run build
npx playwright install chromium
npm run test:e2e
```

Tests use temporary or synthetic data. They do not require private reports, credentials,
the original research datasets, or a running trading account.

## Research limitations

- The original `training.py` walk-forward metric named `target_rank_ic` uses Pearson
  correlation. Historical reports using that name must not be interpreted as Spearman
  RankIC. The newer factor diagnostics explicitly compute Spearman.
- The frozen 64-factor list was selected using a 2026 correlation snapshot. Extending
  its charts backward does not turn those results into clean holdout evidence.
- Some legacy workflows use a current-universe snapshot and have survivorship bias.
- Daily labels sampled every four hours overlap. Non-overlapping daily spread charts
  and 4h correlation statistics have different sampling frequencies.
- Simple-return sums, gross long-short spreads and compounded execution returns are
  different quantities. Costs, funding, liquidity and execution assumptions vary by run.
- Source gaps and delistings must remain visible. No guarantee of complete market history
  or profitable execution is made.

## Public release boundaries

This is a sanitized source release. Secrets, real server addresses, SSH keys, raw market
data, model weights, generated reports, account databases, private notes and dependency
bundles are excluded. Deployment files are templates, not a public account or endpoint.
Existing private deployments are neither configured nor modified by this release.

## License

Original project code: [MIT](LICENSE). Dependencies retain their licenses.
Freqtrade is GPLv3; an applicable combined distribution is not MIT-only.
Read [third-party notices](THIRD_PARTY_NOTICES.md) before redistributing integrations.
