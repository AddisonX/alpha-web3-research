# Contributing

1. Use Python 3.12 and `uv sync --frozen --group freqtrade --group research`.
2. Add focused tests and run `uv run --no-sync pytest -q` after installing the groups above.
3. For dashboard changes, run `npm ci`, `npm run build`, and `npm run test:e2e` in `dashboard/`.
4. Keep strategy versions and experiment artifacts separate. Document timing, universe,
   execution assumptions and any use of future data.
5. Do not commit market-data dumps, trained weights, account files or secrets.

Contributions of original code are submitted under the repository's MIT license.
Do not copy third-party source without preserving its license and checking compatibility.
