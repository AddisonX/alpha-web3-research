import { expect, test } from "@playwright/test";

const liveSnapshot = {
  instance: {
    id: "tokyo-lightgbm", display_name: "tokyo-lightgbm", strategy_id: "tokyo-lightgbm-long-only-v1",
    mode: "dry_run", runtime_state: "running", runtime_updated_at: null,
    audit_state: "healthy", entry_allowed: true, database_ready: true,
  },
  schema_version: 1,
  generated_at: "2026-07-27T11:40:00+00:00",
  mode: "dry_run",
  strategy: "AlphaCrossSectionalLiveLongOnlyStrategy",
  market_data_source: "binance_futures",
  price_state: "healthy",
  summary: {
    starting_balance: 10000,
    equity: 10048.35,
    wallet_balance: 10000,
    available_balance: 5500,
    total_pnl: 48.35,
    realized_pnl: 0,
    unrealized_pnl: 48.35,
    return_pct: 0.4835,
    max_drawdown_pct: 0.18,
    exposure: 4500,
    exposure_pct: 44.78,
    fees: 4.5,
    open_positions: 9,
    closed_trades: 0,
    wins: 0,
    losses: 0,
    win_rate_pct: null,
  },
  signal: {
    state: "healthy",
    generated_at: "2026-07-27T11:30:00+00:00",
    decision_time: "2026-07-27T00:00:00+00:00",
    next_rebalance_time: "2026-07-28T00:00:00+00:00",
    grace_valid_until_time: "2026-07-28T04:00:00+00:00",
    source_experiment: "rolling_walk_forward_ensemble",
    model_count: 5,
    timeframe: "1h",
    horizon_bars: 24,
    label_column: "forward_return_rank_pct",
    cross_section_count: 85,
    selected_count: 9,
    selected_pairs: [
      "BTC/USDT:USDT",
      "BNB/USDT:USDT",
      "TAO/USDT:USDT",
      "PAXG/USDT:USDT",
    ],
    fetch_error_count: 0,
  },
  positions: [
    {
      id: 1,
      pair: "BTC/USDT:USDT",
      side: "long",
      open_rate: 118250,
      current_rate: 119420,
      stake_amount: 500,
      amount: 0.0042,
      pnl_abs: 4.44,
      pnl_pct: 0.89,
      open_date: "2026-07-27T00:05:00+00:00",
      duration_minutes: 695,
      stop_loss: 106425,
      leverage: 1,
      entry_tag: "model_top_quantile",
      model_score: 0.012,
      model_score_rank_pct: 100,
      score_position: 1,
    },
    {
      id: 2,
      pair: "TAO/USDT:USDT",
      side: "long",
      open_rate: 412.5,
      current_rate: 405.3,
      stake_amount: 500,
      amount: 1.21,
      pnl_abs: -9.22,
      pnl_pct: -1.84,
      open_date: "2026-07-27T00:05:00+00:00",
      duration_minutes: 695,
      stop_loss: 371.25,
      leverage: 1,
      entry_tag: "model_top_quantile",
      model_score: 0.009,
      model_score_rank_pct: 98.8,
      score_position: 2,
    },
  ],
  closed_trades: [],
  orders: [
    {
      id: 2,
      trade_id: 2,
      pair: "TAO/USDT:USDT",
      side: "buy",
      status: "closed",
      order_type: "market",
      average: 412.5,
      filled: 1.21,
      cost: 500,
      created_at: "2026-07-27T00:05:00+00:00",
      filled_at: "2026-07-27T00:05:01+00:00",
    },
  ],
  equity_history: Array.from({ length: 48 }, (_, index) => ({
    timestamp: new Date(Date.parse("2026-07-27T00:00:00Z") + index * 15 * 60_000).toISOString(),
    equity: 10000 + index * 1.03 + Math.sin(index / 4) * 8,
    total_pnl: index * 1.03,
    realized_pnl: 0,
    unrealized_pnl: index * 1.03,
    return_pct: index * 0.0103,
  })),
};

const liveEquityHistory = {
  range: "all",
  interval: "raw",
  total_points: liveSnapshot.equity_history.length,
  filtered_points: liveSnapshot.equity_history.length,
  returned_points: liveSnapshot.equity_history.length,
  from: liveSnapshot.equity_history[0].timestamp,
  to: liveSnapshot.equity_history.at(-1)?.timestamp,
  points: liveSnapshot.equity_history,
};

test.beforeEach(async ({ page }) => {
  await page.route(/\/api\/instances(?:\?.*)?$/, route => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({ scope: "local", default_instance: "tokyo-lightgbm", errors: [], instances: [
      liveSnapshot.instance,
      { ...liveSnapshot.instance, id: "return-1-reversal", display_name: "return-1-reversal", strategy_id: "return-1-reversal-long-only-v1" },
    ] }),
  }));
  await page.route(/\/api\/instances\/[^/]+\/equity(?:\?.*)?$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(liveEquityHistory),
    }),
  );
  await page.route(/\/api\/instances\/[^/]+\/live(?:\?.*)?$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(liveSnapshot),
    }),
  );
});

test("renders live dry-run monitor without horizontal page overflow", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "tokyo-lightgbm", exact: true })).toBeVisible();
  await expect(page.getByText("+0.48%", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "当前持仓" })).toBeVisible();
  await expect(page.locator(".live-table").first().locator("tbody tr")).toHaveCount(2);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth),
  ).toBe(false);
  expect(
    await page.evaluate(() =>
      [...document.querySelectorAll(".nav-button")].every((button) => {
        const bounds = button.getBoundingClientRect();
        return bounds.left >= 0 && bounds.right <= window.innerWidth;
      }),
    ),
  ).toBe(true);
  expect(consoleErrors).toEqual([]);
});

test("recovers from one transient live API failure without showing a false alarm", async ({ page }) => {
  let requests = 0;
  await page.unroute(/\/api\/instances\/[^/]+\/live(?:\?.*)?$/);
  await page.route(/\/api\/instances\/[^/]+\/live(?:\?.*)?$/, (route) => {
    requests += 1;
    if (requests === 1) {
      return route.fulfill({ status: 502, body: "temporary upstream error" });
    }
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(liveSnapshot),
    });
  });

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "tokyo-lightgbm", exact: true })).toBeVisible();
  await expect(page.getByText("实时接口异常", { exact: true })).toHaveCount(0);
  await expect(page.locator(".inline-alert")).toHaveCount(0);
  expect(requests).toBe(2);
});

test("filters and zooms the complete live equity history", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "实时权益曲线" })).toBeVisible();
  await expect(page.getByText("48 个完整快照", { exact: false })).toBeVisible();
  await expect(page.locator(".recharts-brush")).toBeVisible();

  await page.getByRole("button", { name: "放大曲线" }).click();
  await expect(page.getByRole("button", { name: "复位曲线" })).toBeEnabled();
  await page.getByRole("button", { name: "复位曲线" }).click();
  await expect(page.getByRole("button", { name: "复位曲线" })).toBeDisabled();

  await page.getByRole("button", { name: "7天" }).click();
  await expect(page.getByRole("button", { name: "7天" })).toHaveClass(/active/);
  await page.getByLabel("权益数据粒度").selectOption("1h");
  await expect(page.getByLabel("权益数据粒度")).toHaveValue("1h");
});

test("live monitor does not depend on historical archives", async ({ page }) => {
  const historicalRequests: string[] = [];
  page.on("request", request => {
    if (request.url().includes("/data/backtests")) historicalRequests.push(request.url());
  });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "当前持仓" })).toBeVisible();
  await expect(page.getByText("策略对比", { exact: true })).toHaveCount(0);
  expect(historicalRequests).toEqual([]);
});

test("initial API failure exposes retry instead of infinite loading", async ({ page }) => {
  await page.unroute(/\/api\/instances\/[^/]+\/live(?:\?.*)?$/);
  await page.route(/\/api\/instances\/[^/]+\/live(?:\?.*)?$/, route => route.fulfill({ status: 503, body: "offline" }));
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("HTTP 503");
  await expect(page.getByRole("button", { name: "重试连接" })).toBeVisible();
});

test("switches positions and equity history without mixing instances", async ({ page }) => {
  await page.route("**/api/instances/return-1-reversal/live?*", route => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({ ...liveSnapshot,
      instance: { ...liveSnapshot.instance, id: "return-1-reversal", display_name: "return-1-reversal" },
      signal: { ...liveSnapshot.signal, model_count: 0 },
      summary: { ...liveSnapshot.summary, equity: 9876.54, return_pct: -1.2346 },
      positions: [liveSnapshot.positions[0]],
    }),
  }));
  await page.route("**/api/instances/return-1-reversal/equity?*", route => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({ ...liveEquityHistory, total_points: 3, points: liveSnapshot.equity_history.slice(0, 3) }),
  }));
  await page.goto("/");
  await expect(page.getByText("48 个完整快照", { exact: false })).toBeVisible();
  await page.getByLabel("监控实例").selectOption("return-1-reversal");
  await expect(page.getByRole("heading", { name: "return-1-reversal", exact: true })).toBeVisible();
  await expect(page.getByText("-1.23%", { exact: true })).toBeVisible();
  await expect(page.getByText("规则策略", { exact: true })).toBeVisible();
  await expect(page.locator(".live-table").first().locator("tbody tr")).toHaveCount(1);
  await expect(page.getByText("3 个完整快照", { exact: false })).toBeVisible();
  await page.getByLabel("监控实例").selectOption("tokyo-lightgbm");
  await expect(page.getByText("+0.48%", { exact: true })).toBeVisible();
  await expect(page.getByText("48 个完整快照", { exact: false })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
});

test("discovers additional instances without changing the page", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel("监控实例").locator("option")).toHaveCount(2);
  await page.route(/\/api\/instances(?:\?.*)?$/, route => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({ scope: "local", default_instance: "tokyo-lightgbm", errors: [], instances: [
      liveSnapshot.instance,
      { ...liveSnapshot.instance, id: "return-1-reversal", display_name: "return-1-reversal" },
      { ...liveSnapshot.instance, id: "new-strategy", display_name: "new-strategy" },
    ] }),
  }));
  await page.getByRole("button", { name: "刷新数据" }).click();
  await expect(page.getByLabel("监控实例").locator("option")).toHaveCount(3);
  await expect(page.getByLabel("监控实例")).toHaveValue("tokyo-lightgbm");
});

test("handles an empty catalog without fabricating account data", async ({ page }) => {
  await page.route(/\/api\/instances(?:\?.*)?$/, route => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({ scope: "local", default_instance: null, instances: [], errors: [] }),
  }));
  await page.goto("/");
  await expect(page.getByText("暂无可监控的本地实例")).toBeVisible();
  await expect(page.getByLabel("监控实例")).toBeDisabled();
  await expect(page.getByRole("heading", { name: "实时权益曲线" })).toHaveCount(0);
});

test("a late response from the previous instance cannot overwrite the new one", async ({ page }) => {
  let release: () => void = () => undefined;
  let blocked = false;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route("**/api/instances/tokyo-lightgbm/live?*", async route => {
    blocked = true;
    await gate;
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(liveSnapshot) }).catch(() => undefined);
  });
  await page.route("**/api/instances/return-1-reversal/live?*", route => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({ ...liveSnapshot,
      instance: { ...liveSnapshot.instance, id: "return-1-reversal", display_name: "return-1-reversal" },
      summary: { ...liveSnapshot.summary, return_pct: -1.2346 },
    }),
  }));
  await page.goto("/");
  await expect.poll(() => blocked).toBe(true);
  await page.getByLabel("监控实例").selectOption("return-1-reversal");
  await expect(page.getByRole("heading", { name: "return-1-reversal", exact: true })).toBeVisible();
  release();
  await page.waitForTimeout(200);
  await expect(page.getByText("-1.23%", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "tokyo-lightgbm", exact: true })).toHaveCount(0);
});

