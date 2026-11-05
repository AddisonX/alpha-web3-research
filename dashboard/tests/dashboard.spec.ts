import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";

const backtests = JSON.parse(readFileSync(new URL("../public/data/backtests.example.json", import.meta.url), "utf8"));

const liveSnapshot = {
  schema_version: 1,
  generated_at: "2026-07-27T11:40:00+00:00",
  mode: "dry_run",
  strategy: "AlphaCrossSectionalLiveLongOnlyStrategy",
  market_data_source: "binance_futures_testnet",
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
  await page.route(/\/data\/backtests\.json(?:\?.*)?$/, (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(backtests) }),
  );
  await page.route(/\/api\/live\/equity(?:\?.*)?$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(liveEquityHistory),
    }),
  );
  await page.route(/\/api\/live(?:\?.*)?$/, (route) =>
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
  await expect(page.getByRole("heading", { name: "AlphaCrossSectionalLiveLongOnlyStrategy" })).toBeVisible();
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
  await page.unroute(/\/api\/live(?:\?.*)?$/);
  await page.route(/\/api\/live(?:\?.*)?$/, (route) => {
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
  await expect(page.getByRole("heading", { name: "AlphaCrossSectionalLiveLongOnlyStrategy" })).toBeVisible();
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

test("renders strategy analytics without horizontal page overflow", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });

  await page.goto("/");
  await page.getByRole("button", { name: "策略总览" }).click();
  await expect(page.getByRole("heading", { name: "SyntheticExampleStrategy" })).toBeVisible();
  await expect(page.getByLabel("核心指标").getByText("+3.00%", { exact: true })).toBeVisible();
  await expect(page.locator(".recharts-wrapper svg")).toHaveCount(3);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth),
  ).toBe(false);
  expect(consoleErrors).toEqual([]);
});

test("switches between comparison and filtered trade views", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "策略对比" }).click();
  await expect(page.getByRole("heading", { name: "运行表现矩阵" })).toBeVisible();
  expect(await page.locator(".strategy-table tbody tr").count()).toBeGreaterThan(0);

  await page.getByRole("button", { name: "交易明细" }).click();
  await page.getByPlaceholder("搜索交易对").fill("BTC");
  await expect(page.locator(".trades-table tbody tr").first()).toContainText("BTC");
  await expect(page.locator(".trades-table tbody tr")).toHaveCount(1);
});
