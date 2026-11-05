import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleDollarSign,
  Clock3,
  Download,
  Gauge,
  LayoutDashboard,
  ListFilter,
  RadioTower,
  RefreshCw,
  RotateCcw,
  Search,
  Server,
  SlidersHorizontal,
  TableProperties,
  TrendingDown,
  TrendingUp,
  WalletCards,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  Brush,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import {
  formatCompact,
  formatBacktestDuration,
  formatDate,
  formatDateTime,
  formatDuration,
  formatMoney,
  formatNumber,
  formatPct,
  runLabel,
  shortRunId,
} from "./format";
import type {
  DashboardData,
  EquityPoint,
  LiveDashboardData,
  LiveEquityHistoryResponse,
  LiveEquityPoint,
  StrategyRun,
  SummaryRow,
  Trade,
} from "./types";

type View = "live" | "overview" | "strategies" | "trades";
type EquityMode = "balance" | "returnPct";
type LiveHistoryRange = "24h" | "7d" | "30d" | "all";
type LiveHistoryInterval = "auto" | "raw" | "15m" | "1h" | "4h" | "1d";

const RUN_COLORS = ["#126b55", "#2d5f9a", "#c47b22", "#99505e", "#6c5d97"];
const POSITIVE = "#126b55";
const NEGATIVE = "#c4473a";
const NEUTRAL = "#77817d";

function valueColor(value: number | null): string {
  if (value === null || value === 0) return NEUTRAL;
  return value > 0 ? POSITIVE : NEGATIVE;
}

async function requestLiveSnapshot(): Promise<LiveDashboardData> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 12_000);
  try {
    const response = await fetch(`/api/live?t=${Date.now()}`, {
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`实时 API 返回 HTTP ${response.status}`);
    if (!response.headers.get("content-type")?.includes("application/json")) {
      throw new Error("实时 API 返回了非 JSON 内容");
    }
    return (await response.json()) as LiveDashboardData;
  } catch (caught) {
    if (caught instanceof DOMException && caught.name === "AbortError") {
      throw new Error("实时 API 请求超过 12 秒");
    }
    if (caught instanceof TypeError) {
      throw new Error("浏览器无法连接实时 API，请检查网络后刷新页面");
    }
    throw caught;
  } finally {
    window.clearTimeout(timeout);
  }
}

function App() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [liveData, setLiveData] = useState<LiveDashboardData | null>(null);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [liveLoading, setLiveLoading] = useState(true);
  const [activeView, setActiveView] = useState<View>("live");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const liveRequestActive = useRef(false);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/data/backtests.json?t=${Date.now()}`, {
        cache: "no-store",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const nextData = (await response.json()) as DashboardData;
      if (!nextData.runs.length) throw new Error("结果中没有策略运行");
      setData(nextData);
      setSelectedRunId((current) => {
        if (current && nextData.runs.some((run) => run.id === current)) return current;
        return nextData.runs.find((run) => run.isLatest)?.id ?? nextData.runs[0].id;
      });
      setCompareIds((current) => {
        const valid = current.filter((id) => nextData.runs.some((run) => run.id === id));
        return valid.length ? valid : nextData.runs.slice(0, 3).map((run) => run.id);
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法读取回测数据");
    } finally {
      setLoading(false);
    }
  }, []);

  const loadLive = useCallback(async () => {
    if (liveRequestActive.current) return;
    liveRequestActive.current = true;
    setLiveLoading(true);
    try {
      let nextData: LiveDashboardData | null = null;
      let lastError: unknown = null;
      for (let attempt = 0; attempt < 2; attempt += 1) {
        try {
          nextData = await requestLiveSnapshot();
          break;
        } catch (caught) {
          lastError = caught;
        }
      }
      if (!nextData) throw lastError;
      setLiveData(nextData);
      setLiveError(null);
    } catch (caught) {
      setLiveError(caught instanceof Error ? caught.message : "无法读取实时模拟盘");
    } finally {
      liveRequestActive.current = false;
      setLiveLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  useEffect(() => {
    void loadLive();
    const timer = window.setInterval(() => void loadLive(), 10_000);
    return () => window.clearInterval(timer);
  }, [loadLive]);

  const selectedRun = data?.runs.find((run) => run.id === selectedRunId) ?? null;

  if (loading && !data) return <LoadingState />;
  if (error && !data) return <ErrorState message={error} retry={loadData} />;
  if (!data || !selectedRun) return null;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark" aria-label="Alpha Strategy Monitor">
          <span className="brand-symbol">A</span>
          <div>
            <strong>Alpha</strong>
            <span>Strategy Monitor</span>
          </div>
        </div>

        <nav className="primary-nav" aria-label="主导航">
          <NavButton
            active={activeView === "live"}
            icon={<RadioTower size={18} />}
            label="实时模拟盘"
            onClick={() => setActiveView("live")}
          />
          <NavButton
            active={activeView === "overview"}
            icon={<LayoutDashboard size={18} />}
            label="策略总览"
            onClick={() => setActiveView("overview")}
          />
          <NavButton
            active={activeView === "strategies"}
            icon={<BarChart3 size={18} />}
            label="策略对比"
            onClick={() => setActiveView("strategies")}
          />
          <NavButton
            active={activeView === "trades"}
            icon={<TableProperties size={18} />}
            label="交易明细"
            onClick={() => setActiveView("trades")}
          />
        </nav>

        <div className="sidebar-status">
          <div className="status-line">
            <span className={liveError ? "status-dot error" : "status-dot"} />
            {liveError ? "实时接口异常" : "Dry-run 运行中"}
          </div>
          <span>{liveData?.summary.open_positions ?? 0} 个当前持仓</span>
          <span>{data.archiveCount} 个回测归档</span>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="mobile-brand">
            <span className="brand-symbol">A</span>
            <strong>Alpha Monitor</strong>
          </div>
          {activeView === "live" ? (
            <div className="live-topbar-title">
              <span className="live-pulse" />
              <strong>
                {liveData
                  ? marketDataSourceLabel(liveData.market_data_source)
                  : "Binance Futures"}
              </strong>
              <span>Long only · Dry-run</span>
            </div>
          ) : (
            <div className="run-control">
              <label htmlFor="run-select">当前运行</label>
              <select
                id="run-select"
                value={selectedRun.id}
                onChange={(event) => setSelectedRunId(event.target.value)}
              >
                {data.runs.map((run) => (
                  <option key={run.id} value={run.id}>
                    {runLabel(run)}
                  </option>
                ))}
              </select>
            </div>
          )}
          <div className="topbar-meta">
            <span>
              更新于{" "}
              {formatDateTime(
                activeView === "live" && liveData ? liveData.generated_at : data.generatedAt,
              )}
            </span>
            <button
              className={liveLoading ? "icon-button refreshing" : "icon-button"}
              type="button"
              title="刷新数据"
              aria-label="刷新数据"
              onClick={() => void (activeView === "live" ? loadLive() : loadData())}
            >
              <RefreshCw size={17} />
            </button>
          </div>
        </header>

        {error ? (
          <div className="inline-alert">
            数据刷新失败：{error}
            <button type="button" onClick={() => setError(null)} aria-label="关闭">
              <X size={16} />
            </button>
          </div>
        ) : null}

        {liveError && activeView === "live" ? (
          <div className="inline-alert">
            实时数据刷新失败：{liveError}。界面保留最后一次成功快照。
            <button type="button" onClick={() => setLiveError(null)} aria-label="关闭">
              <X size={16} />
            </button>
          </div>
        ) : null}

        {activeView === "live" ? (
          liveData ? <LiveView data={liveData} /> : <LiveLoadingState />
        ) : null}
        {activeView === "overview" ? <Overview run={selectedRun} /> : null}
        {activeView === "strategies" ? (
          <StrategiesView
            data={data}
            selectedRunId={selectedRun.id}
            compareIds={compareIds}
            onCompareChange={setCompareIds}
            onSelectRun={(id) => {
              setSelectedRunId(id);
              setActiveView("overview");
            }}
          />
        ) : null}
        {activeView === "trades" ? <TradesView run={selectedRun} /> : null}
      </main>
    </div>
  );
}

function NavButton({
  active,
  icon,
  label,
  onClick,
}: {
  active: boolean;
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button className={active ? "nav-button active" : "nav-button"} type="button" onClick={onClick}>
      {icon}
      <span>{label}</span>
    </button>
  );
}

function LiveView({ data }: { data: LiveDashboardData }) {
  const signalHealthy = data.signal.state === "healthy";
  const priceHealthy = data.price_state === "healthy";
  const marketSource = marketDataSourceLabel(data.market_data_source);

  return (
    <div className="page live-page">
      <div className="page-heading live-heading">
        <div>
          <span className="eyebrow">实时模拟盘</span>
          <h1>{breakStrategyName(data.strategy)}</h1>
          <p>
            10 秒自动刷新
            <span>·</span>
            24 小时调仓
            <span>·</span>
            {marketSource}行情
          </p>
        </div>
        <div className="tag-list" aria-label="运行状态">
          <span className={signalHealthy ? "tag live-ok" : "tag live-warn"}>
            <span className="mini-status-dot" />
            信号 {signalHealthy ? "正常" : data.signal.state}
          </span>
          <span className={priceHealthy ? "tag live-ok" : "tag live-warn"}>
            行情 {priceHealthy ? "正常" : "降级"}
          </span>
          <span className="tag">DRY-RUN</span>
          <span className="tag">LONG ONLY</span>
        </div>
      </div>

      <section className="metric-strip" aria-label="实时核心指标">
        <Metric
          label="当前权益"
          value={formatMoney(data.summary.equity)}
          tone="#2f3835"
          icon={<WalletCards size={18} />}
          note={`可用 ${formatMoney(data.summary.available_balance)}`}
        />
        <Metric
          label="总盈亏"
          value={formatPct(data.summary.return_pct, true)}
          tone={valueColor(data.summary.total_pnl)}
          icon={<TrendingUp size={18} />}
          note={formatMoney(data.summary.total_pnl, "USDT", true)}
        />
        <Metric
          label="未实现盈亏"
          value={formatMoney(data.summary.unrealized_pnl, "USDT", true)}
          tone={valueColor(data.summary.unrealized_pnl)}
          icon={<CircleDollarSign size={18} />}
          note={`已实现 ${formatMoney(data.summary.realized_pnl, "USDT", true)}`}
        />
        <Metric
          label="最大回撤"
          value={formatPct(data.summary.max_drawdown_pct)}
          tone={data.summary.max_drawdown_pct > 0 ? NEGATIVE : NEUTRAL}
          icon={<TrendingDown size={18} />}
          note="基于实时权益快照"
        />
        <Metric
          label="资金敞口"
          value={formatPct(data.summary.exposure_pct)}
          tone="#2d5f9a"
          icon={<Gauge size={18} />}
          note={formatMoney(data.summary.exposure)}
        />
        <Metric
          label="当前持仓"
          value={String(data.summary.open_positions)}
          tone="#2f3835"
          icon={<Activity size={18} />}
          note={`${data.summary.closed_trades} 笔已平仓`}
        />
      </section>

      <div className="live-primary-grid">
        <LiveEquityPanel data={data} />

        <section className="panel signal-panel">
          <PanelHeader title="信号生产状态" subtitle="LightGBM 横截面模型" />
          <div className="signal-health">
            <span className={signalHealthy ? "health-icon healthy" : "health-icon warning"}>
              <Server size={19} />
            </span>
            <div>
              <strong>{signalHealthy ? "生产链路正常" : "生产链路需检查"}</strong>
              <span>{data.signal.fetch_error_count ?? 0} 个行情拉取错误</span>
            </div>
          </div>
          <div className="signal-facts">
            <DiagnosticRow
              label="决策时间"
              value={data.signal.decision_time ? formatDateTime(data.signal.decision_time) : "—"}
            />
            <DiagnosticRow
              label="下次调仓"
              value={
                data.signal.next_rebalance_time
                  ? formatDateTime(data.signal.next_rebalance_time)
                  : "—"
              }
            />
            <DiagnosticRow
              label="模型组合"
              value={`${data.signal.model_count ?? "—"} 个模型`}
            />
            <DiagnosticRow
              label="截面覆盖"
              value={`${data.signal.cross_section_count ?? "—"} 个币种`}
            />
            <DiagnosticRow
              label="周期 / Horizon"
              value={`${data.signal.timeframe ?? "—"} / ${data.signal.horizon_bars ?? "—"} bars`}
            />
          </div>
          <div className="selected-pairs">
            <span>本轮入选 {data.signal.selected_count ?? 0}</span>
            <div>
              {data.signal.selected_pairs.map((pair) => (
                <code key={pair}>{shortPair(pair)}</code>
              ))}
            </div>
          </div>
        </section>
      </div>

      <section className="panel">
        <PanelHeader
          title="当前持仓"
          subtitle={`${data.summary.open_positions} 个 Long · 按未实现盈亏排序`}
          action={
            <span className="table-total-pnl" style={{ color: valueColor(data.summary.unrealized_pnl) }}>
              {formatMoney(data.summary.unrealized_pnl, "USDT", true)}
            </span>
          }
        />
        <div className="live-table-wrap">
          <table className="live-table">
            <thead>
              <tr>
                <th>交易对</th>
                <th>模型排名</th>
                <th>开仓价</th>
                <th>最新价</th>
                <th>投入</th>
                <th>浮盈亏</th>
                <th>收益率</th>
                <th>持仓时长</th>
                <th>止损</th>
              </tr>
            </thead>
            <tbody>
              {data.positions.length ? (
                data.positions.map((position) => (
                  <tr key={position.id}>
                    <td>
                      <strong>{shortPair(position.pair)}</strong>
                      <small>Long · {formatNumber(position.leverage)}x</small>
                    </td>
                    <td>
                      <strong>#{position.score_position ?? "—"}</strong>
                      <small>{formatPct(position.model_score_rank_pct)}</small>
                    </td>
                    <td>{formatPrice(position.open_rate)}</td>
                    <td>{formatPrice(position.current_rate)}</td>
                    <td>{formatMoney(position.stake_amount)}</td>
                    <td className={position.pnl_abs >= 0 ? "positive" : "negative"}>
                      {formatMoney(position.pnl_abs, "USDT", true)}
                    </td>
                    <td className={position.pnl_pct >= 0 ? "positive" : "negative"}>
                      {formatPct(position.pnl_pct, true)}
                    </td>
                    <td>{formatDuration(position.duration_minutes)}</td>
                    <td>{formatPrice(position.stop_loss)}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td className="empty-table" colSpan={9}>当前没有持仓，等待下一次调仓信号。</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <div className="live-secondary-grid">
        <section className="panel">
          <PanelHeader title="最近订单" subtitle="最新 30 条 Freqtrade 订单记录" />
          <div className="live-table-wrap compact-live-table">
            <table className="live-table">
              <thead>
                <tr>
                  <th>时间</th>
                  <th>交易对</th>
                  <th>方向</th>
                  <th>状态</th>
                  <th>成交额</th>
                </tr>
              </thead>
              <tbody>
                {data.orders.slice(0, 10).map((order) => (
                  <tr key={order.id}>
                    <td>{order.filled_at ? formatDateTime(order.filled_at) : "—"}</td>
                    <td><strong>{shortPair(order.pair)}</strong></td>
                    <td>{order.side.toUpperCase()}</td>
                    <td><span className="order-status">{order.status}</span></td>
                    <td>{formatMoney(order.cost)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel">
          <PanelHeader title="运行摘要" subtitle="账户与执行统计" />
          <div className="runtime-summary">
            <DiagnosticRow label="钱包余额" value={formatMoney(data.summary.wallet_balance)} />
            <DiagnosticRow label="可用余额" value={formatMoney(data.summary.available_balance)} />
            <DiagnosticRow label="累计手续费" value={formatMoney(data.summary.fees)} />
            <DiagnosticRow
              label="已平仓胜率"
              value={
                data.summary.win_rate_pct === null
                  ? "暂无已平仓交易"
                  : `${formatPct(data.summary.win_rate_pct)} · ${data.summary.wins}胜 ${data.summary.losses}负`
              }
            />
            <DiagnosticRow label="行情源" value={marketSource} />
            <DiagnosticRow label="账户模式" value="Freqtrade Dry-run" />
          </div>
        </section>
      </div>
    </div>
  );
}

const LIVE_HISTORY_RANGES: Array<{ value: LiveHistoryRange; label: string }> = [
  { value: "24h", label: "24小时" },
  { value: "7d", label: "7天" },
  { value: "30d", label: "30天" },
  { value: "all", label: "全部" },
];

const LIVE_HISTORY_CACHE_TTL_MS = 30_000;
const liveHistoryCache = new Map<
  string,
  { data: LiveEquityHistoryResponse; cachedAt: number }
>();

function liveHistoryCacheKey(range: LiveHistoryRange, interval: LiveHistoryInterval): string {
  return `${range}:${interval}`;
}

async function requestLiveEquityHistory(
  range: LiveHistoryRange,
  interval: LiveHistoryInterval,
  signal?: AbortSignal,
): Promise<LiveEquityHistoryResponse> {
  const params = new URLSearchParams({ range, interval });
  const response = await fetch(`/api/live/equity?${params}`, {
    cache: "default",
    signal,
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = (await response.json()) as LiveEquityHistoryResponse;
  liveHistoryCache.set(liveHistoryCacheKey(range, interval), {
    data,
    cachedAt: Date.now(),
  });
  if (interval === "auto") {
    liveHistoryCache.set(liveHistoryCacheKey(range, data.interval), {
      data,
      cachedAt: Date.now(),
    });
  }
  return data;
}

function LiveEquityPanelComponent({ data }: { data: LiveDashboardData }) {
  const [range, setRange] = useState<LiveHistoryRange>("all");
  const [interval, setInterval] = useState<LiveHistoryInterval>("auto");
  const [history, setHistory] = useState<LiveEquityHistoryResponse | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [zoom, setZoom] = useState<{ startIndex: number; endIndex: number } | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const key = liveHistoryCacheKey(range, interval);
    const cached = liveHistoryCache.get(key);
    if (cached) {
      setHistory(cached.data);
      setHistoryError(null);
      setHistoryLoading(false);
    } else {
      setHistory(null);
      setHistoryLoading(true);
    }

    async function loadHistory(force = false) {
      const current = liveHistoryCache.get(key);
      if (!force && current && Date.now() - current.cachedAt < LIVE_HISTORY_CACHE_TTL_MS) {
        return;
      }
      if (!current) setHistoryLoading(true);
      try {
        const nextHistory = await requestLiveEquityHistory(range, interval, controller.signal);
        setHistory(nextHistory);
        setHistoryError(null);
      } catch (caught) {
        if (controller.signal.aborted) return;
        setHistoryError(caught instanceof Error ? caught.message : "无法读取权益历史");
      } finally {
        if (!controller.signal.aborted) setHistoryLoading(false);
      }
    }

    void loadHistory();
    const timer = window.setInterval(() => void loadHistory(true), 60_000);
    const prefetchTimer = window.setTimeout(() => {
      if (interval !== "auto") return;
      const prefetchTargets: Array<[LiveHistoryRange, LiveHistoryInterval]> = [
        ...LIVE_HISTORY_RANGES
          .filter((option) => option.value !== range)
          .map((option) => [option.value, "auto"] as [LiveHistoryRange, LiveHistoryInterval]),
        [range, "15m"],
        [range, "1h"],
        [range, "4h"],
        [range, "1d"],
      ];
      for (const [targetRange, targetInterval] of prefetchTargets) {
        const prefetchKey = liveHistoryCacheKey(targetRange, targetInterval);
        const prefetched = liveHistoryCache.get(prefetchKey);
        if (prefetched && Date.now() - prefetched.cachedAt < LIVE_HISTORY_CACHE_TTL_MS) continue;
        void requestLiveEquityHistory(targetRange, targetInterval).catch(() => undefined);
      }
    }, 400);
    return () => {
      controller.abort();
      window.clearInterval(timer);
      window.clearTimeout(prefetchTimer);
    };
  }, [range, interval]);

  const points = history?.points ?? data.equity_history;
  const lastIndex = Math.max(0, points.length - 1);
  const startIndex = Math.min(zoom?.startIndex ?? 0, lastIndex);
  const endIndex = Math.min(zoom?.endIndex ?? lastIndex, lastIndex);
  const isZoomed = startIndex > 0 || endIndex < lastIndex;
  const intervalLabel = liveHistoryIntervalLabel(history?.interval ?? interval);
  const chartDataKey = points.length
    ? `${points.length}:${points[0].timestamp}:${points[lastIndex].timestamp}`
    : "empty";
  const subtitle = history
    ? `${history.total_points.toLocaleString("zh-CN")} 个完整快照 · ${intervalLabel}显示 ${history.returned_points.toLocaleString("zh-CN")} 个点`
    : `${points.length.toLocaleString("zh-CN")} 个权益快照`;

  function changeRange(nextRange: LiveHistoryRange) {
    setRange(nextRange);
    setZoom(null);
  }

  function changeInterval(nextInterval: LiveHistoryInterval) {
    setInterval(nextInterval);
    setZoom(null);
  }

  function zoomBy(scale: number) {
    if (points.length < 2) return;
    const currentWidth = endIndex - startIndex + 1;
    const minimumWidth = Math.min(12, points.length);
    const nextWidth = Math.max(
      minimumWidth,
      Math.min(points.length, Math.round(currentWidth * scale)),
    );
    const center = (startIndex + endIndex) / 2;
    let nextStart = Math.max(0, Math.round(center - (nextWidth - 1) / 2));
    let nextEnd = Math.min(lastIndex, nextStart + nextWidth - 1);
    nextStart = Math.max(0, nextEnd - nextWidth + 1);
    setZoom({ startIndex: nextStart, endIndex: nextEnd });
  }

  return (
    <section className="panel chart-panel live-equity-panel">
      <PanelHeader
        title="实时权益曲线"
        subtitle={subtitle}
        action={
          <div className="equity-history-controls">
            <div className="segmented history-range" aria-label="权益时间范围">
              {LIVE_HISTORY_RANGES.map((option) => (
                <button
                  className={range === option.value ? "active" : ""}
                  key={option.value}
                  type="button"
                  onClick={() => changeRange(option.value)}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <label className="history-interval-select">
              <span>粒度</span>
              <select
                aria-label="权益数据粒度"
                value={interval}
                onChange={(event) => changeInterval(event.target.value as LiveHistoryInterval)}
              >
                <option value="auto">自动</option>
                <option value="raw">原始分钟</option>
                <option value="15m">15分钟</option>
                <option value="1h">1小时</option>
                <option value="4h">4小时</option>
                <option value="1d">1天</option>
              </select>
            </label>
            <div className="chart-zoom-controls" aria-label="权益曲线缩放">
              <button
                type="button"
                title="放大曲线"
                aria-label="放大曲线"
                disabled={points.length < 2 || endIndex - startIndex < 12}
                onClick={() => zoomBy(0.5)}
              >
                <ZoomIn size={15} />
              </button>
              <button
                type="button"
                title="缩小曲线"
                aria-label="缩小曲线"
                disabled={!isZoomed}
                onClick={() => zoomBy(2)}
              >
                <ZoomOut size={15} />
              </button>
              <button
                type="button"
                title="复位曲线"
                aria-label="复位曲线"
                disabled={!isZoomed}
                onClick={() => setZoom(null)}
              >
                <RotateCcw size={14} />
              </button>
            </div>
          </div>
        }
      />
      {historyError ? (
        <div className="chart-data-warning">全量历史读取失败，当前显示最近快照：{historyError}</div>
      ) : null}
      <div className={historyLoading && !history ? "live-equity-chart loading" : "live-equity-chart"}>
        {points.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              key={chartDataKey}
              data={points}
              margin={{ top: 12, right: 12, bottom: 8, left: 4 }}
              throttleDelay={32}
            >
              <CartesianGrid stroke="#e4e8e6" vertical={false} />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(value: string) => formatLiveHistoryTick(value, range)}
                tick={{ fontSize: 10, fill: "#77817d" }}
                axisLine={false}
                tickLine={false}
                minTickGap={36}
              />
              <YAxis
                tickFormatter={(value: number) => `${(value / 1000).toFixed(2)}k`}
                tick={{ fontSize: 10, fill: "#77817d" }}
                axisLine={false}
                tickLine={false}
                width={48}
                domain={["auto", "auto"]}
              />
              <Tooltip content={<LiveEquityTooltip />} isAnimationActive={false} />
              <ReferenceLine
                y={data.summary.starting_balance}
                stroke="#aeb6b2"
                strokeDasharray="4 4"
              />
              <Line
                type="monotone"
                dataKey="equity"
                stroke={POSITIVE}
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 4, fill: POSITIVE, stroke: "#fff", strokeWidth: 2 }}
                isAnimationActive={false}
              />
              <Brush
                dataKey="timestamp"
                startIndex={startIndex}
                endIndex={endIndex}
                height={25}
                travellerWidth={8}
                tickFormatter={(value: string) => formatLiveHistoryTick(value, range)}
                stroke="#8da69d"
                fill="#f6f8f7"
                onChange={(next) => {
                  if (typeof next.startIndex !== "number" || typeof next.endIndex !== "number") return;
                  setZoom({ startIndex: next.startIndex, endIndex: next.endIndex });
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <EmptyChart label="暂无权益历史" />
        )}
      </div>
      <div className="chart-legend-row">
        <span><i className="legend-line" />实时估算权益</span>
        <span><i className="legend-dashed" />初始资金</span>
        <span className="zoom-hint">拖动底部滑块或两侧把手查看局部</span>
        <strong>手续费累计 {formatMoney(data.summary.fees)}</strong>
      </div>
    </section>
  );
}

const LiveEquityPanel = memo(
  LiveEquityPanelComponent,
  (previous, next) =>
    previous.data.summary.starting_balance === next.data.summary.starting_balance
    && previous.data.summary.fees === next.data.summary.fees,
);

function liveHistoryIntervalLabel(interval: LiveHistoryInterval): string {
  const labels: Record<LiveHistoryInterval, string> = {
    auto: "自动聚合",
    raw: "原始分钟",
    "15m": "15分钟",
    "1h": "1小时",
    "4h": "4小时",
    "1d": "1天",
  };
  return labels[interval];
}

function formatLiveHistoryTick(value: string, range: LiveHistoryRange): string {
  const date = new Date(value);
  if (range === "24h") {
    return new Intl.DateTimeFormat("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(date);
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: range === "7d" ? "2-digit" : undefined,
    hour12: false,
  }).format(date);
}

function marketDataSourceLabel(source: string): string {
  if (source === "binance_futures") return "Binance Futures 主网";
  if (source === "binance_futures_testnet") return "Binance Futures Testnet";
  return source || "Binance Futures";
}

function LiveEquityTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: LiveEquityPoint }>;
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <strong>{formatDateTime(point.timestamp)}</strong>
      <span>{formatMoney(point.equity)}</span>
      <small>总盈亏 {formatMoney(point.total_pnl, "USDT", true)}</small>
      <small>浮盈亏 {formatMoney(point.unrealized_pnl, "USDT", true)}</small>
    </div>
  );
}

function LiveLoadingState() {
  return (
    <div className="live-loading">
      <RefreshCw size={20} />
      <strong>正在连接 dry-run 实时接口</strong>
      <span>首次读取可能需要几秒钟</span>
    </div>
  );
}

function shortPair(pair: string): string {
  return pair.replace("/USDT:USDT", "");
}

function breakStrategyName(name: string) {
  return name.split(/(?=[A-Z])/).map((part, index) => (
    <span className="strategy-word" key={`${part}-${index}`}>
      {part}
      <wbr />
    </span>
  ));
}

function formatPrice(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  const digits = value >= 1000 ? 2 : value >= 1 ? 4 : 6;
  return value.toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function Overview({ run }: { run: StrategyRun }) {
  const [equityMode, setEquityMode] = useState<EquityMode>("balance");
  const sortedPairs = useMemo(
    () => [...run.pairPerformance].sort((a, b) => (b.profitAbs ?? 0) - (a.profitAbs ?? 0)),
    [run],
  );
  const pairExtremes = useMemo(() => {
    const winners = sortedPairs.slice(0, 6);
    const losers = sortedPairs.slice(Math.max(6, sortedPairs.length - 6));
    return [...winners, ...losers].sort((a, b) => (a.profitAbs ?? 0) - (b.profitAbs ?? 0));
  }, [sortedPairs]);

  return (
    <div className="page">
      <PageHeading run={run} eyebrow="策略总览" />

      <section className="metric-strip" aria-label="核心指标">
        <Metric
          label="总收益"
          value={formatPct(run.summary.returnPct, true)}
          tone={valueColor(run.summary.returnPct)}
          icon={<TrendingUp size={18} />}
          note={formatMoney(run.summary.profitAbs, run.execution.stakeCurrency, true)}
        />
        <Metric
          label="最大回撤"
          value={formatPct(run.summary.maxDrawdownPct)}
          tone={NEGATIVE}
          icon={<TrendingDown size={18} />}
          note={formatMoney(run.summary.maxDrawdownAbs, run.execution.stakeCurrency)}
        />
        <Metric
          label="Profit factor"
          value={formatNumber(run.summary.profitFactor)}
          tone={(run.summary.profitFactor ?? 0) >= 1 ? POSITIVE : NEGATIVE}
          icon={<Gauge size={18} />}
          note={`期望 ${formatMoney(run.summary.expectancy, run.execution.stakeCurrency)}`}
        />
        <Metric
          label="胜率"
          value={formatPct(run.summary.winRatePct)}
          tone="#2d5f9a"
          icon={<Activity size={18} />}
          note={`${run.summary.wins} 胜 / ${run.summary.losses} 负`}
        />
        <Metric
          label="交易数"
          value={run.summary.trades.toLocaleString("zh-CN")}
          tone="#2f3835"
          icon={<ListFilter size={18} />}
          note={`${formatNumber(run.summary.tradesPerDay)} 笔/日`}
        />
        <Metric
          label="最终资金"
          value={formatMoney(run.summary.finalBalance, run.execution.stakeCurrency)}
          tone="#2f3835"
          icon={<WalletCards size={18} />}
          note={`起始 ${formatMoney(run.summary.startingBalance, run.execution.stakeCurrency)}`}
        />
      </section>

      <section className="panel chart-panel">
        <PanelHeader
          title="资金曲线与回撤"
          subtitle={`${run.period.days} 天 · 日频钱包余额`}
          action={
            <div className="segmented" aria-label="资金曲线指标">
              <button
                className={equityMode === "balance" ? "active" : ""}
                type="button"
                onClick={() => setEquityMode("balance")}
              >
                余额
              </button>
              <button
                className={equityMode === "returnPct" ? "active" : ""}
                type="button"
                onClick={() => setEquityMode("returnPct")}
              >
                累计收益
              </button>
            </div>
          }
        />
        <div className="equity-chart">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={run.equity} margin={{ top: 12, right: 10, bottom: 0, left: 4 }}>
              <CartesianGrid stroke="#e4e8e6" vertical={false} />
              <XAxis
                dataKey="date"
                tickFormatter={(value: string) => value.slice(5)}
                tick={{ fontSize: 11, fill: "#77817d" }}
                axisLine={false}
                tickLine={false}
                minTickGap={28}
              />
              <YAxis
                yAxisId="primary"
                tickFormatter={(value: number) =>
                  equityMode === "balance" ? `${Math.round(value / 1000)}k` : `${value}%`
                }
                tick={{ fontSize: 11, fill: "#77817d" }}
                axisLine={false}
                tickLine={false}
                width={44}
                domain={["auto", "auto"]}
              />
              <YAxis
                yAxisId="drawdown"
                orientation="right"
                domain={["dataMin", 0]}
                hide
              />
              <Tooltip content={<EquityTooltip mode={equityMode} currency={run.execution.stakeCurrency} />} />
              <ReferenceLine
                yAxisId="primary"
                y={equityMode === "balance" ? run.summary.startingBalance : 0}
                stroke="#aeb6b2"
                strokeDasharray="4 4"
              />
              <Bar
                yAxisId="drawdown"
                dataKey="drawdownPct"
                fill="#d8a29b"
                opacity={0.35}
                isAnimationActive={false}
              />
              <Line
                yAxisId="primary"
                type="monotone"
                dataKey={equityMode}
                stroke={POSITIVE}
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 4, fill: POSITIVE, stroke: "#fff", strokeWidth: 2 }}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-legend-row">
          <span><i className="legend-line" />{equityMode === "balance" ? "钱包余额" : "累计收益"}</span>
          <span><i className="legend-block" />回撤区间</span>
          <strong>钱包 Sharpe {formatNumber(run.summary.sharpe)}</strong>
        </div>
      </section>

      <div className="analysis-grid">
        <section className="panel">
          <PanelHeader title="多空贡献" subtitle="按初始资金计算的收益贡献" />
          <DirectionComparison run={run} />
        </section>

        <section className="panel">
          <PanelHeader title="退出结构" subtitle="收益与风险从哪里产生" />
          <ExitReasonChart rows={run.exitReasons} />
        </section>

        <section className="panel diagnostics">
          <PanelHeader title="运行诊断" subtitle="执行假设与统计可靠性" />
          <DiagnosticRow label="市场同期涨幅" value={formatPct(run.summary.marketChangePct, true)} />
          <DiagnosticRow label="平均单笔收益" value={formatPct(run.summary.avgProfitPct, true)} />
          <DiagnosticRow label="单笔均值 p-value" value={formatNumber(run.summary.pValue, 4)} warning />
          <DiagnosticRow label="止损 / 手续费" value={`${formatPct(run.execution.stoplossPct)} / ${formatPct(run.execution.feePct)}`} />
          <DiagnosticRow label="杠杆 / 最大持仓" value={`${formatNumber(run.execution.leverage)}x / ${run.execution.maxOpenTrades}`} />
        </section>
      </div>

      <section className="panel">
        <PanelHeader title="币种收益贡献" subtitle={`覆盖 ${run.pairPerformance.length} 个交易对`} />
        <div className="pair-layout">
          <div className="pair-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={pairExtremes}
                layout="vertical"
                margin={{ top: 4, right: 18, bottom: 4, left: 8 }}
              >
                <CartesianGrid stroke="#e4e8e6" horizontal={false} />
                <XAxis
                  type="number"
                  tick={{ fontSize: 10, fill: "#77817d" }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value: number) => `${Math.round(value)}`}
                />
                <YAxis
                  type="category"
                  dataKey="pair"
                  tick={{ fontSize: 10, fill: "#4e5955" }}
                  tickLine={false}
                  axisLine={false}
                  width={116}
                  tickFormatter={(value: string) => value.replace("/USDT:USDT", "")}
                />
                <Tooltip content={<ProfitTooltip currency={run.execution.stakeCurrency} labelKey="pair" />} />
                <ReferenceLine x={0} stroke="#aeb6b2" />
                <Bar dataKey="profitAbs" radius={[0, 2, 2, 0]} isAnimationActive={false}>
                  {pairExtremes.map((row) => (
                    <Cell key={row.pair} fill={(row.profitAbs ?? 0) >= 0 ? POSITIVE : NEGATIVE} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <PairTable rows={sortedPairs.slice(0, 8)} currency={run.execution.stakeCurrency} />
        </div>
      </section>

      <section className="panel">
        <PanelHeader title="周期表现" subtitle="月度与星期分布" />
        <div className="period-grid">
          <PeriodTable title="月度" rows={run.monthly} currency={run.execution.stakeCurrency} />
          <PeriodTable title="星期" rows={run.weekdays} currency={run.execution.stakeCurrency} />
        </div>
      </section>
    </div>
  );
}

function PageHeading({ run, eyebrow }: { run: StrategyRun; eyebrow: string }) {
  return (
    <div className="page-heading">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{run.strategy}</h1>
        <p>
          {formatDate(run.period.start)} 至 {formatDate(run.period.end)}
          <span>·</span>
          {run.archive}
        </p>
      </div>
      <div className="tag-list" aria-label="运行属性">
        {run.isLatest ? <span className="tag latest"><Check size={13} />最新</span> : null}
        <span className="tag">{run.period.timeframe}</span>
        <span className="tag">{run.execution.tradingMode}</span>
        <span className="tag">{run.execution.marginMode}</span>
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  tone,
  icon,
  note,
}: {
  label: string;
  value: string;
  tone: string;
  icon: React.ReactNode;
  note: string;
}) {
  return (
    <article className="metric">
      <div className="metric-label">
        <span style={{ color: tone }}>{icon}</span>
        {label}
      </div>
      <strong style={{ color: tone }}>{value}</strong>
      <small>{note}</small>
    </article>
  );
}

function PanelHeader({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle: string;
  action?: React.ReactNode;
}) {
  return (
    <header className="panel-header">
      <div>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
      {action}
    </header>
  );
}

function EquityTooltip({
  active,
  payload,
  label,
  mode,
  currency,
}: {
  active?: boolean;
  payload?: Array<{ payload: EquityPoint }>;
  label?: string;
  mode: EquityMode;
  currency: string;
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <strong>{label}</strong>
      <span>{mode === "balance" ? formatMoney(point.balance, currency) : formatPct(point.returnPct, true)}</span>
      <small>当日 {formatMoney(point.profitAbs, currency, true)}</small>
      <small>回撤 {formatPct(Math.abs(point.drawdownPct))}</small>
    </div>
  );
}

function ProfitTooltip({
  active,
  payload,
  currency,
  labelKey,
}: {
  active?: boolean;
  payload?: Array<{ payload: SummaryRow }>;
  currency: string;
  labelKey: "pair" | "reason";
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <strong>{row[labelKey]}</strong>
      <span>{formatMoney(row.profitAbs, currency, true)}</span>
      <small>{row.trades} 笔 · 胜率 {formatPct(row.winRatePct)}</small>
    </div>
  );
}

function DirectionComparison({ run }: { run: StrategyRun }) {
  const rows = [
    { side: "Long", ...run.directions.long, color: POSITIVE, icon: <ArrowUpRight size={17} /> },
    { side: "Short", ...run.directions.short, color: "#2d5f9a", icon: <ArrowDownRight size={17} /> },
  ];
  const maxValue = Math.max(...rows.map((row) => Math.abs(row.profitAbs ?? 0)), 1);
  return (
    <div className="direction-list">
      {rows.map((row) => (
        <div className="direction-row" key={row.side}>
          <div className="direction-title" style={{ color: row.color }}>
            {row.icon}
            <strong>{row.side}</strong>
            <span>{row.trades} 笔</span>
          </div>
          <div className="direction-value">
            <strong style={{ color: valueColor(row.profitAbs) }}>
              {formatMoney(row.profitAbs, run.execution.stakeCurrency, true)}
            </strong>
            <span>{formatPct(row.returnPct, true)}</span>
          </div>
          <div className="progress-track">
            <span
              style={{
                width: `${Math.abs(row.profitAbs ?? 0) / maxValue * 100}%`,
                background: valueColor(row.profitAbs),
              }}
            />
          </div>
        </div>
      ))}
      <div className="direction-net">
        <span>净贡献</span>
        <strong>{formatMoney(run.summary.profitAbs, run.execution.stakeCurrency, true)}</strong>
      </div>
    </div>
  );
}

function ExitReasonChart({ rows }: { rows: SummaryRow[] }) {
  return (
    <div className="exit-chart">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 8, right: 4, bottom: 18, left: 0 }}>
          <CartesianGrid stroke="#e4e8e6" vertical={false} />
          <XAxis
            dataKey="reason"
            tick={{ fontSize: 10, fill: "#77817d" }}
            axisLine={false}
            tickLine={false}
            interval={0}
            tickFormatter={(value: string) => value.replace("left_", "").replace("_quantile", "")}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#77817d" }}
            axisLine={false}
            tickLine={false}
            width={36}
            tickFormatter={(value: number) => `${Math.round(value / 1000)}k`}
          />
          <Tooltip content={<ProfitTooltip currency="USDT" labelKey="reason" />} />
          <ReferenceLine y={0} stroke="#aeb6b2" />
          <Bar dataKey="profitAbs" radius={[2, 2, 0, 0]} isAnimationActive={false}>
            {rows.map((row) => (
              <Cell key={row.reason} fill={(row.profitAbs ?? 0) >= 0 ? POSITIVE : NEGATIVE} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function DiagnosticRow({
  label,
  value,
  warning = false,
}: {
  label: string;
  value: string;
  warning?: boolean;
}) {
  return (
    <div className="diagnostic-row">
      <span>{label}</span>
      <strong className={warning ? "warning-value" : ""}>{value}</strong>
    </div>
  );
}

function PairTable({ rows, currency }: { rows: SummaryRow[]; currency: string }) {
  return (
    <div className="compact-table-wrap">
      <table className="compact-table">
        <thead>
          <tr>
            <th>交易对</th>
            <th>交易</th>
            <th>胜率</th>
            <th>贡献</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.pair}>
              <td><strong>{row.pair?.replace("/USDT:USDT", "")}</strong></td>
              <td>{row.trades}</td>
              <td>{formatPct(row.winRatePct)}</td>
              <td className={(row.profitAbs ?? 0) >= 0 ? "positive" : "negative"}>
                {formatMoney(row.profitAbs, currency, true)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PeriodTable({
  title,
  rows,
  currency,
}: {
  title: string;
  rows: StrategyRun["monthly"];
  currency: string;
}) {
  return (
    <div className="period-table">
      <h3>{title}</h3>
      {rows.map((row) => (
        <div className="period-row" key={`${title}-${row.label}`}>
          <span>{row.label}</span>
          <small>{row.trades} 笔</small>
          <strong className={(row.profitAbs ?? 0) >= 0 ? "positive" : "negative"}>
            {formatMoney(row.profitAbs, currency, true)}
          </strong>
        </div>
      ))}
    </div>
  );
}

function StrategiesView({
  data,
  selectedRunId,
  compareIds,
  onCompareChange,
  onSelectRun,
}: {
  data: DashboardData;
  selectedRunId: string;
  compareIds: string[];
  onCompareChange: (ids: string[]) => void;
  onSelectRun: (id: string) => void;
}) {
  const comparisonRuns = data.runs.filter((run) => compareIds.includes(run.id));
  const comparisonData = useMemo(() => mergeEquity(comparisonRuns), [comparisonRuns]);
  const scatterData = data.runs.map((run) => ({
    id: run.id,
    strategy: run.strategy,
    run: shortRunId(run),
    returnPct: run.summary.returnPct ?? 0,
    drawdownPct: run.summary.maxDrawdownPct ?? 0,
    trades: run.summary.trades,
    profitFactor: run.summary.profitFactor ?? 0,
  }));

  function toggleCompare(id: string) {
    if (compareIds.includes(id)) {
      onCompareChange(compareIds.filter((value) => value !== id));
      return;
    }
    onCompareChange([...compareIds.slice(-4), id]);
  }

  return (
    <div className="page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">策略对比</span>
          <h1>运行表现矩阵</h1>
          <p>{data.strategyCount} 个策略 · {data.runCount} 次回测</p>
        </div>
        <div className="tag-list">
          <span className="tag">{compareIds.length} 个运行参与对比</span>
        </div>
      </div>

      <div className="comparison-grid">
        <section className="panel">
          <PanelHeader title="收益与回撤" subtitle="点大小代表交易数量" />
          <div className="scatter-chart">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 14, right: 16, bottom: 18, left: 4 }}>
                <CartesianGrid stroke="#e4e8e6" />
                <XAxis
                  type="number"
                  dataKey="drawdownPct"
                  name="最大回撤"
                  unit="%"
                  tick={{ fontSize: 10, fill: "#77817d" }}
                  axisLine={false}
                  tickLine={false}
                  label={{ value: "最大回撤 %", position: "insideBottom", offset: -10, fill: "#77817d", fontSize: 11 }}
                />
                <YAxis
                  type="number"
                  dataKey="returnPct"
                  name="总收益"
                  unit="%"
                  tick={{ fontSize: 10, fill: "#77817d" }}
                  axisLine={false}
                  tickLine={false}
                  width={42}
                />
                <ZAxis type="number" dataKey="trades" range={[80, 360]} />
                <ReferenceLine y={0} stroke="#aeb6b2" />
                <Tooltip content={<ScatterTooltip />} />
                <Scatter data={scatterData} fill={POSITIVE} isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="panel">
          <PanelHeader title="归一化收益曲线" subtitle="按各自初始资金计算" />
          <div className="compare-line-chart">
            {comparisonRuns.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={comparisonData} margin={{ top: 14, right: 12, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="#e4e8e6" vertical={false} />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(value: string) => value.slice(5)}
                    tick={{ fontSize: 10, fill: "#77817d" }}
                    axisLine={false}
                    tickLine={false}
                    minTickGap={30}
                  />
                  <YAxis
                    tickFormatter={(value: number) => `${value}%`}
                    tick={{ fontSize: 10, fill: "#77817d" }}
                    axisLine={false}
                    tickLine={false}
                    width={38}
                    domain={["auto", "auto"]}
                  />
                  <Tooltip content={<CompareTooltip runs={comparisonRuns} />} />
                  <Legend
                    formatter={(value: string) =>
                      shortRunId(comparisonRuns.find((run) => run.id === value) ?? comparisonRuns[0])
                    }
                    wrapperStyle={{ fontSize: 10 }}
                  />
                  <ReferenceLine y={0} stroke="#aeb6b2" strokeDasharray="4 4" />
                  {comparisonRuns.map((run, index) => (
                    <Line
                      key={run.id}
                      type="monotone"
                      dataKey={run.id}
                      stroke={RUN_COLORS[index % RUN_COLORS.length]}
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <EmptyChart label="尚未选择运行" />
            )}
          </div>
        </section>
      </div>

      <section className="panel">
        <PanelHeader title="全部策略运行" subtitle="选择最多 5 个运行叠加比较" />
        <div className="strategy-table-wrap">
          <table className="strategy-table">
            <thead>
              <tr>
                <th className="check-column">对比</th>
                <th>策略 / 运行</th>
                <th>区间</th>
                <th>收益</th>
                <th>回撤</th>
                <th>Profit factor</th>
                <th>Sharpe</th>
                <th>胜率</th>
                <th>交易数</th>
              </tr>
            </thead>
            <tbody>
              {data.runs.map((run) => (
                <tr key={run.id} className={run.id === selectedRunId ? "selected" : ""}>
                  <td className="check-column">
                    <label className="compare-check" title="加入对比">
                      <input
                        type="checkbox"
                        checked={compareIds.includes(run.id)}
                        onChange={() => toggleCompare(run.id)}
                      />
                      <span>{compareIds.includes(run.id) ? <Check size={13} /> : null}</span>
                    </label>
                  </td>
                  <td>
                    <button className="run-link" type="button" onClick={() => onSelectRun(run.id)}>
                      <strong>{run.strategy}</strong>
                      <span>{shortRunId(run)}</span>
                    </button>
                  </td>
                  <td>{run.period.timeframe} · {run.period.days} 天</td>
                  <td className={(run.summary.returnPct ?? 0) >= 0 ? "positive" : "negative"}>
                    {formatPct(run.summary.returnPct, true)}
                  </td>
                  <td className="negative">{formatPct(run.summary.maxDrawdownPct)}</td>
                  <td>{formatNumber(run.summary.profitFactor)}</td>
                  <td>{formatNumber(run.summary.sharpe)}</td>
                  <td>{formatPct(run.summary.winRatePct)}</td>
                  <td>{run.summary.trades.toLocaleString("zh-CN")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function mergeEquity(runs: StrategyRun[]): Array<Record<string, string | number | null>> {
  const byDate = new Map<string, Record<string, string | number | null>>();
  runs.forEach((run) => {
    run.equity.forEach((point) => {
      const row = byDate.get(point.date) ?? { date: point.date };
      row[run.id] = point.returnPct;
      byDate.set(point.date, row);
    });
  });
  return [...byDate.values()].sort((a, b) => String(a.date).localeCompare(String(b.date)));
}

function ScatterTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: { strategy: string; run: string; returnPct: number; drawdownPct: number; trades: number; profitFactor: number } }>;
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <strong>{point.strategy}</strong>
      <small>{point.run}</small>
      <span>收益 {formatPct(point.returnPct, true)}</span>
      <small>回撤 {formatPct(point.drawdownPct)} · PF {formatNumber(point.profitFactor)}</small>
      <small>{point.trades} 笔交易</small>
    </div>
  );
}

function CompareTooltip({
  active,
  payload,
  label,
  runs,
}: {
  active?: boolean;
  payload?: Array<{ dataKey: string; value: number; color: string }>;
  label?: string;
  runs: StrategyRun[];
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip compare-tooltip">
      <strong>{label}</strong>
      {payload.map((item) => {
        const run = runs.find((candidate) => candidate.id === item.dataKey);
        return (
          <span key={item.dataKey} style={{ color: item.color }}>
            {run ? shortRunId(run) : item.dataKey}: {formatPct(item.value, true)}
          </span>
        );
      })}
    </div>
  );
}

function EmptyChart({ label }: { label: string }) {
  return (
    <div className="empty-chart">
      <BarChart3 size={22} />
      <span>{label}</span>
    </div>
  );
}

function TradesView({ run }: { run: StrategyRun }) {
  const [search, setSearch] = useState("");
  const [side, setSide] = useState<"all" | "long" | "short">("all");
  const [result, setResult] = useState<"all" | "win" | "loss">("all");
  const [exitReason, setExitReason] = useState("all");
  const [page, setPage] = useState(1);
  const pageSize = 50;
  const exitReasons = useMemo(
    () => [...new Set(run.trades.map((trade) => trade.exitReason))].sort(),
    [run],
  );

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return run.trades
      .filter((trade) => !needle || trade.pair.toLowerCase().includes(needle))
      .filter((trade) => side === "all" || trade.side === side)
      .filter((trade) => exitReason === "all" || trade.exitReason === exitReason)
      .filter((trade) => {
        if (result === "win") return (trade.profitAbs ?? 0) > 0;
        if (result === "loss") return (trade.profitAbs ?? 0) < 0;
        return true;
      })
      .sort((a, b) => b.closeDate.localeCompare(a.closeDate));
  }, [run, search, side, result, exitReason]);

  useEffect(() => setPage(1), [search, side, result, exitReason, run.id]);

  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const visible = filtered.slice((page - 1) * pageSize, page * pageSize);
  const filteredPnl = filtered.reduce((sum, trade) => sum + (trade.profitAbs ?? 0), 0);
  const filteredWins = filtered.filter((trade) => (trade.profitAbs ?? 0) > 0).length;

  function downloadCsv() {
    const columns: Array<keyof Trade> = [
      "pair",
      "side",
      "openDate",
      "closeDate",
      "durationMinutes",
      "profitPct",
      "profitAbs",
      "entryTag",
      "exitReason",
    ];
    const rows = [
      columns.join(","),
      ...filtered.map((trade) =>
        columns.map((column) => JSON.stringify(trade[column] ?? "")).join(","),
      ),
    ];
    const url = URL.createObjectURL(new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `${run.strategy}-${shortRunId(run)}-trades.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="page">
      <PageHeading run={run} eyebrow="交易明细" />

      <section className="trade-summary-band">
        <div><span>筛选交易</span><strong>{filtered.length}</strong></div>
        <div><span>筛选盈亏</span><strong className={filteredPnl >= 0 ? "positive" : "negative"}>{formatMoney(filteredPnl, run.execution.stakeCurrency, true)}</strong></div>
        <div><span>筛选胜率</span><strong>{formatPct(filtered.length ? filteredWins / filtered.length * 100 : null)}</strong></div>
        <div><span>平均持仓</span><strong>{formatBacktestDuration(run.summary.avgHolding)}</strong></div>
      </section>

      <section className="panel">
        <div className="filterbar">
          <label className="search-field">
            <Search size={16} />
            <input
              type="search"
              placeholder="搜索交易对"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </label>
          <div className="segmented">
            {(["all", "long", "short"] as const).map((value) => (
              <button
                key={value}
                className={side === value ? "active" : ""}
                type="button"
                onClick={() => setSide(value)}
              >
                {value === "all" ? "全部方向" : value === "long" ? "Long" : "Short"}
              </button>
            ))}
          </div>
          <div className="segmented">
            {(["all", "win", "loss"] as const).map((value) => (
              <button
                key={value}
                className={result === value ? "active" : ""}
                type="button"
                onClick={() => setResult(value)}
              >
                {value === "all" ? "全部结果" : value === "win" ? "盈利" : "亏损"}
              </button>
            ))}
          </div>
          <label className="select-field">
            <SlidersHorizontal size={15} />
            <select value={exitReason} onChange={(event) => setExitReason(event.target.value)}>
              <option value="all">全部退出原因</option>
              {exitReasons.map((reason) => <option key={reason} value={reason}>{reason}</option>)}
            </select>
          </label>
          <button className="secondary-button" type="button" onClick={downloadCsv}>
            <Download size={16} />
            导出 CSV
          </button>
        </div>

        <div className="trades-table-wrap">
          <table className="trades-table">
            <thead>
              <tr>
                <th>交易对</th>
                <th>方向</th>
                <th>开仓</th>
                <th>平仓</th>
                <th>持仓</th>
                <th>收益率</th>
                <th>盈亏</th>
                <th>仓位</th>
                <th>退出原因</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((trade) => (
                <tr key={`${trade.id}-${trade.pair}-${trade.closeDate}`}>
                  <td><strong>{trade.pair.replace("/USDT:USDT", "")}</strong><small>USDT Perp</small></td>
                  <td><span className={`side-badge ${trade.side}`}>{trade.side}</span></td>
                  <td>{formatDateTime(trade.openDate)}</td>
                  <td>{formatDateTime(trade.closeDate)}</td>
                  <td>{formatDuration(trade.durationMinutes)}</td>
                  <td className={(trade.profitPct ?? 0) >= 0 ? "positive" : "negative"}>{formatPct(trade.profitPct, true)}</td>
                  <td className={(trade.profitAbs ?? 0) >= 0 ? "positive" : "negative"}>{formatMoney(trade.profitAbs, run.execution.stakeCurrency, true)}</td>
                  <td>{formatCompact(trade.stakeAmount)}</td>
                  <td><span className="reason-label">{trade.exitReason}</span></td>
                </tr>
              ))}
              {!visible.length ? (
                <tr><td className="empty-row" colSpan={9}>没有符合条件的交易</td></tr>
              ) : null}
            </tbody>
          </table>
        </div>

        <footer className="pagination">
          <span>第 {page} / {pages} 页</span>
          <div>
            <button
              className="icon-button"
              type="button"
              disabled={page <= 1}
              onClick={() => setPage((value) => Math.max(1, value - 1))}
              aria-label="上一页"
              title="上一页"
            >
              <ChevronLeft size={17} />
            </button>
            <button
              className="icon-button"
              type="button"
              disabled={page >= pages}
              onClick={() => setPage((value) => Math.min(pages, value + 1))}
              aria-label="下一页"
              title="下一页"
            >
              <ChevronRight size={17} />
            </button>
          </div>
        </footer>
      </section>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="state-screen">
      <span className="state-symbol"><RefreshCw size={22} className="spin" /></span>
      <h1>正在汇总策略结果</h1>
      <p>读取 Freqtrade 回测归档</p>
    </div>
  );
}

function ErrorState({ message, retry }: { message: string; retry: () => void }) {
  return (
    <div className="state-screen">
      <span className="state-symbol error"><X size={22} /></span>
      <h1>无法载入策略结果</h1>
      <p>{message}</p>
      <code>uv run alpha-web3-research backtest export-dashboard-data</code>
      <button className="primary-button" type="button" onClick={retry}>
        <RefreshCw size={16} />
        重试
      </button>
    </div>
  );
}

export default App;
