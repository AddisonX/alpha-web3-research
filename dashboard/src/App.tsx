import {Activity, BarChart3, CircleDollarSign, Gauge, RadioTower, RefreshCw, RotateCcw, Server, TrendingDown, TrendingUp, WalletCards, ZoomIn, ZoomOut} from "lucide-react";

import {memo, useEffect, useState} from "react";

import {Brush, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";

import {formatDateTime, formatDuration, formatMoney, formatNumber, formatPct} from "./format";

import type {InstanceCatalog, LiveDashboardData, LiveEquityHistoryResponse, LiveEquityPoint} from "./types";

type LiveHistoryRange = "24h" | "7d" | "30d" | "all";

type LiveHistoryInterval = "auto" | "raw" | "15m" | "1h" | "4h" | "1d";

const POSITIVE = "#126b55";

const NEGATIVE = "#c4473a";

const NEUTRAL = "#77817d";

function valueColor(value: number | null): string {
  if (value === null || value === 0) return NEUTRAL;
  return value > 0 ? POSITIVE : NEGATIVE;
}

async function requestLiveSnapshot(instanceId: string, signal: AbortSignal): Promise<LiveDashboardData> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 12_000);
  try {
    const response = await fetch(`/api/instances/${encodeURIComponent(instanceId)}/live?t=${Date.now()}`, {
      cache: "no-store",
      signal: AbortSignal.any([signal, controller.signal]),
    });
    if (!response.ok) throw new Error(`实时 API 返回 HTTP ${response.status}${response.status === 409 ? "：实例数据库尚未就绪" : ""}`);
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

function LiveView({ data }: { data: LiveDashboardData }) {
  const signalHealthy = data.signal.state === "healthy";
  const priceHealthy = data.price_state === "healthy";
  const marketSource = marketDataSourceLabel(data.market_data_source);

  return (
    <div className="page live-page">
      <div className="page-heading live-heading">
        <div>
          <span className="eyebrow">实时模拟盘</span>
          <h1>{data.instance.display_name}</h1>
          <p>
            10 秒自动刷新
            <span>·</span>
            24 小时调仓
            <span>·</span>
            {marketSource}行情
          </p>
        </div>
        <div className="tag-list" aria-label="运行状态">
          <span className={data.instance.runtime_state === "running" ? "tag live-ok" : "tag live-warn"}>
            {data.instance.runtime_state === "running" ? "运行中" : `运行状态 ${data.instance.runtime_state}`}
          </span>
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
          <PanelHeader title="信号生产状态" subtitle={data.instance.strategy_id} />
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
              label="信号引擎"
              value={data.signal.model_count === 0 ? "规则策略" : `${data.signal.model_count ?? "—"} 个模型`}
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
                <th>信号排序</th>
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

function liveHistoryCacheKey(instanceId: string, range: LiveHistoryRange, interval: LiveHistoryInterval): string {
  return `${instanceId}:${range}:${interval}`;
}

async function requestLiveEquityHistory(
  instanceId: string,
  range: LiveHistoryRange,
  interval: LiveHistoryInterval,
  signal?: AbortSignal,
): Promise<LiveEquityHistoryResponse> {
  const params = new URLSearchParams({ range, interval });
  const response = await fetch(`/api/instances/${encodeURIComponent(instanceId)}/equity?${params}`, {
    cache: "default",
    signal,
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = (await response.json()) as LiveEquityHistoryResponse;
  liveHistoryCache.set(liveHistoryCacheKey(instanceId, range, interval), {
    data,
    cachedAt: Date.now(),
  });
  if (interval === "auto") {
    liveHistoryCache.set(liveHistoryCacheKey(instanceId, range, data.interval), {
      data,
      cachedAt: Date.now(),
    });
  }
  return data;
}

function LiveEquityPanelComponent({ data }: { data: LiveDashboardData }) {
  const instanceId = data.instance.id;
  const [range, setRange] = useState<LiveHistoryRange>("all");
  const [interval, setInterval] = useState<LiveHistoryInterval>("auto");
  const [history, setHistory] = useState<LiveEquityHistoryResponse | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [zoom, setZoom] = useState<{ startIndex: number; endIndex: number } | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const key = liveHistoryCacheKey(instanceId, range, interval);
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
        const nextHistory = await requestLiveEquityHistory(instanceId, range, interval, controller.signal);
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
        const prefetchKey = liveHistoryCacheKey(instanceId, targetRange, targetInterval);
        const prefetched = liveHistoryCache.get(prefetchKey);
        if (prefetched && Date.now() - prefetched.cachedAt < LIVE_HISTORY_CACHE_TTL_MS) continue;
        void requestLiveEquityHistory(instanceId, targetRange, targetInterval, controller.signal).catch(() => undefined);
      }
    }, 400);
    return () => {
      controller.abort();
      window.clearInterval(timer);
      window.clearTimeout(prefetchTimer);
    };
  }, [instanceId, range, interval]);

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
    previous.data.instance.id === next.data.instance.id
    && previous.data.summary.starting_balance === next.data.summary.starting_balance
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

function formatPrice(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  const digits = value >= 1000 ? 2 : value >= 1 ? 4 : 6;
  return value.toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
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

function EmptyChart({ label }: { label: string }) {
  return (
    <div className="empty-chart">
      <BarChart3 size={22} />
      <span>{label}</span>
    </div>
  );
}

function App() {
  const [catalog, setCatalog] = useState<InstanceCatalog | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [selected, setSelected] = useState("");
  const [data, setData] = useState<LiveDashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [refreshNonce, setRefreshNonce] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let busy = false;
    async function loadCatalog() {
      if (busy) return;
      busy = true;
      try {
        const response = await fetch("/api/instances", {
          cache: "no-store",
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(12_000)]),
        });
        if (!response.ok) throw new Error(`实例列表 HTTP ${response.status}`);
        const next = await response.json() as InstanceCatalog;
        if (controller.signal.aborted) return;
        setCatalog(next);
        setCatalogError(null);
        setSelected(previous => {
          if (next.instances.some(instance => instance.id === previous)) return previous;
          const requested = new URLSearchParams(window.location.search).get("instance");
          return next.instances.find(instance => instance.id === requested)?.id ?? next.default_instance ?? "";
        });
      } catch (caught) {
        if (!controller.signal.aborted) setCatalogError(caught instanceof Error ? caught.message : "实例列表不可用");
      } finally {
        busy = false;
      }
    }
    void loadCatalog();
    const timer = window.setInterval(() => void loadCatalog(), 10_000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [refreshNonce]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    let busy = false;
    setError(null);
    const url = new URL(window.location.href);
    url.searchParams.set("instance", selected);
    window.history.replaceState(null, "", url);
    async function loadSnapshot() {
      if (busy) return;
      busy = true;
      setLoading(true);
      try {
        let snapshot: LiveDashboardData | null = null;
        let failure: unknown;
        for (let attempt = 0; attempt < 2 && !controller.signal.aborted; attempt++) {
          try { snapshot = await requestLiveSnapshot(selected, controller.signal); break; }
          catch (caught) { failure = caught; }
        }
        if (controller.signal.aborted) return;
        if (!snapshot) throw failure;
        if (snapshot.instance.id !== selected) throw new Error("实例数据不匹配");
        setData(snapshot);
        setError(null);
      } catch (caught) {
        if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : "实时接口异常");
      } finally {
        busy = false;
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void loadSnapshot();
    const timer = window.setInterval(() => void loadSnapshot(), 10_000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [selected, refreshNonce]);

  const visibleData = data?.instance.id === selected ? data : null;
  const failure = catalogError ?? error;
  const refresh = () => setRefreshNonce(value => value + 1);
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand-mark" aria-label="Alpha Strategy Monitor"><span className="brand-symbol">A</span><div><strong>Alpha</strong><span>Strategy Monitor</span></div></div>
      <nav className="primary-nav" aria-label="主导航"><div className="nav-button active"><RadioTower size={18} /><span>实时模拟盘</span></div></nav>
      <div className="sidebar-status">
        <div className="status-line"><span className={failure ? "status-dot error" : "status-dot"} />{failure ? "实时接口异常" : visibleData ? "监控已连接" : "正在连接"}</div>
        <span>{catalog?.instances.length ?? 0} 个本地实例</span>
        <span>{visibleData?.summary.open_positions ?? 0} 个当前持仓</span>
      </div>
    </aside>
    <main className="main-area">
      <header className="topbar">
        <label className="instance-control"><Server size={16} /><span>本地实例</span>
          <select aria-label="监控实例" value={selected} disabled={!catalog?.instances.length} onChange={event => { setError(null); setSelected(event.target.value); }}>
            {!catalog?.instances.length ? <option value="">{catalog ? "暂无实例" : "加载中"}</option> : null}
            {catalog?.instances.map(instance => <option key={instance.id} value={instance.id}>{instance.display_name}{!instance.database_ready ? " · 未就绪" : instance.runtime_state === "running" ? " · 运行中" : ` · ${instance.runtime_state}`}</option>)}
          </select>
        </label>
        <div className="topbar-meta"><span>{visibleData ? "更新于 " + formatDateTime(visibleData.generated_at) : "等待实例数据"}</span><button type="button" className={loading ? "icon-button refreshing" : "icon-button"} title="刷新数据" aria-label="刷新数据" onClick={refresh}><RefreshCw size={17} /></button></div>
      </header>
      {failure ? <div className="inline-alert" role="alert">实时数据刷新失败：{failure}{visibleData ? "。界面保留本实例最后一次成功快照。" : ""}<button type="button" onClick={refresh} aria-label="重试连接"><RefreshCw size={16} /></button></div> : null}
      {catalog?.errors.length ? <div className="inline-alert" role="status">无法接入的实例：{catalog.errors.map(instance => instance.id).join("、")}</div> : null}
      {visibleData ? <LiveView key={selected} data={visibleData} /> : failure ? <div className="loading-state">实时接口暂不可用</div> : catalog && !catalog.instances.length ? <div className="loading-state">暂无可监控的本地实例</div> : <LiveLoadingState />}
    </main>
  </div>;
}
export default App;
