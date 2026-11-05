export type NullableNumber = number | null;

export interface DashboardData {
  schemaVersion: number;
  generatedAt: string;
  sourceDirectory: string;
  archiveCount: number;
  strategyCount: number;
  runCount: number;
  tradeCount: number;
  strategies: StrategySummary[];
  runs: StrategyRun[];
}

export interface StrategySummary {
  name: string;
  runCount: number;
  latestRunId: string;
  bestRunId: string;
  latestReturnPct: NullableNumber;
  latestMaxDrawdownPct: NullableNumber;
  latestProfitFactor: NullableNumber;
}

export interface StrategyRun {
  id: string;
  runId: string;
  strategy: string;
  archive: string;
  generatedAt: string;
  isLatest: boolean;
  period: {
    start: string;
    end: string;
    days: number;
    timeframe: string;
    timeframeDetail: string | null;
  };
  execution: {
    tradingMode: string;
    marginMode: string;
    stakeCurrency: string;
    stakeAmount: number | string;
    maxOpenTrades: number;
    stoplossPct: NullableNumber;
    feePct: NullableNumber;
    leverage: NullableNumber;
  };
  summary: {
    startingBalance: number;
    finalBalance: NullableNumber;
    profitAbs: NullableNumber;
    returnPct: NullableNumber;
    cagrPct: NullableNumber;
    marketChangePct: NullableNumber;
    maxDrawdownPct: NullableNumber;
    maxDrawdownAbs: NullableNumber;
    walletMaxDrawdownPct: NullableNumber;
    sharpe: NullableNumber;
    sortino: NullableNumber;
    calmar: NullableNumber;
    closedTradeSharpe: NullableNumber;
    profitFactor: NullableNumber;
    expectancy: NullableNumber;
    expectancyRatio: NullableNumber;
    sqn: NullableNumber;
    pValue: NullableNumber;
    trades: number;
    tradesPerDay: NullableNumber;
    avgProfitPct: NullableNumber;
    winRatePct: NullableNumber;
    wins: number;
    draws: number;
    losses: number;
    winningDays: number;
    losingDays: number;
    bestDayAbs: NullableNumber;
    worstDayAbs: NullableNumber;
    avgHolding: string;
    rejectedSignals: number;
  };
  directions: {
    long: DirectionSummary;
    short: DirectionSummary;
  };
  equity: EquityPoint[];
  pairPerformance: SummaryRow[];
  exitReasons: SummaryRow[];
  entryTags: SummaryRow[];
  monthly: PeriodRow[];
  weekdays: PeriodRow[];
  trades: Trade[];
}

export interface DirectionSummary {
  trades: number;
  profitAbs: NullableNumber;
  returnPct: NullableNumber;
}

export interface EquityPoint {
  date: string;
  profitAbs: number;
  balance: number;
  returnPct: NullableNumber;
  drawdownPct: number;
}

export interface SummaryRow {
  pair?: string;
  reason?: string;
  tag?: string;
  trades: number;
  profitAbs: NullableNumber;
  returnPct: NullableNumber;
  avgProfitPct: NullableNumber;
  winRatePct: NullableNumber;
  wins: number;
  draws: number;
  losses: number;
  profitFactor: NullableNumber;
  avgDuration: string | null;
}

export interface PeriodRow {
  label: string;
  timestamp: number;
  profitAbs: NullableNumber;
  trades: number;
  wins: number;
  losses: number;
  profitFactor: NullableNumber;
}

export interface Trade {
  id: number;
  pair: string;
  side: "long" | "short";
  openDate: string;
  closeDate: string;
  openRate: NullableNumber;
  closeRate: NullableNumber;
  durationMinutes: number;
  profitAbs: NullableNumber;
  profitPct: NullableNumber;
  stakeAmount: NullableNumber;
  feeOpen: NullableNumber;
  feeClose: NullableNumber;
  fundingFees: NullableNumber;
  leverage: NullableNumber;
  entryTag: string;
  exitReason: string;
}

export interface LiveDashboardData {
  schema_version: number;
  generated_at: string;
  mode: "dry_run";
  strategy: string;
  market_data_source: string;
  price_state: "healthy" | "degraded";
  summary: LiveSummary;
  signal: LiveSignalStatus;
  positions: LivePosition[];
  closed_trades: LiveClosedTrade[];
  orders: LiveOrder[];
  equity_history: LiveEquityPoint[];
}

export interface LiveSummary {
  starting_balance: number;
  equity: number;
  wallet_balance: number;
  available_balance: number;
  total_pnl: number;
  realized_pnl: number;
  unrealized_pnl: number;
  return_pct: number;
  max_drawdown_pct: number;
  exposure: number;
  exposure_pct: number;
  fees: number;
  open_positions: number;
  closed_trades: number;
  wins: number;
  losses: number;
  win_rate_pct: NullableNumber;
}

export interface LiveSignalStatus {
  state: string;
  generated_at: string | null;
  decision_time: string | null;
  next_rebalance_time: string | null;
  grace_valid_until_time: string | null;
  source_experiment: string | null;
  model_count: number | null;
  timeframe: string | null;
  horizon_bars: number | null;
  label_column: string | null;
  cross_section_count: number | null;
  selected_count: number | null;
  selected_pairs: string[];
  fetch_error_count: number | null;
}

export interface LivePosition {
  id: number;
  pair: string;
  side: "long" | "short";
  open_rate: number;
  current_rate: number;
  stake_amount: number;
  amount: number;
  pnl_abs: number;
  pnl_pct: number;
  open_date: string;
  duration_minutes: number;
  stop_loss: NullableNumber;
  leverage: number;
  entry_tag: string | null;
  model_score: NullableNumber;
  model_score_rank_pct: NullableNumber;
  score_position: number | null;
}

export interface LiveClosedTrade {
  id: number;
  pair: string;
  side: "long" | "short";
  open_date: string;
  close_date: string | null;
  open_rate: number;
  close_rate: NullableNumber;
  stake_amount: number;
  pnl_abs: NullableNumber;
  pnl_pct: NullableNumber;
  exit_reason: string | null;
}

export interface LiveOrder {
  id: number;
  trade_id: number;
  pair: string;
  side: string;
  status: string;
  order_type: string;
  average: NullableNumber;
  filled: NullableNumber;
  cost: NullableNumber;
  created_at: string | null;
  filled_at: string | null;
}

export interface LiveEquityPoint {
  timestamp: string;
  equity: number;
  total_pnl: number;
  realized_pnl: number;
  unrealized_pnl: number;
  return_pct: number;
}

export interface LiveEquityHistoryResponse {
  range: "24h" | "7d" | "30d" | "all";
  interval: "raw" | "15m" | "1h" | "4h" | "1d";
  total_points: number;
  filtered_points: number;
  returned_points: number;
  from: string | null;
  to: string | null;
  points: LiveEquityPoint[];
}
