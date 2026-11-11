export type NullableNumber = number | null;

export interface MonitoredInstance {
  id: string;
  display_name: string;
  strategy_id: string;
  mode: "dry_run";
  runtime_state: string;
  runtime_updated_at: string | null;
  audit_state: string;
  entry_allowed: boolean;
  database_ready: boolean;
}

export interface InstanceCatalog {
  scope: "local";
  default_instance: string | null;
  instances: MonitoredInstance[];
  errors: Array<{ id: string; error: string }>;
}

export interface LiveDashboardData {
  instance: MonitoredInstance;
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
  short_pairs?: string[];
  allow_short?: boolean;
  reason?: string | null;
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
