import type { NullableNumber, StrategyRun } from "./types";

const numberFormatter = new Intl.NumberFormat("zh-CN", {
  maximumFractionDigits: 2,
});

const currencyFormatter = new Intl.NumberFormat("zh-CN", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export function formatNumber(value: NullableNumber, digits = 2): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return value.toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatCompact(value: NullableNumber): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return numberFormatter.format(value);
}

export function formatPct(value: NullableNumber, signed = false): string {
  if (value === null || !Number.isFinite(value)) return "—";
  const prefix = signed && value > 0 ? "+" : "";
  return `${prefix}${formatNumber(value)}%`;
}

export function formatMoney(
  value: NullableNumber,
  currency = "USDT",
  signed = false,
): string {
  if (value === null || !Number.isFinite(value)) return "—";
  const prefix = signed && value > 0 ? "+" : "";
  return `${prefix}${currencyFormatter.format(value)} ${currency}`;
}

export function formatDate(value: string): string {
  if (!value) return "—";
  const date = new Date(value.replace(" ", "T"));
  if (Number.isNaN(date.valueOf())) return value.slice(0, 10);
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

export function formatDateTime(value: string): string {
  if (!value) return "—";
  const date = new Date(value.replace(" ", "T"));
  if (Number.isNaN(date.valueOf())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

export function formatDuration(minutes: number): string {
  if (!Number.isFinite(minutes)) return "—";
  const days = Math.floor(minutes / 1440);
  const hours = Math.floor((minutes % 1440) / 60);
  const mins = minutes % 60;
  if (days) return `${days}天 ${hours}小时`;
  if (hours) return `${hours}小时 ${mins}分`;
  return `${mins}分钟`;
}

export function formatBacktestDuration(value: string): string {
  if (!value) return "—";
  return value
    .replace(/(\d+)\s+days?/g, "$1天")
    .replace(/,\s*/g, " ")
    .replace(/^0天\s*/, "");
}

export function shortRunId(run: StrategyRun): string {
  return `${run.archive.replace("backtest-result-", "").replace(".zip", "")}`;
}

export function runLabel(run: StrategyRun): string {
  return `${run.strategy} · ${shortRunId(run)}`;
}
