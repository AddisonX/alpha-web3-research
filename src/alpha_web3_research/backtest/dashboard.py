from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpha_web3_research.config import Settings


@dataclass(frozen=True)
class FreqtradeDashboardExportResult:
    output_path: Path
    archive_count: int
    strategy_count: int
    run_count: int
    trade_count: int


class FreqtradeDashboardExporter:
    """Normalize every Freqtrade result archive for the React dashboard."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def export(
        self,
        results_dir: Path | None = None,
        output_path: Path | None = None,
    ) -> FreqtradeDashboardExportResult:
        source_dir = self._resolve(
            results_dir or Path("freqtrade/user_data/backtest_results")
        )
        destination = self._resolve(
            output_path or Path("dashboard/public/data/backtests.json")
        )
        if not source_dir.exists():
            raise FileNotFoundError(f"Freqtrade result directory not found: {source_dir}")

        archives = sorted(source_dir.glob("*.zip"))
        if not archives:
            raise FileNotFoundError(f"no Freqtrade result archives found in: {source_dir}")

        latest_archive = self._latest_archive_name(source_dir)
        runs: list[dict[str, Any]] = []
        for archive_path in archives:
            payload = self._read_result(archive_path)
            metadata = self._read_metadata(archive_path)
            for strategy_name, result in payload["strategy"].items():
                runs.append(
                    self._normalize_run(
                        archive_path=archive_path,
                        strategy_name=strategy_name,
                        result=result,
                        metadata=metadata.get(strategy_name, {}),
                        is_latest=archive_path.name == latest_archive,
                    )
                )

        runs.sort(
            key=lambda run: (run["generatedAt"], run["strategy"], run["id"]),
            reverse=True,
        )
        strategy_names = sorted({run["strategy"] for run in runs})
        strategies = [
            self._strategy_summary(name, [run for run in runs if run["strategy"] == name])
            for name in strategy_names
        ]
        document = {
            "schemaVersion": 1,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "sourceDirectory": str(source_dir),
            "archiveCount": len(archives),
            "strategyCount": len(strategy_names),
            "runCount": len(runs),
            "tradeCount": sum(run["summary"]["trades"] for run in runs),
            "strategies": strategies,
            "runs": runs,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return FreqtradeDashboardExportResult(
            output_path=destination,
            archive_count=len(archives),
            strategy_count=len(strategy_names),
            run_count=len(runs),
            trade_count=document["tradeCount"],
        )

    def _resolve(self, path: Path) -> Path:
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = self.settings.project_root / candidate
        return candidate.resolve()

    @staticmethod
    def _latest_archive_name(results_dir: Path) -> str | None:
        path = results_dir / ".last_result.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("latest_backtest")
        return str(value) if value else None

    @staticmethod
    def _read_result(archive_path: Path) -> dict[str, Any]:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                if not member.endswith(".json") or member.endswith("_config.json"):
                    continue
                payload = json.loads(archive.read(member))
                if isinstance(payload, dict) and isinstance(payload.get("strategy"), dict):
                    return payload
        raise ValueError(f"no strategy result JSON found in archive: {archive_path}")

    @staticmethod
    def _read_metadata(archive_path: Path) -> dict[str, Any]:
        path = archive_path.with_suffix(".meta.json")
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    def _normalize_run(
        self,
        archive_path: Path,
        strategy_name: str,
        result: dict[str, Any],
        metadata: dict[str, Any],
        is_latest: bool,
    ) -> dict[str, Any]:
        run_id = str(metadata.get("run_id") or archive_path.stem)
        generated_at = datetime.fromtimestamp(
            archive_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
        starting_balance = self._number(result.get("starting_balance")) or 0.0
        daily = self._equity_curve(
            daily_profit=result.get("daily_profit", []),
            starting_balance=starting_balance,
        )
        trades = [self._trade(trade, index) for index, trade in enumerate(result.get("trades", []))]
        return {
            "id": f"{strategy_name}:{run_id}",
            "runId": run_id,
            "strategy": strategy_name,
            "archive": archive_path.name,
            "generatedAt": generated_at,
            "isLatest": is_latest,
            "period": {
                "start": result.get("backtest_start"),
                "end": result.get("backtest_end"),
                "days": self._integer(result.get("backtest_days")),
                "timeframe": result.get("timeframe"),
                "timeframeDetail": result.get("timeframe_detail"),
            },
            "execution": {
                "tradingMode": result.get("trading_mode"),
                "marginMode": result.get("margin_mode"),
                "stakeCurrency": result.get("stake_currency"),
                "stakeAmount": result.get("stake_amount"),
                "maxOpenTrades": self._integer(result.get("max_open_trades")),
                "stoplossPct": self._percent(result.get("stoploss")),
                "feePct": self._percent(trades[0]["feeOpen"]) if trades else None,
                "leverage": trades[0]["leverage"] if trades else None,
            },
            "summary": {
                "startingBalance": starting_balance,
                "finalBalance": self._number(result.get("final_balance")),
                "profitAbs": self._number(result.get("profit_total_abs")),
                "returnPct": self._percent(result.get("profit_total")),
                "cagrPct": self._percent(result.get("cagr")),
                "marketChangePct": self._percent(result.get("market_change")),
                "maxDrawdownPct": self._percent(result.get("max_drawdown_account")),
                "maxDrawdownAbs": self._number(result.get("max_drawdown_abs")),
                "walletMaxDrawdownPct": self._percent(
                    result.get("wallet_stats", {}).get("max_drawdown_account")
                ),
                "sharpe": self._number(result.get("wallet_stats", {}).get("sharpe")),
                "sortino": self._number(result.get("wallet_stats", {}).get("sortino")),
                "calmar": self._number(result.get("wallet_stats", {}).get("calmar")),
                "closedTradeSharpe": self._number(result.get("sharpe")),
                "profitFactor": self._number(result.get("profit_factor")),
                "expectancy": self._number(result.get("expectancy")),
                "expectancyRatio": self._number(result.get("expectancy_ratio")),
                "sqn": self._number(result.get("sqn")),
                "pValue": self._number(result.get("p_value")),
                "trades": self._integer(result.get("total_trades")),
                "tradesPerDay": self._number(result.get("trades_per_day")),
                "avgProfitPct": self._percent(result.get("profit_mean")),
                "winRatePct": self._ratio(
                    result.get("wins"),
                    result.get("total_trades"),
                ),
                "wins": self._integer(result.get("wins")),
                "draws": self._integer(result.get("draws")),
                "losses": self._integer(result.get("losses")),
                "winningDays": self._integer(result.get("winning_days")),
                "losingDays": self._integer(result.get("losing_days")),
                "bestDayAbs": self._number(result.get("backtest_best_day_abs")),
                "worstDayAbs": self._number(result.get("backtest_worst_day_abs")),
                "avgHolding": result.get("holding_avg"),
                "rejectedSignals": self._integer(result.get("rejected_signals")),
            },
            "directions": {
                "long": {
                    "trades": self._integer(result.get("trade_count_long")),
                    "profitAbs": self._number(result.get("profit_total_long_abs")),
                    "returnPct": self._percent(result.get("profit_total_long")),
                },
                "short": {
                    "trades": self._integer(result.get("trade_count_short")),
                    "profitAbs": self._number(result.get("profit_total_short_abs")),
                    "returnPct": self._percent(result.get("profit_total_short")),
                },
            },
            "equity": daily,
            "pairPerformance": self._summary_rows(
                result.get("results_per_pair", []),
                key_name="pair",
            ),
            "exitReasons": self._summary_rows(
                result.get("exit_reason_summary", []),
                key_name="reason",
            ),
            "entryTags": self._summary_rows(
                result.get("results_per_enter_tag", []),
                key_name="tag",
            ),
            "monthly": self._period_rows(result, "month"),
            "weekdays": self._period_rows(result, "weekday"),
            "trades": trades,
        }

    @classmethod
    def _equity_curve(
        cls,
        daily_profit: list[list[Any]],
        starting_balance: float,
    ) -> list[dict[str, Any]]:
        balance = starting_balance
        high_watermark = starting_balance
        points: list[dict[str, Any]] = []
        for row in daily_profit:
            if len(row) < 2:
                continue
            pnl = cls._number(row[1]) or 0.0
            balance += pnl
            high_watermark = max(high_watermark, balance)
            drawdown_pct = 0.0
            if high_watermark:
                drawdown_pct = ((balance / high_watermark) - 1.0) * 100.0
            points.append(
                {
                    "date": str(row[0]),
                    "profitAbs": pnl,
                    "balance": balance,
                    "returnPct": ((balance / starting_balance) - 1.0) * 100.0
                    if starting_balance
                    else None,
                    "drawdownPct": drawdown_pct,
                }
            )
        return points

    @classmethod
    def _summary_rows(
        cls,
        rows: list[dict[str, Any]],
        key_name: str,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for row in rows:
            key = str(row.get("key", ""))
            if key.upper() == "TOTAL":
                continue
            normalized.append(
                {
                    key_name: key,
                    "trades": cls._integer(row.get("trades")),
                    "profitAbs": cls._number(
                        row.get("profit_total_abs", row.get("profit_abs"))
                    ),
                    "returnPct": cls._percent(row.get("profit_total")),
                    "avgProfitPct": cls._percent(row.get("profit_mean")),
                    "winRatePct": cls._percent(row.get("winrate")),
                    "wins": cls._integer(row.get("wins")),
                    "draws": cls._integer(row.get("draws")),
                    "losses": cls._integer(row.get("losses")),
                    "profitFactor": cls._number(row.get("profit_factor")),
                    "avgDuration": row.get("duration_avg"),
                }
            )
        return normalized

    @classmethod
    def _period_rows(
        cls,
        result: dict[str, Any],
        period: str,
    ) -> list[dict[str, Any]]:
        rows = result.get("periodic_breakdown", {}).get(period, [])
        return [
            {
                "label": row.get("date"),
                "timestamp": cls._integer(row.get("date_ts")),
                "profitAbs": cls._number(row.get("profit_abs")),
                "trades": cls._integer(row.get("trades")),
                "wins": cls._integer(row.get("wins")),
                "losses": cls._integer(row.get("losses")),
                "profitFactor": cls._number(row.get("profit_factor")),
            }
            for row in rows
        ]

    @classmethod
    def _trade(cls, trade: dict[str, Any], index: int) -> dict[str, Any]:
        return {
            "id": index + 1,
            "pair": trade.get("pair"),
            "side": "short" if trade.get("is_short") else "long",
            "openDate": trade.get("open_date"),
            "closeDate": trade.get("close_date"),
            "openRate": cls._number(trade.get("open_rate")),
            "closeRate": cls._number(trade.get("close_rate")),
            "durationMinutes": cls._integer(trade.get("trade_duration")),
            "profitAbs": cls._number(trade.get("profit_abs")),
            "profitPct": cls._percent(trade.get("profit_ratio")),
            "stakeAmount": cls._number(trade.get("stake_amount")),
            "feeOpen": cls._number(trade.get("fee_open")),
            "feeClose": cls._number(trade.get("fee_close")),
            "fundingFees": cls._number(trade.get("funding_fees")),
            "leverage": cls._number(trade.get("leverage")),
            "entryTag": trade.get("enter_tag"),
            "exitReason": trade.get("exit_reason"),
        }

    @classmethod
    def _strategy_summary(
        cls,
        name: str,
        runs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        latest = next((run for run in runs if run["isLatest"]), runs[0])
        best = max(runs, key=lambda run: run["summary"]["returnPct"] or -math.inf)
        return {
            "name": name,
            "runCount": len(runs),
            "latestRunId": latest["id"],
            "bestRunId": best["id"],
            "latestReturnPct": latest["summary"]["returnPct"],
            "latestMaxDrawdownPct": latest["summary"]["maxDrawdownPct"],
            "latestProfitFactor": latest["summary"]["profitFactor"],
        }

    @staticmethod
    def _number(value: Any) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _integer(cls, value: Any) -> int:
        number = cls._number(value)
        return int(number) if number is not None else 0

    @classmethod
    def _percent(cls, value: Any) -> float | None:
        number = cls._number(value)
        return number * 100.0 if number is not None else None

    @classmethod
    def _ratio(cls, numerator: Any, denominator: Any) -> float | None:
        numerator_value = cls._number(numerator)
        denominator_value = cls._number(denominator)
        if numerator_value is None or not denominator_value:
            return None
        return numerator_value / denominator_value * 100.0
