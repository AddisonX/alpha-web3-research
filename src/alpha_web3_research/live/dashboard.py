from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

import polars as pl
from fastapi import FastAPI, HTTPException, Response

STARTING_BALANCE = 10_000.0
TESTNET_TICKER_URL = "https://testnet.binancefuture.com/fapi/v1/ticker/price"
MAINNET_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/price"
HISTORY_RANGE_HOURS = {"24h": 24, "7d": 24 * 7, "30d": 24 * 30, "all": None}
HISTORY_INTERVAL_MINUTES = {
    "raw": None,
    "15m": 15,
    "1h": 60,
    "4h": 4 * 60,
    "1d": 24 * 60,
}


class LiveDashboardProvider:
    def __init__(
        self,
        database_path: Path,
        signal_path: Path,
        status_path: Path,
        history_path: Path,
        price_loader: Callable[[list[str]], dict[str, float]] | None = None,
    ) -> None:
        self.database_path = database_path
        self.signal_path = signal_path
        self.status_path = status_path
        self.history_path = history_path
        self.history_database_path = history_path.with_suffix(".sqlite")
        self.price_loader = price_loader or load_testnet_prices
        self._lock = threading.Lock()
        self._price_cache: tuple[float, dict[str, float]] = (0.0, {})
        self._history_ready = False
        self._history_cache: list[dict] | None = None

    def snapshot(self) -> dict:
        with self._lock:
            generated_at = datetime.now(timezone.utc)
            trades = self._read_trades()
            open_trades = [trade for trade in trades if trade["is_open"]]
            closed_trades = [trade for trade in trades if not trade["is_open"]]
            prices, price_state = self._prices([trade["pair"] for trade in open_trades])
            signal_status = self._read_json(self.status_path)
            signal_scores = self._read_signal_scores(signal_status.get("decision_timestamp"))

            positions = [
                position_snapshot(
                    trade=trade,
                    current_rate=prices.get(trade["pair"], trade["open_rate"]),
                    signal=signal_scores.get(trade["pair"], {}),
                    now=generated_at,
                )
                for trade in open_trades
            ]
            positions.sort(key=lambda row: row["pnl_abs"], reverse=True)

            realized_pnl = sum(float(trade["close_profit_abs"] or 0.0) for trade in closed_trades)
            unrealized_pnl = sum(float(position["pnl_abs"]) for position in positions)
            total_pnl = realized_pnl + unrealized_pnl
            equity = STARTING_BALANCE + total_pnl
            exposure = sum(float(trade["stake_amount"]) for trade in open_trades)
            fees = sum(
                float(trade["fee_open_cost"] or 0.0)
                + float(trade["fee_close_cost"] or 0.0)
                for trade in trades
            )
            wins = sum(1 for trade in closed_trades if float(trade["close_profit_abs"] or 0.0) > 0)
            losses = sum(1 for trade in closed_trades if float(trade["close_profit_abs"] or 0.0) < 0)
            self._append_history(
                timestamp=generated_at,
                equity=equity,
                total_pnl=total_pnl,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
            )
            history = self._read_history()
            max_drawdown_pct = history_max_drawdown_pct(history)

            return {
                "schema_version": 1,
                "generated_at": generated_at.isoformat(),
                "mode": "dry_run",
                "strategy": "AlphaCrossSectionalLiveLongOnlyStrategy",
                "market_data_source": signal_status.get(
                    "market_data_source", "binance_futures_testnet"
                ),
                "price_state": price_state,
                "summary": {
                    "starting_balance": STARTING_BALANCE,
                    "equity": equity,
                    "wallet_balance": STARTING_BALANCE + realized_pnl,
                    "available_balance": max(0.0, STARTING_BALANCE + realized_pnl - exposure),
                    "total_pnl": total_pnl,
                    "realized_pnl": realized_pnl,
                    "unrealized_pnl": unrealized_pnl,
                    "return_pct": total_pnl / STARTING_BALANCE * 100.0,
                    "max_drawdown_pct": max_drawdown_pct,
                    "exposure": exposure,
                    "exposure_pct": exposure / max(equity, 1.0) * 100.0,
                    "fees": fees,
                    "open_positions": len(open_trades),
                    "closed_trades": len(closed_trades),
                    "wins": wins,
                    "losses": losses,
                    "win_rate_pct": (
                        wins / len(closed_trades) * 100.0 if closed_trades else None
                    ),
                },
                "signal": {
                    "state": signal_status.get("state", "unknown"),
                    "generated_at": signal_status.get("generated_at"),
                    "decision_time": signal_status.get("decision_time"),
                    "next_rebalance_time": signal_status.get("next_rebalance_time"),
                    "grace_valid_until_time": signal_status.get("grace_valid_until_time"),
                    "source_experiment": signal_status.get("source_experiment"),
                    "model_count": signal_status.get("model_count"),
                    "timeframe": signal_status.get("timeframe"),
                    "horizon_bars": signal_status.get("horizon_bars"),
                    "label_column": signal_status.get("label_column"),
                    "cross_section_count": signal_status.get("cross_section_count"),
                    "selected_count": signal_status.get("selected_count"),
                    "selected_pairs": signal_status.get("selected_pairs", []),
                    "fetch_error_count": signal_status.get("fetch_error_count"),
                },
                "positions": positions,
                "closed_trades": [
                    closed_trade_snapshot(trade) for trade in reversed(closed_trades[-50:])
                ],
                "orders": self._read_orders(),
                "equity_history": history[-120:],
            }

    def equity_history(self, range_key: str, interval_key: str) -> dict:
        with self._lock:
            history = list(self._read_history())
        return build_equity_history_response(history, range_key, interval_key)

    def _read_trades(self) -> list[dict]:
        if not self.database_path.exists():
            return []
        query = """
            SELECT id, pair, is_open, fee_open, fee_open_cost, fee_close,
                   fee_close_cost, open_rate, close_rate, close_profit,
                   close_profit_abs, stake_amount, amount, open_date, close_date,
                   stop_loss, leverage, is_short, funding_fees, enter_tag, exit_reason
            FROM trades
            ORDER BY id
        """
        with sqlite3.connect(
            f"file:{self.database_path}?mode=ro", uri=True, timeout=5
        ) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(query)]

    def _read_orders(self) -> list[dict]:
        if not self.database_path.exists():
            return []
        query = """
            SELECT id, ft_trade_id, ft_pair, ft_order_side, status, order_type,
                   average, filled, cost, order_date, order_filled_date
            FROM orders
            ORDER BY id DESC
            LIMIT 30
        """
        with sqlite3.connect(
            f"file:{self.database_path}?mode=ro", uri=True, timeout=5
        ) as connection:
            connection.row_factory = sqlite3.Row
            return [
                {
                    "id": row["id"],
                    "trade_id": row["ft_trade_id"],
                    "pair": row["ft_pair"],
                    "side": row["ft_order_side"],
                    "status": row["status"],
                    "order_type": row["order_type"],
                    "average": row["average"],
                    "filled": row["filled"],
                    "cost": row["cost"],
                    "created_at": _as_utc_iso(row["order_date"]),
                    "filled_at": _as_utc_iso(row["order_filled_date"]),
                }
                for row in connection.execute(query)
            ]

    def _prices(self, pairs: list[str]) -> tuple[dict[str, float], str]:
        if not pairs:
            return {}, "healthy"
        cache_time, cache = self._price_cache
        if time.monotonic() - cache_time < 5 and all(pair in cache for pair in pairs):
            return cache, "healthy"
        try:
            prices = self.price_loader(pairs)
            self._price_cache = (time.monotonic(), prices)
            return prices, "healthy"
        except Exception:
            return cache, "degraded"

    def _read_signal_scores(self, decision_timestamp: int | None) -> dict[str, dict]:
        if not self.signal_path.exists() or decision_timestamp is None:
            return {}
        rows = (
            pl.read_parquet(self.signal_path)
            .filter(
                (pl.col("decision_timestamp") == int(decision_timestamp))
                & pl.col("is_rebalance")
            )
            .select(
                "pair",
                "model_score",
                "model_score_rank_pct",
                "score_position",
                "target_position",
            )
        )
        return {str(row["pair"]): row for row in rows.iter_rows(named=True)}

    def _append_history(
        self,
        timestamp: datetime,
        equity: float,
        total_pnl: float,
        realized_pnl: float,
        unrealized_pnl: float,
    ) -> None:
        self._ensure_history_database()
        minute = timestamp.replace(second=0, microsecond=0).isoformat()
        point = {
            "timestamp": minute,
            "equity": equity,
            "total_pnl": total_pnl,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "return_pct": total_pnl / STARTING_BALANCE * 100.0,
        }
        with sqlite3.connect(self.history_database_path, timeout=5) as connection:
            connection.execute(
                """
                INSERT INTO equity_history (
                    timestamp, equity, total_pnl, realized_pnl,
                    unrealized_pnl, return_pct
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET
                    equity = excluded.equity,
                    total_pnl = excluded.total_pnl,
                    realized_pnl = excluded.realized_pnl,
                    unrealized_pnl = excluded.unrealized_pnl,
                    return_pct = excluded.return_pct
                """,
                (
                    point["timestamp"],
                    point["equity"],
                    point["total_pnl"],
                    point["realized_pnl"],
                    point["unrealized_pnl"],
                    point["return_pct"],
                ),
            )
        if self._history_cache is not None:
            if self._history_cache and self._history_cache[-1]["timestamp"] == minute:
                self._history_cache[-1] = point
            else:
                self._history_cache.append(point)

    def _read_history(self) -> list[dict]:
        if self._history_cache is not None:
            return self._history_cache
        self._ensure_history_database()
        with sqlite3.connect(self.history_database_path, timeout=5) as connection:
            connection.row_factory = sqlite3.Row
            self._history_cache = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT timestamp, equity, total_pnl, realized_pnl,
                           unrealized_pnl, return_pct
                    FROM equity_history
                    ORDER BY timestamp
                    """
                )
            ]
        return self._history_cache

    def _ensure_history_database(self) -> None:
        if self._history_ready:
            return
        self.history_database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.history_database_path, timeout=5) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS equity_history (
                    timestamp TEXT PRIMARY KEY,
                    equity REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    return_pct REAL NOT NULL
                )
                """
            )
            count = int(
                connection.execute("SELECT COUNT(*) FROM equity_history").fetchone()[0]
            )
            legacy = self._read_json(self.history_path)
            if count == 0 and isinstance(legacy, list) and legacy:
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO equity_history (
                        timestamp, equity, total_pnl, realized_pnl,
                        unrealized_pnl, return_pct
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            point["timestamp"],
                            point["equity"],
                            point["total_pnl"],
                            point["realized_pnl"],
                            point["unrealized_pnl"],
                            point["return_pct"],
                        )
                        for point in legacy
                    ],
                )
        self._history_ready = True

    @staticmethod
    def _read_json(path: Path) -> dict | list:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}


def position_snapshot(
    trade: dict,
    current_rate: float,
    signal: dict,
    now: datetime,
) -> dict:
    leverage = float(trade["leverage"] or 1.0)
    fee_open = float(trade["fee_open"] or 0.0)
    fee_close = float(trade["fee_close"] or 0.0)
    open_rate = float(trade["open_rate"])
    profit_ratio = (
        (current_rate * (1.0 - fee_close))
        / (open_rate * (1.0 + fee_open))
        - 1.0
    ) * leverage
    pnl_abs = float(trade["stake_amount"]) * profit_ratio
    open_date = _parse_database_datetime(trade["open_date"])
    return {
        "id": trade["id"],
        "pair": trade["pair"],
        "side": "short" if trade["is_short"] else "long",
        "open_rate": open_rate,
        "current_rate": current_rate,
        "stake_amount": trade["stake_amount"],
        "amount": trade["amount"],
        "pnl_abs": pnl_abs,
        "pnl_pct": profit_ratio * 100.0,
        "open_date": open_date.isoformat(),
        "duration_minutes": max(0, int((now - open_date).total_seconds() // 60)),
        "stop_loss": trade["stop_loss"],
        "leverage": leverage,
        "entry_tag": trade["enter_tag"],
        "model_score": signal.get("model_score"),
        "model_score_rank_pct": (
            float(signal["model_score_rank_pct"]) * 100.0
            if signal.get("model_score_rank_pct") is not None
            else None
        ),
        "score_position": signal.get("score_position"),
    }


def closed_trade_snapshot(trade: dict) -> dict:
    return {
        "id": trade["id"],
        "pair": trade["pair"],
        "side": "short" if trade["is_short"] else "long",
        "open_date": _as_utc_iso(trade["open_date"]),
        "close_date": _as_utc_iso(trade["close_date"]),
        "open_rate": trade["open_rate"],
        "close_rate": trade["close_rate"],
        "stake_amount": trade["stake_amount"],
        "pnl_abs": trade["close_profit_abs"],
        "pnl_pct": (
            float(trade["close_profit"]) * 100.0
            if trade["close_profit"] is not None
            else None
        ),
        "exit_reason": trade["exit_reason"],
    }


def load_binance_futures_prices(
    pairs: list[str],
    ticker_url: str,
) -> dict[str, float]:
    with urlopen(ticker_url, timeout=8) as response:
        rows = json.load(response)
    by_symbol = {str(row["symbol"]): float(row["price"]) for row in rows}
    return {
        pair: by_symbol[_pair_to_exchange_symbol(pair)]
        for pair in pairs
        if _pair_to_exchange_symbol(pair) in by_symbol
    }


def history_max_drawdown_pct(history: list[dict]) -> float:
    peak = STARTING_BALANCE
    max_drawdown = 0.0
    for point in history:
        equity = float(point["equity"])
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100.0)
    return max_drawdown


def build_equity_history_response(
    history: list[dict],
    range_key: str,
    interval_key: str,
) -> dict:
    if range_key not in HISTORY_RANGE_HOURS:
        raise ValueError(f"unsupported history range: {range_key}")
    if interval_key != "auto" and interval_key not in HISTORY_INTERVAL_MINUTES:
        raise ValueError(f"unsupported history interval: {interval_key}")

    if not history:
        return {
            "range": range_key,
            "interval": "raw" if interval_key == "auto" else interval_key,
            "total_points": 0,
            "filtered_points": 0,
            "returned_points": 0,
            "from": None,
            "to": None,
            "points": [],
        }

    range_hours = HISTORY_RANGE_HOURS[range_key]
    filtered = history
    if range_hours is not None:
        end = datetime.fromisoformat(str(history[-1]["timestamp"]))
        cutoff = end - timedelta(hours=range_hours)
        filtered = [
            point
            for point in history
            if datetime.fromisoformat(str(point["timestamp"])) >= cutoff
        ]

    resolved_interval = (
        _automatic_history_interval(filtered)
        if interval_key == "auto"
        else interval_key
    )
    interval_minutes = HISTORY_INTERVAL_MINUTES[resolved_interval]
    points = (
        filtered
        if interval_minutes is None
        else _aggregate_history_points(filtered, interval_minutes)
    )
    return {
        "range": range_key,
        "interval": resolved_interval,
        "total_points": len(history),
        "filtered_points": len(filtered),
        "returned_points": len(points),
        "from": filtered[0]["timestamp"] if filtered else None,
        "to": filtered[-1]["timestamp"] if filtered else None,
        "points": points,
    }


def _automatic_history_interval(history: list[dict]) -> str:
    if len(history) <= 750:
        return "raw"
    first = datetime.fromisoformat(str(history[0]["timestamp"]))
    last = datetime.fromisoformat(str(history[-1]["timestamp"]))
    span_minutes = max(1.0, (last - first).total_seconds() / 60.0)
    for key in ("15m", "1h", "4h", "1d"):
        minutes = HISTORY_INTERVAL_MINUTES[key]
        if minutes is not None and span_minutes / minutes <= 750:
            return key
    return "1d"


def _aggregate_history_points(history: list[dict], interval_minutes: int) -> list[dict]:
    if not history:
        return []
    buckets: dict[int, dict] = {}
    interval_seconds = interval_minutes * 60
    for point in history:
        timestamp = datetime.fromisoformat(str(point["timestamp"]))
        bucket = int(timestamp.timestamp()) // interval_seconds
        buckets[bucket] = point
    points = list(buckets.values())
    if points[0]["timestamp"] != history[0]["timestamp"]:
        points.insert(0, history[0])
    return points


def create_app() -> FastAPI:
    root = Path(os.environ.get("ALPHA_PROJECT_ROOT", Path.cwd())).resolve()
    user_data = root / "freqtrade" / "user_data"
    market_data_source = os.environ.get(
        "ALPHA_DASHBOARD_MARKET_DATA_SOURCE",
        "binance_futures_testnet",
    )
    ticker_url = (
        MAINNET_TICKER_URL
        if market_data_source == "binance_futures"
        else TESTNET_TICKER_URL
    )
    provider = LiveDashboardProvider(
        database_path=user_data / "tradesv3.live-long-only.dryrun.sqlite",
        signal_path=user_data / "signals" / "alpha_live_long_only_signals.parquet",
        status_path=user_data / "signals" / "alpha_live_long_only_status.json",
        history_path=user_data / "signals" / "alpha_live_equity_history.json",
        price_loader=partial(
            load_binance_futures_prices,
            ticker_url=ticker_url,
        ),
    )
    app = FastAPI(
        title="Alpha Live Monitor",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/api/live")
    def live(response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        return provider.snapshot()

    @app.get("/api/live/equity")
    def live_equity(
        response: Response,
        range: str = "all",
        interval: str = "auto",
    ) -> dict:
        response.headers["Cache-Control"] = "public, max-age=15, stale-while-revalidate=45"
        try:
            return provider.equity_history(range, interval)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/health")
    def health(response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    return app


app = create_app()


def _pair_to_exchange_symbol(pair: str) -> str:
    return pair.split(":", maxsplit=1)[0].replace("/", "")


def _parse_database_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace(" ", "T"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _as_utc_iso(value: str | None) -> str | None:
    return _parse_database_datetime(value).isoformat() if value else None
