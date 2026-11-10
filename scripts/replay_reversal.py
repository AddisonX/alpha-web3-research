"""Replay public historical candles through the unchanged live pipeline, offline."""
import argparse
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import ZipFile

import ccxt
import pandas as pd
import polars as pl

from src.config import Settings
from src.live.safety import check_live_signal_status, validate_signal_rows
from src.live.signals import LiveLongOnlySignalProducer
from src.strategies.reversal import STRATEGY_ID

ROOT = Path(__file__).resolve().parents[1]
BAR_MS = 14_400_000
DAY_MS = 86_400_000


def download_snapshot(pairs_path: Path, destination: Path, month: str) -> None:
    pairs = json.loads(pairs_path.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    def fetch(pair):
        native = pair.split("/")[0] + "USDT"
        key = f"data/futures/um/monthly/klines/{native}/4h/{native}-4h-{month}.zip"
        url = "https://data.binance.vision/" + key
        try:
            with urlopen(url, timeout=30) as response:
                content = response.read()
            with urlopen(url + ".CHECKSUM", timeout=30) as response:
                expected = response.read().decode().split()[0]
        except HTTPError as exc:
            if exc.code == 404:
                return None, {"pair": pair, "missing_archive": url}
            raise
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected:
            raise ValueError(f"archive checksum mismatch: {pair}")
        with ZipFile(io.BytesIO(content)) as archive:
            names = [n for n in archive.namelist() if n.endswith(".csv")]
            if len(names) != 1:
                raise ValueError("expected one CSV per archive")
            csv = archive.read(names[0])
        columns = ["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
        frame = pl.read_csv(io.BytesIO(csv), has_header=not csv.splitlines()[0].split(b",")[0].isdigit(),
                            new_columns=columns, schema_overrides={c: pl.Float64 for c in columns})
        frame = frame.select(pl.lit(pair).alias("symbol"),
            pl.when(pl.col("timestamp") > 10**14).then(pl.col("timestamp") / 1000)
              .otherwise(pl.col("timestamp")).cast(pl.Int64).alias("timestamp"),
            "open", "high", "low", "close", "volume")
        return frame, {"pair": pair, "url": url, "sha256": digest, "rows": frame.height}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fetch, pairs))
    frames = [frame for frame, _ in results if frame is not None]
    if not frames:
        raise ValueError("no historical archives available")
    pl.concat(frames).sort(["symbol", "timestamp"]).write_parquet(destination)
    destination.with_suffix(".manifest.json").write_text(json.dumps({
        "purpose": "historical compatibility acceptance, not a performance backtest",
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "month": month, "archives": [info for _, info in results],
    }, indent=2))


class SnapshotExchange:
    id = "binanceusdm"
    urls = {"api": {}}
    parse_timeframe = staticmethod(ccxt.Exchange.parse_timeframe)

    def __init__(self, bars: pl.DataFrame):
        self.bars = bars
        self.now_ms = 0

    def milliseconds(self):
        return self.now_ms

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        assert timeframe == "4h"
        return self.bars.filter((pl.col("symbol") == symbol) & (pl.col("timestamp") >= since)
            & (pl.col("timestamp") + BAR_MS <= self.now_ms)).sort("timestamp").head(limit).select(
                "timestamp", "open", "high", "low", "close", "volume").rows()


def replay(snapshot: Path, output: Path, start_ms: int, days: int) -> dict:
    if days < 1:
        raise ValueError("days must be positive")
    bars = pl.read_parquet(snapshot)
    if bars.select(pl.struct("symbol", "timestamp").n_unique()).item() != bars.height:
        raise ValueError("duplicate snapshot candles")
    output.mkdir(parents=True, exist_ok=True)
    # Fix candidates from what is visible before the first decision, not future survival.
    pairs = sorted(bars.filter(pl.col("timestamp") + BAR_MS <= start_ms + BAR_MS)["symbol"].unique().to_list())
    pairs_path = output / "pairs.json"
    pairs_path.write_text(json.dumps(pairs))
    exchange = SnapshotExchange(bars)
    producer = LiveLongOnlySignalProducer(Settings(project_root=ROOT), exchange, pairs_path,
        output / "alpha_live_long_only_signals.parquet", output / "alpha_live_long_only_status.json",
        strategy_id=STRATEGY_ID)
    runs = []
    for day in range(days):
        stamp = start_ms + day * DAY_MS
        now = datetime.fromtimestamp((stamp + BAR_MS + 300_000) / 1000, timezone.utc)
        exchange.now_ms = int(now.timestamp() * 1000)
        class ReplayClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return now if tz else now.replace(tzinfo=None)
        with patch("src.live.signal_store.datetime", ReplayClock):
            result = producer.run(force=True)
        status = json.loads(producer.status_path.read_text())
        health = check_live_signal_status(status, now)
        if not health.ready or validate_signal_rows(pd.read_parquet(producer.signal_path), health) is not None:
            raise AssertionError("replayed signals failed the existing execution safety contract")
        runs.append({"decision_timestamp": stamp, "cross_section": result.cross_section_count,
                     "selected_pairs": result.selected_pairs, "safety_validated": True})
    report = {"strategy_id": STRATEGY_ID, "model_count": 0, "source_snapshot": str(snapshot.resolve()),
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(), "runs": runs,
        "performance_claim": False, "execution_worker_started": False}
    (output / "replay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/acceptance/reversal/bars.parquet")
    parser.add_argument("--pairs", type=Path, default=ROOT / "freqtrade/instances/tokyo-lightgbm/pairs.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/acceptance/reversal/replay")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--month", default="2026-10")
    parser.add_argument("--start", default="2026-10-08")
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()
    if args.download:
        download_snapshot(args.pairs, args.snapshot, args.month)
    stamp = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    print(json.dumps(replay(args.snapshot, args.output, stamp, args.days), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
