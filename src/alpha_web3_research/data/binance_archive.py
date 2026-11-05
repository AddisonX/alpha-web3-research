"""Public Binance USD-M archives, including instruments no longer listed today."""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

import polars as pl

from alpha_web3_research.data.ohlcv import OHLCV_COLUMNS

ARCHIVE_BASE = "https://data.binance.vision/"
LIST_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
NAMESPACE = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
BAR_MS = 14_400_000


def native_to_symbol(native: str) -> str:
    if not re.fullmatch(r"[A-Z0-9]+USDT", native):
        raise ValueError(f"not a USD-M USDT perpetual archive identifier: {native}")
    return f"{native[:-4]}/USDT:USDT"


def parse_archive(content: bytes, native: str, source: str) -> pl.DataFrame:
    with ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError("kline archive must contain exactly one CSV")
        raw = archive.read(names[0])
    header = raw.splitlines()[0].split(b",")[0]
    columns = ["timestamp", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "trade_count", "taker_base", "taker_quote", "ignore"]
    frame = pl.read_csv(io.BytesIO(raw), has_header=not header.isdigit(), new_columns=columns,
                        schema_overrides={name: pl.Float64 for name in columns})
    frame = frame.with_columns(
        pl.when(pl.col("timestamp") > 10**14)
        .then(pl.col("timestamp") / 1000).otherwise(pl.col("timestamp"))
        .cast(pl.Int64).alias("timestamp"),
        pl.lit("binanceusdm").alias("exchange"), pl.lit("swap").alias("market_type"),
        pl.lit(native_to_symbol(native)).alias("symbol"), pl.lit("4h").alias("timeframe"),
        pl.lit(datetime.now(timezone.utc).isoformat()).alias("fetched_at"),
        pl.lit(source).alias("source"),
    ).select(*OHLCV_COLUMNS, "quote_volume")
    if frame.filter(
        (pl.col("timestamp") % BAR_MS != 0) | (pl.col("low") <= 0)
        | (pl.col("high") < pl.max_horizontal("open", "close", "low"))
        | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
        | (pl.col("volume") < 0)
        | ~pl.all_horizontal([pl.col(name).is_finite() for name in ("open", "high", "low", "close", "volume", "quote_volume")])
    ).height:
        raise ValueError(f"invalid OHLCV archive: {source}")
    return frame


class BinanceArchiveClient:
    def __init__(self, cache_root: Path, workers: int = 12) -> None:
        if not 1 <= workers <= 32:
            raise ValueError("archive workers must be in [1, 32]")
        self.cache_root = cache_root
        self.workers = workers

    @staticmethod
    def _request(url: str) -> bytes:
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={"User-Agent": "AlphaWeb3Research/1.0"}), timeout=25) as response:
                    return response.read()
            except HTTPError as error:
                if error.code == 404 or (error.code < 500 and error.code != 429):
                    raise
                if attempt == 2:
                    raise
            except (URLError, TimeoutError, OSError):
                if attempt == 2:
                    raise
            time.sleep(0.5 * (attempt + 1))
        raise RuntimeError("unreachable request retry state")

    def list_objects(self, prefix: str, delimiter: str | None = None) -> list[str]:
        result: list[str] = []
        continuation = None
        while True:
            params = {"list-type": "2", "prefix": prefix}
            if delimiter:
                params["delimiter"] = delimiter
            if continuation:
                params["continuation-token"] = continuation
            document = ET.fromstring(self._request(LIST_BASE + "?" + urlencode(params)))
            selector = "s:CommonPrefixes/s:Prefix" if delimiter else "s:Contents/s:Key"
            result.extend(element.text for element in document.findall(selector, NAMESPACE) if element.text)
            if document.findtext("s:IsTruncated", namespaces=NAMESPACE) != "true":
                break
            continuation = document.findtext("s:NextContinuationToken", namespaces=NAMESPACE)
            if not continuation:
                raise ValueError("truncated archive listing without a continuation token")
        return result

    def historical_index(self, snapshot_path: Path, progress: Callable[[str], None] = print) -> dict[str, list[str]]:
        if snapshot_path.exists():
            return json.loads(snapshot_path.read_text())["monthly_keys"]
        prefixes = self.list_objects("data/futures/um/monthly/klines/", delimiter="/")
        natives = sorted(prefix.rstrip("/").split("/")[-1] for prefix in prefixes
                         if re.fullmatch(r"[A-Z0-9]+USDT", prefix.rstrip("/").split("/")[-1]))
        progress(f"Discovering archived 4h markets: {len(natives)} historical USDT identifiers")
        result: dict[str, list[str]] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.list_objects, f"data/futures/um/monthly/klines/{native}/4h/"): native
                       for native in natives}
            for index, future in enumerate(as_completed(futures), 1):
                native = futures[future]
                result[native] = sorted(key for key in future.result() if key.endswith(".zip"))
                if index % 100 == 0:
                    progress(f"Archived market discovery: {index}/{len(natives)}")
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps({
            "source": LIST_BASE, "created_at": datetime.now(timezone.utc).isoformat(),
            "monthly_keys": result,
        }, indent=2) + "\n")
        return result

    def download_key(self, key: str) -> tuple[bytes, dict[str, object]]:
        path = self.cache_root / key
        if path.exists():
            content = path.read_bytes()
        else:
            content = self._request(ARCHIVE_BASE + key)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".part")
            temporary.write_bytes(content)
            temporary.replace(path)
        checksum_path = path.with_suffix(path.suffix + ".CHECKSUM")
        if checksum_path.exists():
            checksum = checksum_path.read_text().split()[0]
        else:
            checksum_bytes = self._request(ARCHIVE_BASE + key + ".CHECKSUM")
            checksum = checksum_bytes.decode().split()[0]
            checksum_path.write_bytes(checksum_bytes)
        actual = hashlib.sha256(content).hexdigest()
        if actual != checksum:
            raise ValueError(f"SHA256 mismatch for {key}; remove the invalid cached archive before retrying")
        return content, {"key": key, "sha256": actual, "bytes": len(content)}

    def fetch_symbol(
        self, native: str, keys: list[str], start_ms: int, end_ms: int, output_path: Path,
    ) -> dict[str, object]:
        frames = []
        sources = []
        for key in keys:
            content, provenance = self.download_key(key)
            frame = parse_archive(content, native, ARCHIVE_BASE + key).filter(
                (pl.col("timestamp") >= start_ms) & (pl.col("timestamp") + BAR_MS <= end_ms)
            )
            if not frame.is_empty():
                frames.append(frame)
            sources.append(provenance)
        if not frames:
            return {"native_symbol": native, "rows": 0, "archives": sources}
        combined = pl.concat(frames).unique(subset="timestamp", keep="last").sort("timestamp")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(output_path)
        return {
            "native_symbol": native, "symbol": native_to_symbol(native), "rows": combined.height,
            "start_ms": combined["timestamp"].min(), "end_ms": combined["timestamp"].max(),
            "archives": sources, "path": str(output_path),
        }

    def fetch_many(
        self, requests: dict[str, list[str]], start_ms: int, end_ms: int, output_root: Path,
        progress: Callable[[str], None] = print,
    ) -> list[dict[str, object]]:
        output_root.mkdir(parents=True, exist_ok=True)
        results = []
        errors = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.fetch_symbol, native, keys, start_ms, end_ms,
                                   output_root / f"{native}.parquet"): native for native, keys in requests.items()}
            for index, future in enumerate(as_completed(futures), 1):
                try:
                    results.append(future.result())
                except Exception as error:
                    errors.append({"native_symbol": futures[future], "error": f"{type(error).__name__}: {error}"})
                if index % 20 == 0 or index == len(futures):
                    progress(f"Archive symbols: {index}/{len(futures)}; failures={len(errors)}")
        (output_root / "acquisition.json").write_text(json.dumps({
            "fetched_at": datetime.now(timezone.utc).isoformat(), "sources": results, "errors": errors,
        }, indent=2) + "\n")
        if errors:
            raise RuntimeError(f"{len(errors)} archive downloads failed; inspect {output_root / 'acquisition.json'}")
        return results
