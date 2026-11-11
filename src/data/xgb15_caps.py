"""CMC daily observations adapted from research's collect_market_cap_2026.py."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

CAP_COLUMNS = ["symbol", "date", "cmc_id", "market_cap_usd", "marketcap_identity_error"]
BAR_MS = 14_400_000


def current_caps_path(folder: Path) -> Path:
    live = folder / "runtime/caps.parquet"
    return live if live.exists() else folder / "caps.parquet"


def required_cap_date(decision_timestamp: int) -> pd.Timestamp:
    close = pd.Timestamp(decision_timestamp + BAR_MS, unit="ms", tz="UTC")
    return (close - pd.Timedelta(hours=24)).floor("D").tz_localize(None) - pd.Timedelta(days=1)


def normalize_caps(frame: pd.DataFrame) -> pd.DataFrame:
    if not set(CAP_COLUMNS).issubset(frame.columns):
        raise ValueError(f"caps require columns {CAP_COLUMNS}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.date, utc=True).dt.tz_localize(None)
    if frame.date.isna().any() or not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError("cap dates must be UTC calendar days")
    if not frame.symbol.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("invalid cap symbols")
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("duplicate cap observations")
    for column in CAP_COLUMNS[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    ids = frame.cmc_id.dropna()
    if not ids.map(lambda value: math.isfinite(value) and value > 0 and value == int(value)).all():
        raise ValueError("invalid CMC IDs")
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def cap_coverage(frame: pd.DataFrame, pairs: list[str], stamp: int) -> dict:
    frame = normalize_caps(frame)
    last = required_cap_date(stamp)
    dates = pd.date_range(last - pd.Timedelta(days=39), last)
    complete, incomplete = [], []
    for pair in pairs:
        history = frame.loc[frame.symbol == pair].set_index("date").reindex(dates)
        valid = history.market_cap_usd.map(lambda value: pd.notna(value) and math.isfinite(value) and value > 0)
        valid &= ~(history.marketcap_identity_error > .01)
        if valid.all() and history.cmc_id.notna().all() and history.cmc_id.nunique() == 1:
            complete.append(pair)
        else:
            incomplete.append(pair)
    return {"required_date": str(last.date()), "window_start": str(dates[0].date()),
            "complete_count": len(complete), "pool_count": len(pairs), "incomplete_pairs": incomplete}


def read_mapping(path: Path, pairs: list[str]) -> dict[str, list[int]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("asset_mapping.json must be the research mapping list")
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each CMC mapping record must be an object")
        pair = record.get("contract")
        if pair not in pairs:
            continue
        ids = record.get("cmc_ids")
        if pair in result or not isinstance(ids, list):
            raise ValueError("missing or duplicate CMC mapping")
        if not ids:
            if record.get("instrument_type") != "index":
                raise ValueError("token contract requires a CMC ID")
            result[pair] = []
            continue
        if any(type(identifier) is not int or identifier <= 0 for identifier in ids) or len(set(ids)) != len(ids):
            raise ValueError("CMC IDs must be ordered unique positive integers")
        result[pair] = ids
    if set(result) != set(pairs):
        raise ValueError("CMC mapping must cover the complete annual pool; do not guess ticker identities")
    return result


def _positive(value) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (ValueError, TypeError):
        return None


def parse_history(payload: dict, identifier: int, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    if str(payload.get("status", {}).get("error_code", "0")) != "0":
        raise ValueError("CMC returned an error status")
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("quotes"), list):
        raise ValueError("CMC response has no historical quotes")
    if data.get("id") is not None and int(data["id"]) != identifier:
        raise ValueError("CMC response identity mismatch")
    observed = {}
    for item in data["quotes"]:
        opened = pd.Timestamp(item["timeOpen"])
        closed = pd.Timestamp(item["timeClose"])
        if opened.tzinfo is None or closed.tzinfo is None:
            raise ValueError("CMC timestamps must include timezone")
        opened, closed = opened.tz_convert("UTC"), closed.tz_convert("UTC")
        if opened != opened.floor("D") or closed.floor("s") != opened + pd.Timedelta(days=1) - pd.Timedelta(seconds=1):
            raise ValueError("CMC response is not a completed daily observation")
        day = opened.tz_localize(None)
        if not start <= day <= end or day in observed:
            raise ValueError("duplicate or out-of-range CMC date")
        quote = item["quote"]
        cap, price, supply = (_positive(quote.get(key)) for key in ("marketCap", "close", "circulatingSupply"))
        observed[day] = {"cmc_id": identifier, "market_cap_usd": cap,
                         "marketcap_identity_error": abs(cap - price * supply) / cap if cap and price and supply else None}
    return observed


def fetch_history(url: str) -> dict:
    # No authentication workaround, alternate endpoint, or retry after access denial.
    with urlopen(Request(url, headers={"User-Agent": "AlphaWeb3Research/1.0"}), timeout=30) as response:
        return json.load(response)


def atomic_caps(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".parquet")
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def refresh_caps(folder: Path, pairs: list[str], stamp: int, now: datetime, fetcher=fetch_history) -> dict:
    """Refresh observed data only; failed requests leave the previous Parquet intact."""
    mapping = read_mapping(folder / "asset_mapping.json", pairs)
    start = required_cap_date(stamp) - pd.Timedelta(days=39)
    end = pd.Timestamp(now).tz_convert("UTC").floor("D").tz_localize(None) - pd.Timedelta(days=1)
    if start > end:
        raise ValueError("no completed CMC daily history is available")
    observations = {}
    for identifier in sorted({identifier for ids in mapping.values() for identifier in ids}):
        params = {"id": identifier, "convertId": 2781, "timeStart": int(start.tz_localize("UTC").timestamp()),
                  "timeEnd": int((end.tz_localize("UTC") + pd.Timedelta(days=1)).timestamp()) - 1, "interval": "1d"}
        url = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/historical?" + urlencode(params)
        payload = fetcher(url)
        observations[identifier] = parse_history(payload, identifier, start, end)
        raw = json.dumps(payload, ensure_ascii=False).encode()
        archive = folder / "raw" / f"{identifier}_{start.date()}_{end.date()}_{hashlib.sha256(raw).hexdigest()[:12]}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(json.dumps({"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                                      "response_sha256": hashlib.sha256(raw).hexdigest(), "response": payload}, ensure_ascii=False))
    rows = []
    for pair, ids in mapping.items():
        for day in sorted({day for identifier in ids for day in observations[identifier]}):
            candidates = [observations[identifier][day] for identifier in ids if day in observations[identifier]]
            valid = [row for row in candidates if row["market_cap_usd"] is not None
                     and (row["marketcap_identity_error"] is None or row["marketcap_identity_error"] <= .01)]
            rows.append({"symbol": pair, "date": day, **(valid or candidates)[0]})
    if not rows:
        raise ValueError("CMC returned no observations; previous caps preserved")
    fresh = normalize_caps(pd.DataFrame(rows))
    source_path = current_caps_path(folder)
    if source_path.exists():
        # Freeze previously observed history; do not silently rewrite it with revisions.
        old = normalize_caps(pd.read_parquet(source_path))
        fresh = normalize_caps(pd.concat([old, fresh], ignore_index=True).drop_duplicates(["symbol", "date"], keep="first"))
    atomic_caps(folder / "runtime/caps.parquet", fresh)
    return cap_coverage(fresh, pairs, stamp)
