"""Collect public CMC historical observations with explicit contract/asset mappings."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path("/workspace/alpha-web3-research")
OUT = ROOT / "data/fundamentals/cmc_2021_2025"
BASE = "https://api.coinmarketcap.com/data-api/v3/"
STOP = threading.Event()
INDEX_CONTRACTS = {"BTCDOM", "DEFI", "BLUEBIRD", "FOOTBALL"}
ALIASES = {
    "BEAMX": ["BEAM"], "DODOX": ["DODO"], "RAYSOL": ["RAY"],
    "1MBABYDOGE": ["BABYDOGE"], "EOS": ["EOS", "A"],
    "LIT": ["LIT", "HEI"], "BNX": ["BNX", "FORM"],
}

# CMC numeric IDs distinguish unrelated assets sharing the same ticker.
# Lists preserve provider asset versions, newest first. No market caps are added.
IDENTITIES = {
    "1000000MOG": [27659], "1000BONK": [23095], "1000CAT": [32724],
    "1000CHEEMS": [33280], "1000FLOKI": [10804], "1000PEPE": [24478],
    "1000RATS": [28452], "1000SATS": [28194, 28683], "1000SHIB": [5994],
    "1MBABYDOGE": [10407], "ACE": [28674], "ACT": [33566], "ACX": [22620],
    "ADA": [2010], "AERO": [29270], "AI": [28846], "ALPHA": [7232],
    "ALT": [29073], "AMB": [2081], "ANT": [1680], "APE": [18876],
    "APT": [21794], "ARB": [11841], "ARK": [1586], "ATOM": [3794],
    "AVA": [2776], "AXL": [17799], "BAN": [33881], "BANANA": [28066],
    "BB": [30746], "BEAMX": [28298], "BEL": [6928], "BLUR": [23121],
    "BNX": [23635, 9891], "BOME": [29870], "BRETT": [29743], "BTC": [1],
    "CHESS": [10974], "CHILLGUY": [34125], "COMBO": [4275], "COMP": [5692],
    "COS": [4036], "COW": [19269], "CYBER": [24781], "DAR": [11374],
    "DASH": [131], "DEGEN": [30096], "DOGE": [74], "DOGS": [32698],
    "DYDX": [28324, 11156], "EDU": [24613], "EOS": [36462, 1765],
    "ETH": [1027], "FARTCOIN": [33597], "FLUX": [3029], "G": [32120],
    "GAL": [11877], "GAS": [1785], "GMT": [18069], "GOAT": [33440],
    "GRT": [6719], "GTC": [10052], "HFT": [22461], "HIPPO": [33258],
    "HNT": [5665], "HOT": [2682], "ID": [21846], "IMX": [10603],
    "JOE": [11396], "JUP": [29210], "KNC": [9444, 1982], "LINA": [7102],
    "LIT": [35724, 6833], "LUNA2": [20314], "MASK": [8536], "MAV": [18037],
    "ME": [32197], "MEME": [28301], "MOODENG": [33093], "MOVE": [32452],
    "NEIRO": [32521], "NTRN": [26680], "OMG": [1808], "ONE": [3945],
    "ONG": [3217], "ORCA": [11165], "PENGU": [34466], "PEOPLE": [14806],
    "PHB": [13969, 2112], "PIXEL": [29335], "PNUT": [33788], "POL": [28321],
    "POPCAT": [28782], "PORTAL": [29555], "RARE": [11294], "RIF": [3701],
    "RONIN": [14101], "RUNE": [4157], "SAFE": [21585], "SCR": [26998],
    "SOL": [5426], "SRM": [6187], "STRK": [22691], "STX": [4847],
    "SUI": [20947], "SUN": [10529, 6990], "SUPER": [8290], "SYN": [12147],
    "TAO": [22974], "THE": [23335], "TON": [11419], "TRU": [7725],
    "TURBO": [24911], "UNFI": [7672], "UNI": [7083], "USTC": [7129],
    "VELODROME": [20435], "WIF": [28752], "XAI": [28933], "XRP": [52],
    "XVG": [693], "ZEC": [1437], "ZEN": [1698],
}


def timestamp(year: int) -> int:
    return int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())


def request_cached(url: str, path: Path) -> dict:
    if path.exists():
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    for attempt in range(4):
        if STOP.is_set():
            raise RuntimeError("collection stopped after access denial")
        try:
            with urlopen(Request(url, headers={"User-Agent": "AlphaWeb3Research/1.0"}), timeout=45) as response:
                raw = response.read()
            result = json.loads(raw)
            status = result.get("status", {})
            if str(status.get("error_code", "0")) != "0":
                raise ValueError(f"provider status {status}")
            envelope = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "response_sha256": hashlib.sha256(raw).hexdigest(), "response": result}
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".part")
            with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                json.dump(envelope, stream, ensure_ascii=False)
            temporary.replace(path)
            return envelope
        except HTTPError as error:
            if error.code in {401, 403}:
                STOP.set()
                raise
            if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
            delay = max(60, int(error.headers.get("Retry-After", "60"))) if error.code == 429 else 2 ** (attempt + 1)
        except (URLError, TimeoutError, ValueError) as error:
            if attempt == 3:
                raise
            delay = 2 ** (attempt + 1)
        time.sleep(delay)
    raise RuntimeError("request retry exhausted")


def scope() -> list[dict]:
    import polars as pl
    membership: dict[str, list[int]] = {}
    for year in range(2021, 2026):
        path = ROOT / ("data/research/universe_full_2024/selected.csv" if year == 2024
                       else f"data/research/annual_universes/{year + 1}/selected.csv")
        for symbol in pl.read_csv(path)["symbol"]:
            membership.setdefault(symbol, []).append(year)
    rows = []
    for contract, years in sorted(membership.items()):
        base = contract.split("/")[0]
        multiplier, underlying = 1, base
        if base.startswith("1000000"):
            multiplier, underlying = 1_000_000, base[7:]
        elif base.startswith("1000"):
            multiplier, underlying = 1000, base[4:]
        elif base == "1MBABYDOGE":
            multiplier, underlying = 1_000_000, "BABYDOGE"
        aliases = ALIASES.get(base, [underlying])
        if base != underlying:
            aliases = sorted(set([*aliases, base]))
        rows.append({"contract": contract, "base": base, "underlying": underlying,
                     "contract_multiplier": multiplier, "research_years": years,
                     "aliases": [alias.upper() for alias in aliases],
                     "instrument_type": "index" if base in INDEX_CONTRACTS else "token"})
    return rows


def bootstrap() -> None:
    assets = scope()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scope.json").write_text(json.dumps(assets, indent=2))
    dates = ["2021-01-01", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]

    def snapshot(date: str) -> tuple[str, list[dict]]:
        url = BASE + "cryptocurrency/listings/historical?" + urlencode(
            {"date": date, "start": 1, "limit": 5000, "convert": "USD"})
        result = request_cached(url, OUT / "raw/catalog" / f"{date}.json.gz")
        return date, result["response"]["data"]

    catalogs = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        for date, records in executor.map(snapshot, dates):
            catalogs[date] = records
            print(f"Catalog {date}: {len(records)} records", flush=True)
    candidates = []
    for asset in assets:
        matches = {}
        if asset["instrument_type"] != "index":
            for date, records in catalogs.items():
                for record in records:
                    if record["symbol"].upper() not in asset["aliases"]:
                        continue
                    cmc_id = str(record["id"])
                    item = matches.setdefault(cmc_id, {"cmc_id": record["id"], "names": [], "slugs": [], "symbols": [], "anchors": []})
                    for field, value in (("names", record["name"]), ("slugs", record["slug"]), ("symbols", record["symbol"])):
                        if value not in item[field]:
                            item[field].append(value)
                    quote = next((quote for quote in record.get("quotes", []) if quote.get("name") == "2781"), {})
                    item["anchors"].append({"date": date, "price_usd": quote.get("price"),
                                            "market_cap_usd": quote.get("marketCap"),
                                            "circulating_supply": record.get("circulatingSupply")})
        candidates.append({**asset, "candidates": list(matches.values())})
    (OUT / "mapping_candidates.json").write_text(json.dumps(candidates, indent=2))
    print("Candidate counts", {number: sum(len(row["candidates"]) == number for row in candidates)
                               for number in sorted({len(row["candidates"]) for row in candidates})}, flush=True)
    for row in candidates:
        if len(row["candidates"]) != 1:
            print(row["base"], [(candidate["cmc_id"], candidate["names"], candidate["slugs"]) for candidate in row["candidates"]], flush=True)


def collect() -> None:
    mapping = json.loads((OUT / "asset_mapping.json").read_text())
    identifiers = sorted({identifier for row in mapping for identifier in row.get("cmc_ids", [])})

    def history(cmc_id: int, year: int) -> dict:
        params = {"id": cmc_id, "convertId": 2781, "timeStart": timestamp(year),
                  "timeEnd": timestamp(year + 1) - 1, "interval": "1d"}
        url = BASE + "cryptocurrency/historical?" + urlencode(params)
        envelope = request_cached(url, OUT / "raw/history" / str(cmc_id) / f"{year}.json.gz")
        data = envelope["response"].get("data") or {}
        quotes = data.get("quotes", [])
        dates = [quote["timeOpen"][:10] for quote in quotes]
        if len(dates) != len(set(dates)) or any(not date.startswith(str(year)) for date in dates):
            raise ValueError(f"invalid dates for {cmc_id}/{year}")
        if any(quote["timeOpen"][11:19] != "00:00:00" or quote["timeClose"][11:19] != "23:59:59" for quote in quotes):
            raise ValueError(f"non-daily aggregation for {cmc_id}/{year}")
        return {"cmc_id": cmc_id, "year": year, "rows": len(quotes), "url": url,
                "response_sha256": envelope["response_sha256"]}

    jobs = [(cmc_id, year) for cmc_id in identifiers for year in range(2021, 2026)]
    results, errors = [], []
    print(f"Fetching {len(identifiers)} unique CMC ids, {len(jobs)} asset-year requests", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(history, cmc_id, year): (cmc_id, year) for cmc_id, year in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append({"job": futures[future], "error": repr(error)})
                print("ERROR", errors[-1], flush=True)
            if number % 40 == 0 or number == len(jobs):
                print(f"Historical requests {number}/{len(jobs)}, errors={len(errors)}", flush=True)
    (OUT / "acquisition.json").write_text(json.dumps({"results": results, "errors": errors}, indent=2))
    if errors:
        raise RuntimeError(f"{len(errors)} historical requests failed")


def create_mapping() -> None:
    candidates = json.loads((OUT / "mapping_candidates.json").read_text())
    registry = {}
    for path in sorted((OUT / "raw/catalog").glob("*.gz")):
        with gzip.open(path, "rt") as stream:
            snapshot = json.load(stream)
        for record in snapshot["response"]["data"]:
            registry[record["id"]] = {key: record.get(key) for key in ("id", "name", "symbol", "slug", "dateAdded")}
    rows = []
    for asset in candidates:
        base = asset["base"]
        if asset["instrument_type"] == "index":
            identifiers, method = [], "index_has_no_token_supply"
        elif base in IDENTITIES:
            identifiers, method = IDENTITIES[base], "explicit_project_identity"
        elif len(asset["candidates"]) == 1:
            identifiers = [asset["candidates"][0]["cmc_id"]]
            method = "unique_historical_symbol_candidate"
        else:
            raise ValueError(f"unresolved asset identity: {base}")
        if any(identifier not in registry for identifier in identifiers):
            raise ValueError(f"identity not present in source catalogs: {base}/{identifiers}")
        rows.append({key: value for key, value in asset.items() if key != "candidates"} | {
            "cmc_ids": identifiers, "mapping_method": method,
            "provider_assets": [registry[identifier] for identifier in identifiers],
            "source_unit_multipliers": {str(identifier): 1000 if identifier == 28683 else 1 for identifier in identifiers},
            "version_selection": ("prefer a mapped version reporting cap and supply on that date; preserve provider id and units"
                                  if base in {"1000SATS", "DYDX"} else
                                  "prefer first listed version with an observation on that date; preserve selected provider id and units"),
        })
    (OUT / "asset_mapping.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Mapped {len(rows)} contracts, {len({i for r in rows for i in r['cmc_ids']})} provider asset versions", flush=True)
    for row in rows:
        if len(row["cmc_ids"]) > 1:
            print("VERSIONS", row["base"], row["cmc_ids"], flush=True)


def positive(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def supplement() -> None:
    import polars as pl
    frame = pl.read_parquet(OUT / "contract_daily.parquet")
    target = frame.filter(
        ((pl.col("contract") == "1000SATS/USDT:USDT")
         & (pl.col("status") == "provider_missing_market_cap_or_supply"))
        | ((pl.col("contract") == "SUN/USDT:USDT")
           & (pl.col("status") == "before_provider_history"))
        | (pl.col("in_research_pool") & pl.col("contract").is_in([
            "BAND/USDT:USDT", "CRV/USDT:USDT", "LINK/USDT:USDT", "NEAR/USDT:USDT"])
           & ~pl.col("has_market_cap_and_supply"))
    )
    dates = sorted({str(day) for day in target["date"].to_list()} | {"2025-01-01", "2025-04-01"})

    def fetch(day: str) -> tuple[str, int]:
        params = {"date": day, "start": 1, "limit": 1000, "convert": "USD"}
        envelope = request_cached(BASE + "cryptocurrency/listings/historical?" + urlencode(params),
                                  OUT / "raw/supplemental_snapshots" / f"{day}.json.gz")
        return day, len(envelope["response"]["data"])

    errors = []
    print(f"Supplementing {len(dates)} public daily snapshots", flush=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch, day): day for day in dates}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:
                errors.append({"date": futures[future], "error": repr(error)})
            if number % 30 == 0 or number == len(dates):
                print(f"Supplement snapshots {number}/{len(dates)}, errors={len(errors)}", flush=True)
    (OUT / "supplement_acquisition.json").write_text(json.dumps({"dates": dates, "errors": errors}, indent=2))
    if errors:
        raise RuntimeError("supplemental snapshot acquisition failed")


def normalize() -> None:
    import polars as pl
    mapping = json.loads((OUT / "asset_mapping.json").read_text())
    acquisition = json.loads((OUT / "acquisition.json").read_text())
    if acquisition["errors"]:
        raise ValueError("acquisition contains errors; complete collection before normalization")
    identifiers = sorted({identifier for asset in mapping for identifier in asset["cmc_ids"]})
    provider_rows, by_asset = [], {}
    for identifier in identifiers:
        observations = {}
        for year in range(2021, 2026):
            path = OUT / "raw/history" / str(identifier) / f"{year}.json.gz"
            with gzip.open(path, "rt") as stream:
                envelope = json.load(stream)
            asset = envelope["response"].get("data") or {}
            if asset.get("quotes") and int(asset["id"]) != identifier:
                raise ValueError(f"provider id mismatch for {identifier}/{year}")
            for item in asset.get("quotes", []):
                quote = item["quote"]
                observed_date = date.fromisoformat(item["timeOpen"][:10])
                if observed_date.year != year:
                    raise ValueError("observation outside requested year")
                if observed_date in observations:
                    raise ValueError("duplicate provider date")
                close = datetime.fromisoformat(item["timeClose"].replace("Z", "+00:00"))
                row = {
                    "date": observed_date, "cmc_id": identifier,
                    "provider_symbol": asset.get("symbol"), "provider_name": asset.get("name"),
                    "provider_price_usd": quote.get("close"),
                    "provider_market_cap_usd": quote.get("marketCap"),
                    "provider_circulating_supply": quote.get("circulatingSupply"),
                    "provider_volume_24h_usd": quote.get("volume"),
                    "period_end_utc": close,
                    "usable_not_before_utc": close + timedelta(milliseconds=1),
                    "retrieved_at": envelope["retrieved_at"], "source_url": envelope["url"],
                }
                observations[observed_date] = row
        by_asset[identifier] = observations
    for path in sorted((OUT / "raw/supplemental_snapshots").glob("*.gz")):
        with gzip.open(path, "rt") as stream:
            envelope = json.load(stream)
        observed_date = date.fromisoformat(path.name[:10])
        for asset in envelope["response"]["data"]:
            identifier = asset["id"]
            if identifier not in by_asset:
                continue
            quote = next((quote for quote in asset.get("quotes", []) if quote.get("name") == "2781"), {})
            current = by_asset[identifier].get(observed_date)
            complete = positive(quote.get("marketCap")) and positive(asset.get("circulatingSupply"))
            if current is not None and (positive(current["provider_market_cap_usd"])
                                        and positive(current["provider_circulating_supply"]) or not complete):
                continue
            end = datetime.combine(observed_date + timedelta(days=1), datetime.min.time(), timezone.utc)
            by_asset[identifier][observed_date] = {
                "date": observed_date, "cmc_id": identifier,
                "provider_symbol": asset.get("symbol"), "provider_name": asset.get("name"),
                "provider_price_usd": quote.get("price"), "provider_market_cap_usd": quote.get("marketCap"),
                "provider_circulating_supply": asset.get("circulatingSupply"),
                "provider_volume_24h_usd": quote.get("volume24h"),
                "period_end_utc": end - timedelta(milliseconds=1), "usable_not_before_utc": end,
                "retrieved_at": envelope["retrieved_at"], "source_url": envelope["url"],
            }
    provider_rows = [row for observations in by_asset.values() for row in observations.values()]
    provider_schema = {column: pl.Float64 for column in (
        "provider_price_usd", "provider_market_cap_usd", "provider_circulating_supply", "provider_volume_24h_usd")}
    provider = pl.DataFrame(provider_rows, schema_overrides=provider_schema).sort(["cmc_id", "date"])
    assert provider.select(pl.struct("cmc_id", "date").n_unique()).item() == provider.height
    provider.write_parquet(OUT / "provider_assets_daily.parquet", compression="zstd")
    dates = [date(2021, 1, 1) + timedelta(days=offset) for offset in range(1826)]
    rows, coverage, map_rows = [], [], []
    for asset in mapping:
        identifiers = asset["cmc_ids"]
        observed_dates = sorted({day for identifier in identifiers for day in by_asset[identifier]})
        first = observed_dates[0] if observed_dates else None
        last = observed_dates[-1] if observed_dates else None
        asset_rows = []
        for day in dates:
            candidates = [by_asset[identifier][day] for identifier in identifiers if day in by_asset[identifier]]
            if asset["base"] in {"1000SATS", "DYDX"}:
                # Preserve the reported historical presentation during unit/chain migration.
                complete = [row for row in candidates if positive(row["provider_market_cap_usd"])
                            and positive(row["provider_circulating_supply"])]
                candidates = complete or candidates
            source = candidates[0] if candidates else None
            in_pool = day.year in asset["research_years"]
            row = {"date": day, "contract": asset["contract"], "underlying": asset["underlying"],
                   "contract_multiplier": asset["contract_multiplier"], "in_research_pool": in_pool,
                   "research_pool_year": day.year + 1 if in_pool else None,
                   "cmc_id": None, "provider_symbol": None, "provider_name": None,
                   "market_cap_usd": None, "circulating_supply_tokens": None,
                   "price_usd_per_token": None, "contract_equivalent_supply": None,
                   "volume_24h_usd": None, "provider_unit_multiplier": None,
                   "period_end_utc": None, "usable_not_before_utc": None,
                   "retrieved_at": None, "source_url": None,
                   "marketcap_identity_error": None,
                   "has_market_cap_and_supply": False,
                   "asset_version_mapping": len(identifiers) > 1}
            # Provider histories can be restated in replacement-token units before
            # the actual exchange migration. Do not silently approve turnover use.
            row["supply_unit_review_required"] = asset["base"] in {"SUN", "BNX", "PHB", "EOS", "LIT", "KNC", "DYDX"}
            row["supply_unit_basis"] = "selected_provider_asset_version"
            if source:
                scale = asset["source_unit_multipliers"][str(source["cmc_id"])]
                cap = positive(source["provider_market_cap_usd"])
                supply = positive(source["provider_circulating_supply"])
                price = positive(source["provider_price_usd"])
                row.update({key: source[key] for key in (
                    "cmc_id", "provider_symbol", "provider_name", "period_end_utc",
                    "usable_not_before_utc", "retrieved_at", "source_url")})
                native_supply = supply * scale if supply is not None else None
                row.update({"market_cap_usd": cap, "circulating_supply_tokens": native_supply,
                            "price_usd_per_token": price / scale if price is not None else None,
                            "contract_equivalent_supply": native_supply / asset["contract_multiplier"] if native_supply is not None else None,
                            "volume_24h_usd": positive(source["provider_volume_24h_usd"]),
                            "provider_unit_multiplier": scale,
                            "has_market_cap_and_supply": cap is not None and supply is not None,
                            "marketcap_identity_error": abs(cap - price * supply) / cap if cap and price and supply else None})
                row["status"] = "available" if row["has_market_cap_and_supply"] else "provider_missing_market_cap_or_supply"
            elif asset["instrument_type"] == "index":
                row["status"] = "not_applicable_index_contract"
            elif first is None:
                row["status"] = "no_provider_history"
            elif day < first:
                row["status"] = "before_provider_history"
            elif day > last:
                row["status"] = "after_provider_history"
            else:
                row["status"] = "provider_history_gap"
            asset_rows.append(row)
        required = [row for row in asset_rows if row["in_research_pool"]]
        valid = [row for row in asset_rows if row["has_market_cap_and_supply"]]
        required_valid = [row for row in required if row["has_market_cap_and_supply"]]
        coverage.append({"contract": asset["contract"], "underlying": asset["underlying"],
                         "instrument_type": asset["instrument_type"], "cmc_ids": ";".join(map(str, identifiers)),
                         "provider_asset_names": ";".join(str(item["name"]) for item in asset["provider_assets"]),
                         "days_with_market_cap": sum(row["market_cap_usd"] is not None for row in asset_rows),
                         "days_with_circulating_supply": sum(row["circulating_supply_tokens"] is not None for row in asset_rows),
                         "days_with_both": len(valid), "research_required_days": len(required),
                         "research_days_with_both": len(required_valid),
                         "research_coverage": len(required_valid) / len(required),
                         "first_provider_observation": first, "last_provider_observation": last,
                         "provider_missing_field_days": sum(row["status"] == "provider_missing_market_cap_or_supply" for row in asset_rows),
                         "identity_error_over_1pct_days": sum(row["marketcap_identity_error"] is not None and row["marketcap_identity_error"] > .01 for row in asset_rows),
                         "asset_version_mapping": len(identifiers) > 1})
        map_rows.append({"contract": asset["contract"], "underlying": asset["underlying"],
                         "contract_multiplier": asset["contract_multiplier"], "cmc_ids": ";".join(map(str, identifiers)),
                         "mapping_method": asset["mapping_method"],
                         "provider_asset_names": ";".join(str(item["name"]) for item in asset["provider_assets"]),
                         "version_selection": asset["version_selection"]})
        rows.extend(asset_rows)
    float_columns = ("market_cap_usd", "circulating_supply_tokens", "price_usd_per_token",
                     "contract_equivalent_supply", "volume_24h_usd", "marketcap_identity_error")
    frame = pl.DataFrame(rows, infer_schema_length=None,
                         schema_overrides={column: pl.Float64 for column in float_columns}).sort(["contract", "date"])
    assert frame.height == len(mapping) * len(dates)
    assert frame.select(pl.struct("contract", "date").n_unique()).item() == frame.height
    assert frame.filter(pl.col("has_market_cap_and_supply") & ((pl.col("market_cap_usd") <= 0) | (pl.col("circulating_supply_tokens") <= 0))).is_empty()
    frame.write_parquet(OUT / "contract_daily.parquet", compression="zstd")
    with gzip.open(OUT / "contract_daily.csv.gz", "wb", compresslevel=4) as stream:
        frame.write_csv(stream)
    cover = pl.DataFrame(coverage).sort("contract")
    cover.write_csv(OUT / "coverage.csv")
    pl.DataFrame(map_rows).sort("contract").write_csv(OUT / "asset_mapping.csv")
    required = frame.filter(pl.col("in_research_pool"))
    token_required = required.filter(pl.col("status") != "not_applicable_index_contract")
    yearly = token_required.group_by(pl.col("date").dt.year().alias("year")).agg(
        pl.len().alias("required_token_days"), pl.col("has_market_cap_and_supply").sum().alias("available_token_days"),
    ).sort("year").with_columns((pl.col("available_token_days") / pl.col("required_token_days")).alias("coverage"))
    yearly.write_csv(OUT / "yearly_coverage.csv")
    summary = {
        "contract_count": len(mapping), "token_contracts": sum(asset["instrument_type"] == "token" for asset in mapping),
        "index_contracts": sorted(asset["contract"] for asset in mapping if asset["instrument_type"] == "index"),
        "provider_asset_versions": provider["cmc_id"].n_unique(),
        "period": ["2021-01-01", "2025-12-31"], "frequency": "daily UTC, end of day",
        "grid_rows": frame.height, "provider_observed_rows": provider.height,
        "rows_with_both": frame["has_market_cap_and_supply"].sum(),
        "token_contracts_with_any_data": cover.filter((pl.col("instrument_type") == "token") & (pl.col("days_with_both") > 0)).height,
        "token_contracts_full_research_coverage": cover.filter((pl.col("instrument_type") == "token") & (pl.col("research_coverage") == 1)).height,
        "research_token_rows": token_required.height,
        "research_token_rows_with_both": token_required["has_market_cap_and_supply"].sum(),
        "yearly": yearly.to_dicts(),
        "source": "CoinMarketCap public historical market pages and their public data responses",
        "source_example": "https://coinmarketcap.com/historical/20210101/",
        "supply": "source circulatingSupply field; no market-cap/price derivation and no total-supply substitution",
        "unit_handling": "native token units; provider CMC id 28683 uses a 1000-SATS unit; contract-equivalent supply divides native supply by contract multiplier",
        "missing": "no interpolation, no forward/back fill, no present-day supply backfill; source zeros retained in provider_assets_daily but missing in normalized columns",
        "versioning": "multiple CMC ids for token migrations are preserved; preferred id is first with a same-day observation; SATS and DYDX presentations prefer reported cap and supply",
        "timestamp": "usable_not_before_utc is next UTC midnight, a period-end availability bound, not a verified original publication timestamp",
        "point_in_time": "retrieved historical data can contain provider revisions; original publication vintage is not verified",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("bootstrap", "map", "collect", "supplement", "normalize"))
    args = parser.parse_args()
    if args.mode == "bootstrap":
        bootstrap()
    elif args.mode == "map":
        create_mapping()
    elif args.mode == "collect":
        collect()
    elif args.mode == "supplement":
        supplement()
    else:
        normalize()


if __name__ == "__main__":
    main()
