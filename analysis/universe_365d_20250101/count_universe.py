import json, math
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import polars as pl
from alpha_web3_research.data.binance_archive import BinanceArchiveClient,native_to_symbol,BAR_MS
from alpha_web3_research.universe.point_in_time import build_point_in_time_universe,PointInTimeUniverseRules
from alpha_web3_research.universe.rules import is_stable_base,is_non_crypto_base

out=Path("/workspace/alpha-web3-research/data/research/universe_365d_20250101")
out.mkdir(parents=True,exist_ok=True)
cache=Path("/tmp/alpha-universe-365")
client=BinanceArchiveClient(cache/"archives",24)
index=client.historical_index(out/"archive_index.json",lambda x:print(x,flush=True))
asof=int(datetime(2025,1,1,tzinfo=timezone.utc).timestamp()*1000)
start=asof-365*86400000
requests={}
for native,keys in index.items():
    base=native[:-4]
    if is_stable_base(base) or is_non_crypto_base(base) or base in {"PAXG","XAUT"} or base.endswith(("BULL","BEAR","2L","2S","3L","3S")):
        continue
    chosen=[k for k in keys if "-4h-2024-" in k]
    if any(k.endswith("-2024-01.zip") for k in chosen):
        requests[native]=chosen
print("Candidates with January 2024 archives:",len(requests),flush=True)
# Only candidates observed by the 365-day boundary can qualify.
def initial(item):
    native,keys=item
    key=next(k for k in keys if k.endswith("-2024-01.zip"))
    from alpha_web3_research.data.binance_archive import parse_archive
    content,_=client.download_key(key)
    f=parse_archive(content,native,key)
    return native, f["timestamp"].min()<=start
qualified={}
with ThreadPoolExecutor(max_workers=24) as pool:
    for future in as_completed([pool.submit(initial,item) for item in requests.items()]):
        native,ok=future.result()
        if ok: qualified[native]=requests[native]
print("Candidates meeting initial history boundary:",len(qualified),flush=True)
client.fetch_many(qualified,int(datetime(2024,1,1,tzinfo=timezone.utc).timestamp()*1000),asof,cache/"bars",lambda x:print(x,flush=True))
bars=pl.concat([pl.read_parquet(p) for p in sorted((cache/"bars").glob("*.parquet"))])
audit=build_point_in_time_universe(bars,asof,PointInTimeUniverseRules(365,30,0,0.98,10000))
rows=audit.to_dicts()
for row in rows:
    reasons=[r for r in row["exclusion_reason"].split(",") if r and r!="incomplete_liquidity_window"]
    row["exclusion_reason"]=",".join(reasons)
    row["eligible"]=not reasons
eligible=sorted([r for r in rows if r["eligible"]],key=lambda r: (-(r["median_daily_quote_volume"] or 0),r["symbol"]))
ranks={r["symbol"]:i+1 for i,r in enumerate(eligible)}
for row in rows:
    row["liquidity_rank"]=ranks.get(row["symbol"])
    row["included"]=row["eligible"] and row["liquidity_rank"]<=120
    row["history_coverage"]=(row["history_bars"] or 0)/2190
result=pl.DataFrame(rows).sort(["included","liquidity_rank","symbol"],descending=[True,False,False],nulls_last=True)
result.write_csv(out/"universe_audit.csv")
result.filter(pl.col("included")).write_csv(out/"selected.csv")
summary={"as_of_utc":"2025-01-01T00:00:00Z","window_start_utc":datetime.fromtimestamp(start/1000,timezone.utc).isoformat(),"required_unique_4h_bars":math.ceil(365*6*.98),"history_boundary_candidates":len(qualified),"eligible_before_cap":len(eligible),"selected":min(120,len(eligible)),"incomplete_30d_among_eligible":sum(r["complete_liquidity_days"]<30 for r in eligible),"top_symbols":[r["symbol"] for r in eligible[:120]],"excluded":[{"symbol":r["symbol"],"reason":r["exclusion_reason"]} for r in rows if not r["eligible"]],"rules":{"days":365,"coverage":.98,"max_symbols":120,"min_daily_quote_volume":None,"require_30_complete_days":False,"ranking":"30 UTC days median daily quote_volume on complete days"}}
(out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
