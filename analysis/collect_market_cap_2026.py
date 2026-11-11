"""Supplement public CMC 2026 market caps for the fixed 2026 pool only."""
import gzip,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone,timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import polars as pl

ROOT=Path('/workspace/alpha-web3-research')
OUT=ROOT/'data/fundamentals/cmc_2026_supplement'
(OUT/'raw').mkdir(parents=True,exist_ok=True)
mapping=[a for a in json.loads((ROOT/'data/fundamentals/cmc_2021_2025/asset_mapping.json').read_text()) if 2025 in a['research_years']]
ids=sorted({i for a in mapping for i in a['cmc_ids']})
end=datetime.now(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)-timedelta(seconds=1)

def collect(identifier):
    path=OUT/'raw'/f'{identifier}_{end:%Y%m%d}.json.gz'
    if path.exists():
        with gzip.open(path,'rt') as f:return identifier,json.load(f)
    url='https://api.coinmarketcap.com/data-api/v3/cryptocurrency/historical?'+urlencode({'id':identifier,'convertId':2781,'timeStart':1767225600,'timeEnd':int(end.timestamp()),'interval':'1d'})
    for attempt in range(4):
        try:
            with urlopen(Request(url,headers={'User-Agent':'AlphaWeb3Research/1.0'}),timeout=45) as r:raw=r.read()
            result=json.loads(raw)
            if str(result.get('status',{}).get('error_code','0'))!='0':raise ValueError(str(result.get('status')))
            record={'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat(),'response_sha256':hashlib.sha256(raw).hexdigest(),'response':result}
            with gzip.open(path,'wt') as f:json.dump(record,f)
            return identifier,record
        except HTTPError as e:
            if e.code in {401,403} or attempt==3:raise
            time.sleep(60 if e.code==429 else 2**attempt)
        except Exception:
            if attempt==3:raise
            time.sleep(2**attempt)

assets,errors={},[]
with ThreadPoolExecutor(max_workers=4) as ex:
    fs={ex.submit(collect,i):i for i in ids}
    for n,f in enumerate(as_completed(fs),1):
        try:
            identifier,record=f.result();data=record['response'].get('data') or {};observed={}
            for quote in data.get('quotes',[]):
                day=quote['timeOpen'][:10];q=quote['quote'];cap=q.get('marketCap');price=q.get('close');supply=q.get('circulatingSupply')
                assert day.startswith('2026-') and day not in observed
                observed[day]={'cmc_id':identifier,'market_cap_usd':cap if cap and cap>0 else None,
                               'marketcap_identity_error':abs(cap-price*supply)/cap if cap and cap>0 and price and supply else None,
                               'source_url':record['url'],'retrieved_at':record['retrieved_at']}
            assets[identifier]=observed
        except Exception as e:errors.append({'id':fs[f],'error':repr(e)})
        if n%40==0 or n==len(fs):print(f'CMC2026 {n}/{len(fs)} errors={len(errors)}',flush=True)
if errors:
    (OUT/'errors.json').write_text(json.dumps(errors,indent=2));raise RuntimeError(str(errors[:3]))
rows=[]
for a in mapping:
    for day in sorted({d for i in a['cmc_ids'] for d in assets[i]}):
        candidates=[assets[i][day] for i in a['cmc_ids'] if day in assets[i]]
        valid=[r for r in candidates if r['market_cap_usd'] is not None and (r['marketcap_identity_error'] is None or r['marketcap_identity_error']<=.01)]
        r=(valid or candidates)[0]
        rows.append({'contract':a['contract'],'date':day,**r})
frame=pl.DataFrame(rows,infer_schema_length=None,schema_overrides={'market_cap_usd':pl.Float64,'marketcap_identity_error':pl.Float64}).sort('contract','date')
assert frame.select(pl.struct('contract','date').n_unique()).item()==frame.height
frame.write_parquet(OUT/'contract_market_cap_2026.parquet',compression='zstd')
(OUT/'summary.json').write_text(json.dumps({'pool_contracts':len(mapping),'provider_ids':len(ids),'rows':frame.height,'last_date':str(end.date()),'source':'public CMC history responses','missing':'no fill or current-supply reconstruction'},indent=2))
print((OUT/'summary.json').read_text(),flush=True)
