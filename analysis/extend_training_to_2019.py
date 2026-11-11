"""Append 2019-2020 observations to five existing 15-factor training tables."""
import sys
sys.path.append('/workspace/alpha-web3-research/.venv/lib/python3.12/site-packages')
import json,types,hashlib,io,zipfile,csv,gzip
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd
import numpy as np
import requests
ROOT=Path('/tmp/extend2019');ROOT.mkdir(exist_ok=True)
OLD=Path('/tmp/alpha15-run/data/model_tables/factors15_2021_2026')
OUT=ROOT/'data/model_tables/factors15_2019_2026';OUT.mkdir(parents=True,exist_ok=True)
RET=ROOT/'data/returns/annual_pools_2019_2020';RET.mkdir(parents=True,exist_ok=True)
CAP=ROOT/'data/fundamentals/cmc_2019_2020';CAP.mkdir(parents=True,exist_ok=True)
def module(name,file,replacements={}):
    m=types.ModuleType(name);m.__file__=str(ROOT/'analysis'/file);sys.modules[name]=m
    code=Path(m.__file__).read_text()
    for a,b in replacements.items():code=code.replace(a,b)
    exec(compile(code,m.__file__,'exec'),m.__dict__);return m
d=module('returns_builder','download_pool_returns.py',{"pd.Timestamp('2021-01-01',tz='Asia/Shanghai')":"pd.Timestamp('2019-01-01',tz='Asia/Shanghai')"})
d.OUT=RET;d.CACHE=ROOT/'data/cache/early_4h'
for folder in [d.CACHE,RET/'by_symbol',RET/'gap_repairs']:folder.mkdir(parents=True,exist_ok=True)
b=module('table_builder','build_train_predict_tables.py',{"pd.Timestamp('2021-01-01',tz='Asia/Shanghai')":"pd.Timestamp('2019-01-01',tz='Asia/Shanghai')"})
b.RET=RET;b.CACHE=d.CACHE;b.OUT=OUT;b.FEATURE_ROOT=OUT/'raw_features_by_symbol';b.FEATURE_ROOT.mkdir(exist_ok=True)
b.FEATURES=json.loads((OLD/'feature_columns.json').read_text());b.PRICE_FEATURES=b.FEATURES[4:]
c=module('cmc_builder','collect_market_cap_supply.py');c.OUT=CAP
mapping=json.loads((ROOT/'asset_mapping.json').read_text())
lists={p.stem:json.loads(p.read_text()) for p in (ROOT/'archive_lists').glob('*.json')}
keys={n:[] for n in lists};tasks=[]
for n,v in lists.items():
    for item in v['objects']:
        if '/monthly/' in item['key'] and '2019-01'<=item['key'][-11:-4]<'2021-01':
            tasks.append(item);keys[n].append(item['key'])
sources=[]
with ThreadPoolExecutor(max_workers=8) as ex:
    fs=[ex.submit(d.download,item,{}) for item in tasks]
    for i,f in enumerate(as_completed(fs),1):
        sources.append(f.result())
        if i%100==0:print('Archives',i,len(tasks),flush=True)
print('Monthly archives done',len(sources),flush=True)
api_status=[]
def api2019(n):
    path=d.CACHE/f'{n}-4h-2019-api.zip'
    if not path.exists():
        rows=[];start=1546300800000
        while start<1577836800000:
            r=requests.get('https://fapi.binance.com/fapi/v1/klines',params=dict(symbol=n,interval='4h',startTime=start,endTime=1577836799999,limit=1500),timeout=40)
            if r.status_code==400 and r.json().get('code')==-1121:return n,None,'symbol_unavailable_on_live_api'
            r.raise_for_status();part=r.json()
            if not part:break
            rows.extend(part);start=part[-1][0]+14400000
        if not rows:return n,None,'no_2019_bars_returned'
        stream=io.StringIO();csv.writer(stream,lineterminator='\n').writerows(rows)
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:z.writestr(path.stem+'.csv',stream.getvalue())
    key=f'data/futures/um/api/klines/{n}/4h/{path.name}'
    return n,dict(key=key,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size),'available'
# Only contracts with January2020 archived trading can have earlier2019 history.
early=[n for n,k in keys.items() if any(x.endswith('2020-01.zip') for x in k)]
with ThreadPoolExecutor(max_workers=4) as ex:
    for n,rec,status in ex.map(api2019,early):
        api_status.append(dict(native=n,status=status))
        if rec:sources.append(rec);keys[n].append(rec['key'])
print('2019 API',api_status,flush=True)
(RET/'acquisition.json').write_text(json.dumps(dict(sources=sources,api_status=api_status),indent=2))
active={n for n,k in keys.items() if k}
needed=[a for a in mapping if a['base']+'USDT' in active]
ids=sorted({i for a in needed for i in a['cmc_ids']})
def history(job):
    cid,year=job
    url=c.BASE+'cryptocurrency/historical?'+c.urlencode(dict(id=cid,convertId=2781,timeStart=c.timestamp(year),timeEnd=c.timestamp(year+1)-1,interval='1d'))
    env=c.request_cached(url,CAP/'raw/history'/str(cid)/f'{year}.json.gz')
    return cid,year,env
provider={};jobs=[(i,y) for i in ids for y in [2019,2020]]
with ThreadPoolExecutor(max_workers=4) as ex:
    fs=[ex.submit(history,j) for j in jobs]
    for i,f in enumerate(as_completed(fs),1):
        cid,year,env=f.result();data=env['response'].get('data') or {}
        for q in data.get('quotes',[]):
            assert q['timeOpen'][:4]==str(year)
            assert q['timeOpen'][11:19]=='00:00:00' and q['timeClose'][11:19]=='23:59:59'
            provider[cid,q['timeOpen'][:10]]=q['quote']
        if i%30==0:print('CMC histories',i,len(jobs),flush=True)
caprows=[]
for a in needed:
    for dt in pd.date_range('2019-01-01','2020-12-31'):
        day=dt.strftime('%Y-%m-%d');choices=[(cid,provider[cid,day]) for cid in a['cmc_ids'] if (cid,day) in provider]
        if not choices:continue
        cid,q=choices[0];cap=c.positive(q.get('marketCap'));supply=c.positive(q.get('circulatingSupply'));price=c.positive(q.get('close'))
        err=abs(cap-price*supply)/cap if cap and supply and price else np.nan
        caprows.append(dict(contract=a['contract'],date=dt,cmc_id=cid,market_cap_usd=cap,marketcap_identity_error=err))
caps=pd.DataFrame(caprows);caps.to_parquet(CAP/'contract_daily.parquet',index=False)
caps.loc[(caps.market_cap_usd<=0)|(caps.marketcap_identity_error>.01),'market_cap_usd']=np.nan
panel=caps.pivot(index='date',columns='contract',values='market_cap_usd').reindex(pd.date_range('2019-01-01','2020-12-31'))
idpanel=caps.pivot(index='date',columns='contract',values='cmc_id').reindex_like(panel)
cappanels={f'market_cap_mean_{w}d':panel.rolling(w,min_periods=w).mean().where(idpanel.rolling(w,min_periods=w).min().eq(idpanel.rolling(w,min_periods=w).max())) for w in [10,20,30,40]}
frames=[];features=[];sha={r['key']:r['sha256'] for r in sources}
with ThreadPoolExecutor(max_workers=4) as ex:
    fs={ex.submit(d.build,n,2020,keys[n]):n for n in sorted(active)}
    for i,f in enumerate(as_completed(fs),1):
        n=fs[f];frames.append(f.result());features.append(b.symbol_features(n,{k:sha[k] for k in keys[n]}))
        if i%20==0:print('Raw symbol data',i,len(active),flush=True)
union=pd.concat(frames,ignore_index=True);raw=pd.concat(features,ignore_index=True)
raw.to_parquet(OUT/'raw_features_2019_2020.parquet',index=False)
summary=[];validation=[]
for t in range(2021,2026):
    oldpath=OLD/'train_table_2021-2025'/f'train_table_{t}.parquet';old=pd.read_parquet(oldpath)
    pool=t+1;members=sorted(old.symbol.unique());local=union.loc[union.symbol.isin(members)].copy()
    # Explicit member-date grid retains prelisting missingness and pool coverage.
    grid=pd.MultiIndex.from_product([members,pd.date_range('2019-01-01','2020-12-31',tz='Asia/Shanghai')],names=['symbol','entry_time_utc']).to_frame(index=False)
    grid.entry_time_utc=grid.entry_time_utc.dt.tz_convert('UTC')
    local=grid.merge(local,on=['symbol','entry_time_utc'],how='left',validate='one_to_one')
    local['pool_year']=pool;local['data_year']=local.entry_time_utc.dt.tz_convert('Asia/Shanghai').dt.year
    local['decision_date_beijing']=local.entry_time_utc.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d')
    local['exit_time_utc']=local.entry_time_utc+pd.Timedelta(days=1)
    local['return_known_not_before_utc']=local.exit_time_utc+pd.Timedelta(hours=4)
    local['status']=local.status.fillna('no_historical_contract_data')
    for col in ['has_complete_window','all_seven_bars_positive_volume']:local[col]=local[col].fillna(False).astype(bool)
    local['bars_present_7']=local.bars_present_7.fillna(0).astype('int64')
    dest=RET/f'pool_{pool}';dest.mkdir(exist_ok=True);local.to_parquet(dest/'daily_returns.parquet',index=False)
    b.make_table(t,'train',raw,cappanels)
    dest=OUT/'train_table_2021-2025'/f'train_table_{t}.parquet';new=pd.read_parquet(dest)
    assert new.data_year.lt(2021).all()
    combined=pd.concat([new,old],ignore_index=True).sort_values(['entry_time_utc','symbol']).reset_index(drop=True)
    preserved=combined.loc[combined.data_year>=2021,old.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(preserved,old.sort_values(['entry_time_utc','symbol']).reset_index(drop=True),check_dtype=False)
    assert not combined.duplicated(['symbol','entry_time_utc']).any()
    combined.to_parquet(dest,index=False)
    # Remove intermediate early-only compressed CSV; main artifact is full parquet.
    dest.with_suffix('.csv.gz').unlink(missing_ok=True)
    for year,p in combined.groupby('data_year'):
        summary.append(dict(table=f'train_table_{t}',pool_year=pool,data_year=int(year),rows=len(p),members=p.symbol.nunique(),features_complete=int(p.features_complete.sum()),trainable_rows=int((p.features_complete&p.label_available).sum()),label_rows=int(p.label_available.sum())))
    validation.append(dict(table=dest.name,original_rows_preserved=len(old),appended_rows=len(new),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    print('EXTENDED',dest.name,len(combined),flush=True)
pd.DataFrame(summary).to_csv(OUT/'coverage_by_table_year.csv',index=False)
(OUT/'feature_columns.json').write_text(json.dumps(b.FEATURES,indent=2))
(OUT/'manifest.json').write_text(json.dumps(dict(start='2019-01-01',tables=validation,features=b.FEATURES,api_status=api_status,original_2021_onward='preserved and verified',missing='null; no spot substitution or forward fill',percentiles='per annual pool per date, independent of future return availability',time='Beijing00, next24h open-to-open; marketcap EOD plus24h availability delay',predict_tables='unchanged; use existing factors15_2021_2026 predict tables'),indent=2))
print(pd.DataFrame(summary).to_string(index=False),flush=True)
print('DONE',OUT,flush=True)
