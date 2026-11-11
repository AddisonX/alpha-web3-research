import sys
sys.path.append('/workspace/alpha-web3-research/.venv/lib/python3.12/site-packages')
import requests,time,io,zipfile,json,hashlib,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd,numpy as np
BASE=Path('/tmp/alpha15-run/data/model_tables/factors15_2021_2026')
OUT=Path('/tmp/funding_four_factors');OUT.mkdir(exist_ok=True)
CACHE=OUT/'cache';CACHE.mkdir(exist_ok=True)
tables={y:pd.read_parquet(BASE/f'predict_table_2022-2026/predict_table_{y}.parquet') for y in range(2022,2027)}
jobs=set()
for y,df in tables.items():
    for symbol in df.symbol.unique():
        native=symbol.split('/')[0]+'USDT'
        for month in pd.period_range(f'{y-1}-12',f'{y}-'+('10' if y==2026 else '12'),freq='M'):
            jobs.add((native,str(month),'fundingRate'))
            if month.year==y:jobs.add((native,str(month),'markPriceKlines'))
def fetch(job):
    n,m,kind=job
    name=f'{n}-fundingRate-{m}.zip' if kind=='fundingRate' else f'{n}-1h-{m}.zip'
    path=CACHE/kind/name;path.parent.mkdir(exist_ok=True)
    suffix=f'{kind}/{n}/'+('1h/' if kind=='markPriceKlines' else '')+name
    url='https://data.binance.vision/data/futures/um/monthly/'+suffix
    if not path.exists():
        for attempt in range(3):
            r=requests.get(url,timeout=35)
            if r.status_code==404:return dict(native=n,month=m,kind=kind,status='missing',url=url)
            if r.status_code==200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:assert z.testzip() is None
                path.write_bytes(r.content);break
            r.raise_for_status()
    return dict(native=n,month=m,kind=kind,status='ok',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
records=[]
print('Downloading archives',len(jobs),flush=True)
with ThreadPoolExecutor(max_workers=12) as ex:
    fs=[ex.submit(fetch,j) for j in sorted(jobs)]
    for i,f in enumerate(as_completed(fs),1):
        records.append(f.result())
        if i%400==0:print('Archives',i,'/',len(jobs),flush=True)
(OUT/'acquisition.json').write_text(json.dumps(records,indent=2))
def read(path):
    with zipfile.ZipFile(path) as z:return pd.read_csv(z.open(z.namelist()[0]))
fund={};marks={}
for n in sorted({j[0] for j in jobs}):
    fs=[read(r['path']) for r in records if r['native']==n and r['kind']=='fundingRate' and r['status']=='ok']
    ms=[]
    for r in records:
        if r['native']==n and r['kind']=='markPriceKlines' and r['status']=='ok':
            with zipfile.ZipFile(r['path']) as z:
                p=pd.read_csv(z.open(z.namelist()[0]),header=None,usecols=[0,1],dtype=str)
            p=p.loc[p[0].str.match(r'^\d+$')].astype(float);p.columns=['stamp','mark']
            ms.append(p)
    if fs:
        f=pd.concat(fs).drop_duplicates('calc_time').sort_values('calc_time')
        fund[n]=f
    if ms:
        m=pd.concat(ms).drop_duplicates('stamp').sort_values('stamp');marks[n]=m.set_index('stamp').mark
print('Sources parsed',len(fund),len(marks),flush=True)
features=['funding_latest_hourly','funding_sum_3d','funding_sum_7d','funding_mean_1d_minus_7d']
summaries=[]
for year,df in tables.items():
    if year==2026:df=df.loc[df.decision_date_beijing.le('2026-10-27')]
    results=[]
    for symbol,q in df.groupby('symbol',sort=False):
        q=q.copy();n=symbol.split('/')[0]+'USDT'
        for col in [*features,'funding_long_cost_24h']:q[col]=np.nan
        q['funding_settlements_24h']=0
        if n not in fund or n not in marks:results.append(q);continue
        f=fund[n];times=(f.calc_time.to_numpy(dtype=np.int64)//3600000)*3600000
        rates=f.last_funding_rate.to_numpy(float);hours=f.funding_interval_hours.to_numpy(float)
        m=marks[n].reindex(times).to_numpy()
        # Events use nominal settlement hour. Signals exclude the current00:00 settlement;
        # the prior holdings pay it. Forward holding receives events in(t,t+24h].
        for idx,row in q.iterrows():
            t=int(row.entry_time_utc.timestamp()*1000)
            k=np.searchsorted(times,t,side='left')
            if k==0:continue
            last=k-1
            if t-times[last]>8*3600000:continue
            q.loc[idx,features[0]]=rates[last]/hours[last]
            sums={}
            for days in [1,3,7]:
                lo=np.searchsorted(times,t-days*86400000,side='left')
                tt=times[lo:k];hh=hours[lo:k]
                # Require a continuous settlement history across the lookback window.
                good=len(tt)>0 and (t-tt[0]<=days*86400000) and (tt[0]<=t-days*86400000+8*3600000)
                good=good and np.all(np.diff(tt)==hh[1:]*3600000)
                good=good and abs(hh.sum()-days*24)<1e-8
                if good:sums[days]=rates[lo:k].sum()
            if 3 in sums:q.loc[idx,features[1]]=sums[3]
            if 7 in sums:q.loc[idx,features[2]]=sums[7]
            if 1 in sums and 7 in sums:q.loc[idx,features[3]]=sums[1]/24-sums[7]/168
            a=np.searchsorted(times,t,side='right');b=np.searchsorted(times,t+86400000,side='right')
            tt=times[a:b];hh=hours[a:b]
            good=len(tt)>0 and tt[0]-hh[0]*3600000==t and tt[-1]==t+86400000 and np.all(np.diff(tt)==hh[1:]*3600000)
            if good and np.isfinite(m[a:b]).all() and np.isfinite(row.entry_open) and row.entry_open>0:
                q.loc[idx,'funding_long_cost_24h']=np.sum(rates[a:b]*m[a:b]/row.entry_open)
                q.loc[idx,'funding_settlements_24h']=len(tt)
        results.append(q)
    result=pd.concat(results,ignore_index=True)
    cols=['pool_year','data_year','symbol','decision_date_beijing','entry_time_utc','exit_time_utc','entry_open','exit_open','realized_return_24h','benchmark_return_24h','pool_known_at_decision',*features,'funding_long_cost_24h','funding_settlements_24h']
    result[cols].to_parquet(OUT/f'funding_table_{year}.parquet',index=False)
    summaries.append(dict(year=year,rows=len(result),**{c:int(result[c].notna().sum()) for c in features},funding_cost_rows=int(result.funding_long_cost_24h.notna().sum())))
    print('TABLE',summaries[-1],flush=True)
pd.DataFrame(summaries).to_csv(OUT/'coverage.csv',index=False)
(OUT/'feature_columns.json').write_text(json.dumps(features))
print('FUNDING TABLES DONE',flush=True)
