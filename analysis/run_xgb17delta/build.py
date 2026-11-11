from pathlib import Path
import pandas as pd,numpy as np,json,requests,zipfile,io,time,torch,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path('/workspace/xgb17deltarun');BASE=Path('/workspace/xgb16run/data/model_tables/factors15_2021_2026');OUT=ROOT/'data/model_tables/factors17delta_2021_2026';OUT.mkdir(parents=True,exist_ok=True)
cache=Path('/workspace/xgb16run/funding_cache');cache.mkdir(exist_ok=True)
paths=sorted(BASE.glob('train_table_2021-2025/*.parquet'))+sorted(BASE.glob('predict_table_2022-2026/*.parquet'))
frames={p:pd.read_parquet(p) for p in paths};cut=pd.Timestamp('2026-10-28',tz='Asia/Shanghai').tz_convert('UTC')
need=pd.concat([f.loc[f.entry_open.notna()&f.entry_time_utc.lt(cut),['symbol','entry_time_utc']] for f in frames.values()]).drop_duplicates()
seed=pd.read_parquet(ROOT/'seed.parquet');cols=['funding_mean_1d_minus_7d','funding_sum_7d'];keys=['symbol','entry_time_utc']
missing=need.merge(seed[keys].assign(present=True),on=keys,how='left');missing=missing.loc[missing.present.isna(),keys]
jobs=set()
for sym,q in missing.groupby('symbol'):
 for m in q.entry_time_utc.dt.tz_localize(None).dt.to_period('M').unique():
  for mm in [m-1,m]:jobs.add((sym,str(mm)))
print('Missing symbol-days',len(missing),'archives',len(jobs),flush=True)
sessions=threading.local()
def fetch(job):
 if not hasattr(sessions,"http"):sessions.http=requests.Session()
 sym,month=job;n=sym.split('/')[0]+'USDT';p=cache/f'{n}-fundingRate-{month}.zip'
 if p.exists():return sym,str(p),'cached'
 if p.with_suffix('.404').exists():return sym,None,'404'
 url=f'https://data.binance.vision/data/futures/um/monthly/fundingRate/{n}/{p.name}'
 for attempt in range(4):
  try:
   r=sessions.http.get(url,timeout=25)
   if r.status_code==404:p.with_suffix('.404').touch();return sym,None,'404'
   r.raise_for_status()
   with zipfile.ZipFile(io.BytesIO(r.content)) as z:assert z.testzip() is None
   p.write_bytes(r.content);return sym,str(p),'downloaded'
  except Exception:
   if attempt==3:raise
   time.sleep(1+attempt)
records=[]
with ThreadPoolExecutor(max_workers=24) as ex:
 for i,f in enumerate(as_completed([ex.submit(fetch,j) for j in sorted(jobs)]),1):
  records.append(f.result())
  if i%400==0:print('Archives',i,'/',len(jobs),flush=True)
(ROOT/'funding_acquisition.json').write_text(json.dumps(records))
lookup={}
for sym,p,status in records:
 if p:lookup.setdefault(sym,[]).append(p)
parts=[]
for sym,q in missing.groupby('symbol'):
 q=q.copy();q[cols]=np.nan
 if sym not in lookup:parts.append(q);continue
 fs=[]
 for p in lookup[sym]:
  with zipfile.ZipFile(p) as z:fs.append(pd.read_csv(z.open(z.namelist()[0])))
 f=pd.concat(fs).drop_duplicates('calc_time').sort_values('calc_time')
 tt=f.calc_time.to_numpy(dtype=np.int64)//3600000*3600000;hh=f.funding_interval_hours.to_numpy(float);rr=f.last_funding_rate.to_numpy(float)
 t=q.entry_time_utc.dt.as_unit('ms').astype('int64').to_numpy();k=np.searchsorted(tt,t,side='left');last=np.maximum(k-1,0);fresh=(k>0)&(t-tt[last]<=8*3600000)
 pr=np.r_[0,np.cumsum(rr)];ph=np.r_[0,np.cumsum(hh)];bad=np.r_[0,np.diff(tt)!=hh[1:]*3600000];pb=np.r_[0,np.cumsum(bad)]
 for days,col in [(1,cols[0]),(7,cols[1])]:
  lo=np.searchsorted(tt,t-days*86400000,side='left');li=np.minimum(lo,len(tt)-1)
  good=fresh&(k>lo)&(tt[li]<=t-days*86400000+8*3600000)&(np.abs(ph[k]-ph[lo]-days*24)<1e-8)&((pb[k]-pb[np.minimum(lo+1,len(tt))])==0)
  q[col]=np.where(good,pr[k]-pr[lo],np.nan)
 q[cols[0]]=q[cols[0]]/24-q[cols[1]]/168
 parts.append(q)
raw=pd.concat([seed,*parts],ignore_index=True).drop_duplicates(keys);raw.to_parquet(ROOT/'funding_raw_all.parquet',index=False)
features=json.loads((BASE/'feature_columns.json').read_text())+cols
(OUT/'feature_columns.json').write_text(json.dumps(features,indent=2));assert len(features)==17;assert torch.cuda.is_available()
def ranks(v,m):
 order=torch.argsort(torch.where(m,v,torch.inf),dim=1,stable=True);s=v.gather(1,order);n=v.shape[1];p=torch.arange(n,device='cuda')[None,:].expand_as(v)
 c=torch.ones_like(m);c[:,1:]=s[:,1:]!=s[:,:-1];st=torch.cummax(torch.where(c,p,0),1).values
 c=torch.ones_like(m);c[:,:-1]=s[:,:-1]!=s[:,1:];en=torch.flip(torch.cummin(torch.flip(torch.where(c,p,n-1),[1]),1).values,[1])
 r=torch.empty_like(v).scatter_(1,order,(st+en).double()/2+1);return torch.where(m,r/m.sum(1).clamp(min=1)[:,None],torch.nan)
coverage=[]
for p,f in frames.items():
 f=f.merge(raw,on=keys,how='left',validate='many_to_one').sort_values(['entry_time_utc','symbol']).reset_index(drop=True)
 n=f.symbol.nunique();d=f.entry_time_utc.nunique();assert len(f)==d*n
 for col in cols:
  v=torch.tensor(f[col].to_numpy().reshape(d,n),device='cuda',dtype=torch.float64);f[col]=ranks(v,torch.isfinite(v)).cpu().numpy().reshape(-1)
 dest=OUT/p.relative_to(BASE);dest.parent.mkdir(exist_ok=True);f.to_parquet(dest,index=False)
 coverage.append(dict(table=p.name,rows=len(f),complete15=int(np.isfinite(f[features[:-2]]).all(1).sum()),complete17=int(np.isfinite(f[features]).all(1).sum())))
 print('TABLE',coverage[-1],flush=True)
pd.DataFrame(coverage).to_csv(ROOT/'table_coverage.csv',index=False)
(ROOT/'funding_method.json').write_text(json.dumps(dict(signals='Delta=sum[t-1d,t)/24 minus sum[t-7d,t)/168; other feature=sum[t-7d,t); simultaneous settlement excluded; continuous interval coverage required',percentiles='GPU average-tie ranks within each fixed pool and decision date, no future labels',cached='Reuse raw existing funding tables; download missing symbol-date histories only',features=features,gpu=torch.cuda.get_device_name()),indent=2))
