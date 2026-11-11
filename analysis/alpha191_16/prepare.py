"""Read existing verified archives only; no factor computation on CPU."""
from pathlib import Path
import importlib.util, json, io, zipfile, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2]
O=R/'reports/alpha191_16';O.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('builder',R/'analysis/build_train_predict_tables.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
B=R/'data/model_tables/factors15_2021_2026'
labels=[]
cols=['symbol','entry_time_utc','exit_time_utc','decision_date_beijing','pool_year','data_year','pool_known_at_decision','realized_return_24h','benchmark_return_24h','entry_open','exit_open']
for year in range(2021,2027):
 p=B/('train_table_2021-2025/train_table_2021.parquet' if year==2021 else f'predict_table_2022-2026/predict_table_{year}.parquet')
 f=pd.read_parquet(p,columns=cols);f=f.loc[f.data_year.eq(year)&f.decision_date_beijing.le('2026-10-27')];labels.append(f)
labels=pd.concat(labels,ignore_index=True);labels.to_parquet(O/'labels.parquet',index=False)
native={s.split('/')[0]+'USDT' for s in labels.symbol.unique()}
keys=b.sources()
def read(item):
 sym,ks=item;parts=[]
 for key,sha in sorted(ks.items(),key=lambda kv:('/daily/' in kv[0],kv[0])):
  path=b.CACHE/key.rsplit('/',1)[-1];body=path.read_bytes()
  assert hashlib.sha256(body).hexdigest()==sha,key
  with zipfile.ZipFile(io.BytesIO(body)) as z:raw=z.read(next(n for n in z.namelist() if n.endswith('.csv')))
  a=np.loadtxt(io.BytesIO(raw),delimiter=',',skiprows=0 if raw.split(b',',1)[0].isdigit() else 1,usecols=[0,1,2,3,4,5,7],ndmin=2)
  if len(a):
   a[:,0]=np.where(a[:,0]>1e14,a[:,0]/1000,a[:,0]);parts.append(a)
 f=pd.DataFrame(np.concatenate(parts),columns=['timestamp','open','high','low','close','volume','amount'])
 f=f.drop_duplicates('timestamp',keep='first').sort_values('timestamp');f=f.loc[(f.timestamp>=1577836800000)&(f.timestamp<1793203200000)]
 f['timestamp']=f.timestamp.astype('int64');f['symbol']=sym[:-4]+'/USDT:USDT';return f
frames=[]
with ThreadPoolExecutor(max_workers=6) as ex:
 fs=[ex.submit(read,(s,keys[s])) for s in sorted(native)]
 for i,f in enumerate(as_completed(fs),1):
  frames.append(f.result())
  if i%40==0:print('archives parsed',i,'/',len(fs),flush=True)
bars=pd.concat(frames,ignore_index=True);bars.to_parquet(O/'bars.parquet',index=False,compression='zstd')
print('READY bars',len(bars),'labels',len(labels),'MB',round((O/'bars.parquet').stat().st_size/1e6,1),flush=True)
