"""Append42 screened Alpha191 factors to original15; internal and outer percentile ranks."""
from pathlib import Path
import importlib.util,json,sys
sys.path.insert(0,"/tmp/alpha191_all_code")
import formulas_gpu as fg
import numpy as np,pandas as pd,torch
spec=importlib.util.spec_from_file_location('alpha','/workspace/alpha191_16_gpu.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
R=Path('/tmp/xgb22run');B=R/'data/model_tables/factors15_2021_2026';O=R/'data/model_tables/factors57alpha_2021_2026'
NEW=pd.read_csv('/tmp/alpha191_all_pct/selected_factors.csv').factor.sort_values().tolist();assert len(NEW)==42
names=json.loads((B/'feature_columns.json').read_text())+NEW
O.mkdir(parents=True,exist_ok=True);(O/'feature_columns.json').write_text(json.dumps(names,indent=2))
bars=pd.read_parquet('/workspace/bars.parquet');audit=[]
def selected(o,h,l,c,v,amt):
 calc=fg.Formulas(o,h,l,c,v,amt)
 ready=a.roll((torch.isfinite(c)&torch.isfinite(v)).double(),120,'sum')==120
 values=[]
 for name in NEW:
  z=calc.functions[int(name[-3:])]()
  values.append(torch.where(ready&torch.isfinite(z),z,torch.nan))
 return torch.stack(values,-1)
for year in range(2022,2027):
 tr=B/f'train_table_2021-2025/train_table_{year-1}.parquet';te=B/f'predict_table_2022-2026/predict_table_{year}.parquet'
 train=pd.read_parquet(tr);test=pd.read_parquet(te)
 assert train.data_year.min()==2021 and train.data_year.max()==year-1
 q=pd.concat([train,test],ignore_index=True).sort_values(['entry_time_utc','symbol'])
 assert not q.duplicated(['symbol','entry_time_utc']).any()
 syms=sorted(q.symbol.unique());dates=pd.DatetimeIndex(sorted(q.entry_time_utc.unique()));stamps=dates.as_unit('ms').asi8
 # Match per-calendar-year SMA initialization to the screened-factor calculation.
 out=[]
 for data_year in range(2021,year+1):
  ds=dates[dates.tz_convert('Asia/Shanghai').year==data_year]
  if not len(ds):continue
  dsms=ds.as_unit('ms').asi8;start=dsms[0]-360*a.BAR;last=dsms[-1]
  grid=np.arange(start,last,a.BAR,dtype=np.int64)
  b=bars.loc[bars.symbol.isin(syms)&bars.timestamp.between(start,last-a.BAR)]
  ri=torch.tensor(pd.Index(grid).get_indexer(b.timestamp),device='cuda');ci=torch.tensor(pd.Index(syms).get_indexer(b.symbol),device='cuda')
  z=torch.full((len(grid),len(syms),6),torch.nan,device='cuda')
  z[ri,ci]=torch.tensor(b[['open','high','low','close','volume','amount']].to_numpy(),device='cuda')
  good=torch.isfinite(z).all(-1)&(z>0).all(-1);z=torch.where(good[:,:,None],z,torch.nan)
  v=selected(*z.unbind(-1));idx=pd.Index(grid+a.BAR).get_indexer(dsms);assert (idx>=0).all()
  v=v[torch.tensor(idx,device='cuda')]
  for j in range(len(NEW)):
   raw=v[:,:,j];v[:,:,j]=a.rank(raw)/torch.isfinite(raw).sum(1).clamp(min=1)[:,None]
  f=pd.DataFrame(v.reshape(-1,len(NEW)).cpu().numpy(),columns=NEW)
  f['symbol']=np.tile(syms,len(ds));f['entry_time_utc']=np.repeat(ds,len(syms));out.append(f)
 addon=pd.concat(out,ignore_index=True)
 for split,source,p in [('train',train,tr),('predict',test,te)]:
  result=source.merge(addon,on=['symbol','entry_time_utc'],how='left',validate='one_to_one',sort=False)
  assert len(result)==len(source)
  # Original15 features and all labels preserved exactly, no future-outcome sample filtering.
  for col in source.columns:assert result[col].reset_index(drop=True).equals(source[col].reset_index(drop=True)),col
  result['features_complete']=np.isfinite(result[names]).all(axis=1)
  result['prediction_eligible_on_features']=result.features_complete&result.pool_known_at_decision
  dest=O/p.parent.name/p.name;dest.parent.mkdir(exist_ok=True);result.to_parquet(dest,index=False)
  audit.append(dict(table=p.name,rows=len(result),complete57=int(result.features_complete.sum()),complete15=int(np.isfinite(source[names[:15]]).all(axis=1).sum()),pool_year=year))
 print('GPU pool-specific57 ready',year,flush=True)
pd.DataFrame(audit).to_csv(O/'audit.csv',index=False)
(O/'method.json').write_text(json.dumps(dict(features=names,additional=NEW,normalization='Each additional factor ranked independently within fixed target-year pool on each date; average-tie percentiles; labels never used for ranking',cohort='Training history is recomputed for target-year pool, including Alpha001 internal cross-sectional ranks, not merged from another annual pool',source='Existing SHA-verified Binance4h archives; original15 table values and labels preserved',gpu=torch.cuda.get_device_name()),indent=2))
