"""CUDA daily cross-sectional Pearson IC of the ten percentile features."""
import sys
sys.path.append('/workspace/alpha-web3-research/.venv/lib/python3.12/site-packages')
import json,time
from pathlib import Path
import torch
import numpy as np
import pandas as pd

ROOT=Path('/workspace/alpha-web3-research')
BASE=ROOT/'data/model_tables/factors10_2021_2026'
OUT=Path('/tmp/individual_factor_ic_predict_2022_2026');OUT.mkdir(exist_ok=True)
names=json.loads((BASE/'feature_columns.json').read_text())
assert torch.cuda.is_available()
records=[];all_daily=[];times=[]
for year in range(2022,2027):
    f=pd.read_parquet(BASE/f'predict_table_2022-2026/predict_table_{year}.parquet').sort_values(['entry_time_utc','symbol'])
    n=f.symbol.nunique();d=f.entry_time_utc.nunique();assert len(f)==n*d
    torch.cuda.synchronize();start=time.perf_counter()
    x=torch.tensor(f[names].to_numpy().reshape(d,n,10),dtype=torch.float64,device='cuda')
    y=torch.tensor(f.excess_return_24h.to_numpy().reshape(d,n),dtype=torch.float64,device='cuda')
    known=torch.tensor(f.pool_known_at_decision.to_numpy().reshape(d,n),device='cuda')
    members=torch.isfinite(x).all(dim=2)&known
    eligible_day=(members.sum(dim=1)>=20)&((~members)|torch.isfinite(y)).all(dim=1)
    count=members.sum(dim=1).clamp(min=1)
    clean_x=torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
    clean_y=torch.nan_to_num(y,nan=0.,posinf=0.,neginf=0.)
    mean_x=(clean_x*members[:,:,None]).sum(dim=1)/count[:,None]
    mean_y=(clean_y*members).sum(dim=1)/count
    dx=(clean_x-mean_x[:,None,:])*members[:,:,None]
    dy=(clean_y-mean_y[:,None])*members
    denom=(dx.square().sum(dim=1)*dy.square().sum(dim=1)[:,None]).sqrt()
    ic=(dx*dy[:,:,None]).sum(dim=1)/denom
    ic=torch.where(denom>0,ic,torch.nan)[eligible_day]
    all_daily.append(ic)
    mean=torch.nanmean(ic,dim=0)
    counts=torch.isfinite(ic).sum(dim=0)
    std=(torch.nansum((ic-mean).square(),dim=0)/(counts-1)).sqrt()
    ir=mean/std
    torch.cuda.synchronize();times.append(time.perf_counter()-start)
    values=mean.cpu().numpy();irs=ir.cpu().numpy();ns=counts.cpu().numpy()
    for i,name in enumerate(names):records.append({'factor':name,'year':str(year),'IC':float(values[i]),'ICIR':float(irs[i]),'days':int(ns[i])})
    dates=f.drop_duplicates('entry_time_utc').decision_date_beijing.to_numpy()[eligible_day.cpu().numpy()]
    pd.DataFrame(ic.cpu().numpy(),index=dates,columns=names).rename_axis('date_beijing').to_csv(OUT/f'daily_ic_{year}.csv')
    print('YEAR',year,'CUDA_IC_SECONDS',times[-1],'DAYS',len(ic),flush=True)
combined=torch.cat(all_daily,dim=0);mean=torch.nanmean(combined,dim=0)
counts=torch.isfinite(combined).sum(dim=0);std=(torch.nansum((combined-mean).square(),dim=0)/(counts-1)).sqrt()
for i,name in enumerate(names):records.append({'factor':name,'year':'2022-2026','IC':float(mean[i].item()),'ICIR':float((mean/std)[i].item()),'days':int(counts[i].item())})
result=pd.DataFrame(records);result.to_csv(OUT/'annual_factor_ic_long.csv',index=False)
pivot=result.pivot(index='factor',columns='year',values='IC').reindex(names)
pivot.to_csv(OUT/'annual_factor_ic.csv')
result.pivot(index='factor',columns='year',values='ICIR').reindex(names).to_csv(OUT/'annual_factor_icir.csv')
meta={'device':'cuda','GPU':torch.cuda.get_device_name(),'torch':torch.__version__,'gpu_compute_seconds':sum(times),
      'metric':'daily cross-sectional Pearson of original-direction factor percentile and realized next24h excess, annual daily mean',
      'sample':'same complete-ten-feature and known-pool sample as XGBoost, at least20 members; evaluation days require complete labels for all forecastable members',
      'note':'not raw arithmetic market-cap Pearson IC, not model prediction IC; no sign flipping; overall pools daily IC observations'}
(OUT/'method.json').write_text(json.dumps(meta,indent=2))
print(pivot.to_string(),flush=True)
