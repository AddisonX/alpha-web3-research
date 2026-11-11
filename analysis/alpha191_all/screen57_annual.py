from pathlib import Path
import pandas as pd,numpy as np,torch,json,sys
sys.path.insert(0,'/tmp/alpha191_all_code');from formulas_gpu import rank,div
B=Path('/tmp/xgb22run/data/model_tables/factors15_2021_2026');names=json.loads((B/'feature_columns.json').read_text());out=[]
for yr in range(2021,2026):
 p=B/('train_table_2021-2025/train_table_2021.parquet' if yr==2021 else f'predict_table_2022-2026/predict_table_{yr}.parquet')
 q=pd.read_parquet(p);q=q.loc[q.data_year.eq(yr)].sort_values(['entry_time_utc','symbol']);D=q.entry_time_utc.nunique();N=q.symbol.nunique();assert len(q)==D*N
 y=torch.tensor(q.excess_return_24h.to_numpy().reshape(D,N),device='cuda');known=torch.ones((D,N),device='cuda',dtype=torch.bool) if yr==2021 else torch.tensor(q.pool_known_at_decision.to_numpy().reshape(D,N),device='cuda')
 for name in names:
  x=torch.tensor(q[name].to_numpy().reshape(D,N),device='cuda');m=torch.isfinite(x)&torch.isfinite(y)&known;xr=rank(x,m);yrank=rank(y,m);dx=torch.where(m,xr-torch.nanmean(xr,1)[:,None],0.);dy=torch.where(m,yrank-torch.nanmean(yrank,1)[:,None],0.)
  ic=div((dx*dy).sum(1),(dx.square().sum(1)*dy.square().sum(1)).sqrt());ic=ic[(m.sum(1)>=20)&torch.isfinite(ic)]
  out.append(dict(factor=name,year=yr,RankIC=ic.mean().item(),days=len(ic)))
pd.DataFrame(out).to_csv('/tmp/original15_individual_rankic_2021_2025.csv',index=False);print('15 factors annual RankIC saved; individual valid samples; GPU')
