"""CUDA daily Spearman; mask training cells by data year using labels known at fit time."""
from pathlib import Path
import importlib.util,json
import numpy as np,pandas as pd,torch
spec=importlib.util.spec_from_file_location('alpha','/workspace/alpha191_16_gpu.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def apply_masks(train,fitting,features,fit_asof,prediction_year,out):
 rows=[];before=len(fitting);labels=fitting.excess_return_24h.copy()
 for year,q in train.groupby('data_year',sort=True):
  assert year<prediction_year
  q=q.sort_values(['entry_time_utc','symbol']);D=q.entry_time_utc.nunique();N=q.symbol.nunique();assert len(q)==D*N
  x=torch.tensor(q[features].to_numpy().reshape(D,N,len(features)),device='cuda',dtype=torch.float64)
  y=torch.tensor(q.excess_return_24h.to_numpy().reshape(D,N),device='cuda',dtype=torch.float64)
  known=torch.tensor(q.label_price_time_utc.le(fit_asof).to_numpy().reshape(D,N),device='cuda')
  for j,name in enumerate(features):
   m=torch.isfinite(x[:,:,j])&torch.isfinite(y)&known;count=m.sum(1)
   xr=a.rank(x[:,:,j],m);yr=a.rank(y,m)
   dx=torch.where(m,xr-torch.nanmean(xr,1)[:,None],0.);dy=torch.where(m,yr-torch.nanmean(yr,1)[:,None],0.)
   ic=(dx*dy).sum(1)/(dx.square().sum(1)*dy.square().sum(1)).sqrt()
   valid=(count>=20)&torch.isfinite(ic);v=ic[valid]
   annual=float(v.mean()) if len(v) else float('nan')
   masked=not np.isfinite(annual) or abs(annual)<=.01
   samples=fitting.data_year.eq(year);cells=int(samples.sum()) if masked else 0
   if masked:fitting.loc[samples,name]=np.nan
   rows.append(dict(prediction_year=prediction_year,pool_year=prediction_year,data_year=int(year),factor=name,RankIC=annual,ic_days=len(v),masked=masked,masked_training_cells=cells,reason='undefined_IC' if not np.isfinite(annual) else ('abs_IC_le_0.01' if masked else 'retained')))
 assert len(fitting)==before and fitting.excess_return_24h.equals(labels)
 audit=pd.DataFrame(rows);audit.to_csv(out/'year_factor_mask_audit.csv',index=False)
 print('MASK',prediction_year,audit.loc[audit.masked].groupby('data_year').factor.apply(list).to_dict(),flush=True)
 (out/'mask_method.json').write_text(json.dumps(dict(threshold=.01,IC='Mean daily Spearman within this training table target-year pool, factor-specific finite feature and label pairs, min20 coins; available labels only',mask='abs(IC)<=0.01 -> NaN for that factor and historical data year; undefined IC also NaN',rows='Original complete-feature training cohort selected BEFORE artificial masking; artificial NaNs never remove rows; original labels unchanged',inference='Original factor values and complete-feature inference eligibility preserved; no forecast-year outcomes used to mask',cuda=torch.cuda.get_device_name(),training_rows=before,artificially_masked_cells=int(audit.masked_training_cells.sum())),indent=2))
