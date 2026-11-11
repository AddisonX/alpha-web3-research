import importlib.util
import numpy as np,pandas as pd,torch
spec=importlib.util.spec_from_file_location('alpha','/workspace/alpha191_16_gpu.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def apply_mask(hist,fit,features,start,boundary,out):
 q=hist.loc[hist.entry_time_utc.ge(start)&hist.entry_time_utc.lt(boundary)].sort_values(['entry_time_utc','symbol'])
 D=q.entry_time_utc.nunique();N=q.symbol.nunique();assert len(q)==D*N
 x=torch.tensor(q[features].to_numpy().reshape(D,N,len(features)),device='cuda',dtype=torch.float64)
 y=torch.tensor(q.excess_return_24h.to_numpy().reshape(D,N),device='cuda',dtype=torch.float64)
 known=torch.tensor(q.label_price_time_utc.le(boundary).to_numpy().reshape(D,N),device='cuda');rows=[]
 for j,name in enumerate(features):
  m=torch.isfinite(x[:,:,j])&torch.isfinite(y)&known
  xr=a.rank(x[:,:,j],m);yr=a.rank(y,m)
  dx=torch.where(m,xr-torch.nanmean(xr,1)[:,None],0.);dy=torch.where(m,yr-torch.nanmean(yr,1)[:,None],0.)
  ic=(dx*dy).sum(1)/(dx.square().sum(1)*dy.square().sum(1)).sqrt()
  v=ic[(m.sum(1)>=20)&torch.isfinite(ic)];value=float(v.mean()) if len(v) else np.nan
  masked=not np.isfinite(value) or abs(value)<=.01
  if masked:fit.loc[:,name]=np.nan
  rows.append(dict(block=str(boundary),factor=name,RankIC=value,ic_days=len(v),masked=masked))
 out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'mask_audit.csv',index=False)
