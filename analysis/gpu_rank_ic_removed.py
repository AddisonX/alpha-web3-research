import numpy as np,torch,json,csv
from pathlib import Path
assert torch.cuda.is_available()
r=Path('/tmp/ic15');a=np.load(r/'inputs.npz');names=a['names'].tolist();idx=[a['names'].tolist().index(n) for n in names];out=[];daily=[]
def rank(v,m):
 order=torch.argsort(torch.where(m,v,torch.inf),dim=1,stable=True);s=v.gather(1,order);n=v.shape[1];p=torch.arange(n,device='cuda')[None,:].expand_as(v)
 change=torch.ones_like(m);change[:,1:]=s[:,1:]!=s[:,:-1];start=torch.cummax(torch.where(change,p,0),1).values
 endchange=torch.ones_like(m);endchange[:,:-1]=s[:,:-1]!=s[:,1:];end=torch.flip(torch.cummin(torch.flip(torch.where(endchange,p,n-1),[1]),1).values,[1])
 ans=torch.empty_like(v).scatter_(1,order,(start+end).double()/2+1);return torch.where(m,ans,0)
for year in range(2021,2027):
 x=torch.as_tensor(a[f'x{year}'][:,:,idx],device='cuda',dtype=torch.float64);y=torch.as_tensor(a[f'y{year}'],device='cuda',dtype=torch.float64);known=torch.as_tensor(a[f'k{year}'],device='cuda',dtype=torch.bool)
 active=[i for i,n in enumerate(names) if n!='range_to_quote_volume_2'];mask=torch.isfinite(x[:,:,active]).all(2)&torch.isfinite(x[:,:,names.index('range_to_quote_volume_2')])&known&torch.isfinite(y);count=mask.sum(1);valid=count>=20
 yr=rank(y,mask);dy=torch.where(mask,yr-yr.sum(1)[:,None]/count.clamp(min=1)[:,None],0)
 for j,name in enumerate(names):
  if name!='range_to_quote_volume_2':continue
  xr=rank(x[:,:,j],mask);dx=torch.where(mask,xr-xr.sum(1)[:,None]/count.clamp(min=1)[:,None],0)
  ic=(dx*dy).sum(1)/(dx.square().sum(1)*dy.square().sum(1)).sqrt();good=valid&torch.isfinite(ic);v=ic[good];dates=a[f'd{year}'][good.cpu().numpy()]
  out.append(dict(year=year,factor=name,IC=v.mean().item(),ICIR=(v.mean()/v.std(correction=1)).item(),days=len(v),first=dates[0],last=dates[-1]))
  daily.extend(dict(year=year,factor=name,date=date,IC=float(value)) for date,value in zip(dates,v.cpu().numpy()))
for filename,rows in [('removed_factor_annual_rank_ic.csv',out),('removed_factor_daily_rank_ic.csv',daily)]:
 with open(r/filename,'w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(r/'removed_factor_rank_method.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),IC='Spearman: Pearson correlation of average-tie ranks of factor and next24h price excess return; annual mean daily IC',ICIR='Mean daily RankIC / sample standard deviation, not annualized',sample='All14 active factors finite, knownpool for2022+, finite label; individual missing coins excluded; at least20 members',year2021='Retrospective2022pool training data, not out of sample'),indent=2))
for year in range(2021,2027):print(year,[(v['factor'],round(v['IC'],4)) for v in out if v['year']==year],flush=True)
