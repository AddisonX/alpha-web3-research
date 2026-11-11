import numpy as np,torch,json,csv
from pathlib import Path
assert torch.cuda.is_available()
r=Path('/tmp/ic15');a=np.load(r/'inputs.npz');names=a['names'].tolist();out=[]
for year in range(2021,2027):
 x=torch.as_tensor(a[f'x{year}'],device='cuda',dtype=torch.float64);y=torch.as_tensor(a[f'y{year}'],device='cuda',dtype=torch.float64);known=torch.as_tensor(a[f'k{year}'],device='cuda',dtype=torch.bool)
 mask=torch.isfinite(x).all(2)&known&torch.isfinite(y)
 count=mask.sum(1);valid=count>=20
 xc=torch.where(mask[:,:,None],x,0);yc=torch.where(mask,y,0)
 dx=torch.where(mask[:,:,None],xc-xc.sum(1)[:,None,:]/count.clamp(min=1)[:,None,None],0)
 dy=torch.where(mask,yc-yc.sum(1)[:,None]/count.clamp(min=1)[:,None],0)
 ic=(dx*dy[:,:,None]).sum(1)/(dx.square().sum(1)*dy.square().sum(1)[:,None]).sqrt()
 ic=ic[valid];avg=torch.nanmean(ic,0).cpu().numpy();dates=a[f'd{year}'][valid.cpu().numpy()]
 for j,name in enumerate(names):out.append(dict(year=year,factor=name,IC=float(avg[j]),days=int(torch.isfinite(ic[:,j]).sum()),first=dates[0],last=dates[-1]))
with open(r/'annual_ic.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
(r/'method.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),metric='Annual mean daily Pearson IC: original factor percentiles vs next24h price excess return; no fees or funding',sample='All15 features finite, known pool for2022+, finite return; omit individual missing rows, at least20 members per day',year2021='Training table2021, retrospective2022pool; not out of sample'),indent=2))
for year in range(2021,2027):print(year,[(v['factor'],round(v['IC'],4)) for v in out if v['year']==year],flush=True)
