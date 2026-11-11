import sys
sys.path.append('/workspace/alpha-web3-research/.venv/lib/python3.12/site-packages')
import torch,pandas as pd,numpy as np,json
from pathlib import Path
BASE=Path('/tmp/alpha15-run/data/model_tables/factors15_2021_2026')
OUT=Path('/tmp/ten_factor_direction_net');OUT.mkdir(exist_ok=True)
names=json.loads((BASE/'feature_columns.json').read_text())[:10]
signs=[1,1,1,1,-1,1,-1,-1,-1,-1]
frames=[pd.read_parquet(BASE/f'predict_table_2022-2026/predict_table_{y}.parquet') for y in range(2022,2027)]
df=pd.concat(frames,ignore_index=True)
dates=pd.date_range('2022-01-02','2026-10-28',tz='Asia/Shanghai')
stamps=dates.tz_convert('UTC').as_unit('ms').asi8
symbols=sorted(df.symbol.unique());D=len(dates);N=len(symbols);F=len(names)
df=df.loc[df.entry_time_utc.dt.as_unit('ms').astype('int64').isin(stamps)]
ri=pd.Index(stamps).get_indexer(df.entry_time_utc.dt.as_unit('ms').astype('int64'));ci=pd.Index(symbols).get_indexer(df.symbol)
dev='cuda';assert torch.cuda.is_available()
rr=torch.tensor(ri,device=dev);cc=torch.tensor(ci,device=dev)
x=torch.full((D,N,F),torch.nan,device=dev,dtype=torch.float64)
y=torch.full((D,N),torch.nan,device=dev,dtype=torch.float64)
px=torch.full_like(y,torch.nan);known=torch.zeros((D,N),device=dev,dtype=torch.bool)
x[rr,cc]=torch.tensor(df[names].to_numpy(),device=dev,dtype=torch.float64)
y[rr,cc]=torch.tensor(df.realized_return_24h.to_numpy(),device=dev,dtype=torch.float64)
px[rr,cc]=torch.tensor(df.entry_open.to_numpy(),device=dev,dtype=torch.float64)
known[rr,cc]=torch.tensor(df.pool_known_at_decision.to_numpy(),device=dev)
# Use preceding exit prices where the next row's entry observation is absent.
ex=pd.concat([p[['symbol','exit_time_utc','exit_open']] for p in frames]).dropna(subset=['exit_open'])
ex=ex.loc[ex.exit_time_utc.dt.as_unit('ms').astype('int64').isin(stamps)]
er=torch.tensor(pd.Index(stamps).get_indexer(ex.exit_time_utc.dt.as_unit('ms').astype('int64')),device=dev)
ec=torch.tensor(pd.Index(symbols).get_indexer(ex.symbol),device=dev)
px[er,ec]=torch.where(torch.isfinite(px[er,ec]),px[er,ec],torch.tensor(ex.exit_open.to_numpy(),device=dev,dtype=torch.float64))
eligible=torch.isfinite(x)&known[:,:,None]&torch.isfinite(px)[:,:,None]&px.gt(0)[:,:,None]
eligible[-1]=False
counts=eligible.sum(1);eligible&=(counts>=10)[:,None,:];counts=eligible.sum(1)
order=torch.argsort(torch.where(eligible,x,torch.inf),dim=1,stable=True)
rank=torch.empty_like(order);rank.scatter_(1,order,torch.arange(N,device=dev)[None,:,None].expand(D,N,F))
low=(rank<(counts//10+(counts%10>0))[:,None,:])&eligible
high=(rank>=(counts-counts//10)[:,None,:])&eligible
select=high.clone();select[:,:,:4]=low[:,:,:4]
w=select.to(torch.float64)/select.sum(1).clamp(min=1)[:,None,:]*torch.tensor(signs,device=dev)[None,None,:]
g=(w*torch.nan_to_num(y,nan=0)[:,:,None]).sum(1)
missing_y=(w.ne(0)&~torch.isfinite(y)[:,:,None]).any(1)
old=torch.cat([torch.zeros_like(w[:1]),w[:-1]])
prior_g=torch.cat([torch.zeros_like(g[:1]),g[:-1]])
prior_px=torch.cat([px[:1],px[:-1]])
good=torch.isfinite(px)&torch.isfinite(prior_px)&px.gt(0)&prior_px.gt(0)
ratio=torch.where(good,px/prior_px,torch.ones_like(px))
before=old*ratio[:,:,None]/(1+prior_g[:,None,:])
missing_marks=(old.ne(0)&~good[:,:,None]).any(1)
assert (1+g>0).all(),'A daily short loss exceeded portfolio equity; insolvency handling needed'
f=torch.zeros_like(g)
for _ in range(20):f=.0005*((1-f[:,None,:])*w-before).abs().sum(1)
net=(1-f)*(1+g)-1
assert (1+net>0).all(),'Net equity exhausted'
annual=[];daily=[]
for yr in range(2022,2027):
    mask=torch.tensor(dates.year==yr,device=dev)
    result=torch.prod(1+net[mask],dim=0)-1
    for j,name in enumerate(names):
        annual.append(dict(year=yr,factor=name,direction='long' if signs[j]==1 else 'short',decile='G1' if j<4 else 'G10',net_compound_return=result[j].item(),mean_daily_oneway_turnover=(f[mask,j].mean()/.001).item(),estimated_return_days=missing_y[mask,j].sum().item(),estimated_fee_days=missing_marks[mask,j].sum().item()))
for j,name in enumerate(names):
    daily.append(pd.DataFrame(dict(date_beijing=dates.strftime('%Y-%m-%d'),factor=name,gross_return=g[:,j].cpu().numpy(),fee_fraction=f[:,j].cpu().numpy(),net_return=net[:,j].cpu().numpy(),members=select[:,:,j].sum(1).cpu().numpy(),return_estimated=missing_y[:,j].cpu().numpy(),fee_estimated=missing_marks[:,j].cpu().numpy())))
a=pd.DataFrame(annual);a.to_csv(OUT/'annual_net_returns.csv',index=False)
p=a.pivot(index='factor',columns='year',values='net_compound_return').reindex(names);p.to_csv(OUT/'annual_net_returns_wide.csv')
pd.concat(daily).to_parquet(OUT/'daily_net_returns.parquet',index=False)
(OUT/'method.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),directions='Original persistent-extreme baseline pass; none of10 pass both sides. Direction frozen across years; retrospective selection.',
    eligibility='Each factor independently finite, known annual pool, current positive entry price; minimum10 eligible. Future label availability does not affect holdings.',
    portfolio='Direct original decile selection, no15% buffer; 100% post-fee equity notional on selected side, equalweights daily. G1 low four marketcap factors, G10 high others.',
    fees='0.05% per traded notional, price and equity drift included; initial/cross-year/final exits charged',
    returns='Actual raw long or short24h return, compounded after fees, not benchmark excess. No funding or slippage.',
    missing='Missing held return0, missing price mark unchanged; flagged in outputs.',horizon='2022-01-02 through2026-10-28; last holdingOct27'),indent=2))
print(p.to_string(),flush=True)
print(a.groupby('factor')[['estimated_return_days','estimated_fee_days']].sum().to_string(),flush=True)
