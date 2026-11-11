"""Alpha191 subset, 4h adaptation. All rolling/ranking/portfolio arithmetic on CUDA."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import torch
torch.set_default_dtype(torch.float64)
assert torch.cuda.is_available()
assert '4090' in torch.cuda.get_device_name(),torch.cuda.get_device_name()
DEV='cuda';ROOT=Path('/tmp/alpha191_16');ROOT.mkdir(exist_ok=True)
NAMES=[f'alpha191_{n:03d}' for n in [1,2,5,40,63,76,85,102,109,112,128,133,144,145,168,172]]
BAR=14400000
def lag(x,n=1):
 return torch.cat([torch.full_like(x[:n],torch.nan),x[:-n]],0)
def roll(x,n,op='mean'):
 z=x.unfold(0,n,1)
 if op=='mean':v=z.mean(-1)
 elif op=='sum':v=z.sum(-1)
 elif op=='std':v=z.std(-1,correction=1)
 elif op=='min':v=z.min(-1).values
 elif op=='max':v=z.max(-1).values
 return torch.cat([torch.full_like(x[:n-1],torch.nan),v],0)
def div(a,b):return torch.where(b!=0,a/b,torch.nan)
def rank(x,m=None):
 if m is None:m=torch.isfinite(x)
 order=torch.argsort(torch.where(m,x,torch.inf),dim=1,stable=True)
 s=x.gather(1,order);n=x.shape[1];p=torch.arange(n,device=DEV)[None,:].expand_as(x)
 change=torch.ones_like(m);change[:,1:]=s[:,1:]!=s[:,:-1]
 start=torch.cummax(torch.where(change,p,0),1).values
 change=torch.ones_like(m);change[:,:-1]=s[:,:-1]!=s[:,1:]
 end=torch.flip(torch.cummin(torch.flip(torch.where(change,p,n-1),[1]),1).values,[1])
 r=torch.empty_like(x).scatter_(1,order,(start+end).double()/2+1)
 return torch.where(m,r,torch.nan)
def tsrank(x,n):
 z=x.unfold(0,n,1);last=z[:,:,-1:]
 r=(z<last).sum(-1)+(z==last).sum(-1).double()/2+.5
 r=torch.where(torch.isfinite(z).all(-1),r,torch.nan)
 return torch.cat([torch.full_like(x[:n-1],torch.nan),r],0)
def corr(x,y,n):
 a=x.unfold(0,n,1);b=y.unfold(0,n,1)
 a=a-a.mean(-1,keepdim=True);b=b-b.mean(-1,keepdim=True)
 c=div((a*b).sum(-1),(a.square().sum(-1)*b.square().sum(-1)).sqrt())
 return torch.cat([torch.full_like(x[:n-1],torch.nan),c.clamp(-1,1)],0)
def sma(x,n,m):
 # Chinese SMA: seed first finite observation; reset at missing candles.
 out=torch.empty_like(x);state=torch.full_like(x[0],torch.nan)
 for i in range(len(x)):
  state=torch.where(torch.isfinite(state),m/n*x[i]+(1-m/n)*state,x[i]);out[i]=state
 return out
def age(x,n,high):
 z=x.unfold(0,n,1).flip(-1)
 a=z.argmax(-1) if high else z.argmin(-1)
 a=torch.where(torch.isfinite(z).all(-1),a.double(),torch.nan)
 return torch.cat([torch.full_like(x[:n-1],torch.nan),a],0)
def factors(o,h,l,c,v,a):
 pc=lag(c);delta=c-pc;r=div(c,pc)-1;rg=div(2*c-h-l,h-l)
 up=torch.where(torch.isfinite(delta),delta.clamp(min=0),torch.nan)
 dv=v-lag(v);uv=torch.where(torch.isfinite(dv),dv.clamp(min=0),torch.nan)
 typ=(h+l+c)/3;dt=typ-lag(typ);flow=typ*v
 pos=torch.where(dt>0,flow,0.);neg=torch.where(dt<0,flow,0.)
 pos=torch.where(torch.isfinite(dt),pos,torch.nan);neg=torch.where(torch.isfinite(dt),neg,torch.nan)
 p14=roll(pos,14,'sum');n14=roll(neg,14,'sum')
 down=torch.where(torch.isfinite(r),(r<0).double(),torch.nan)
 udvol=torch.where(torch.isfinite(delta),torch.where(delta>0,v,0.),torch.nan)
 ddvol=torch.where(torch.isfinite(delta),torch.where(delta<=0,v,0.),torch.nan)
 hd=h-lag(h);ld=lag(l)-l;tr=torch.maximum(h-l,torch.maximum((h-pc).abs(),(l-pc).abs()))
 plus=torch.where(torch.isfinite(hd)&torch.isfinite(ld),torch.where((hd>0)&(hd>ld),hd,0.),torch.nan)
 minus=torch.where(torch.isfinite(hd)&torch.isfinite(ld),torch.where((ld>0)&(ld>hd),ld,0.),torch.nan)
 pdi=100*div(roll(plus,14,'sum'),roll(tr,14,'sum'));mdi=100*div(roll(minus,14,'sum'),roll(tr,14,'sum'))
 x={
  1:-corr(rank(torch.log(v)-lag(torch.log(v))),rank(div(c-o,o)),6),
  2:-(rg-lag(rg)),
  5:-roll(corr(tsrank(v,5),tsrank(h,5),5),3,'max'),
  40:100*div(roll(udvol,26,'sum'),roll(ddvol,26,'sum')),
  63:100*div(sma(up,6,1),sma(delta.abs(),6,1)),
  76:div(roll(div(r.abs(),v),20,'std'),roll(div(r.abs(),v),20)),
  85:tsrank(div(v,roll(v,20)),20)*tsrank(-(c-lag(c,7)),8),
  102:100*div(sma(uv,6,1),sma(dv.abs(),6,1)),
  109:div(sma(h-l,10,2),sma(sma(h-l,10,2),10,2)),
  112:100*div(roll(delta,12,'sum'),roll(delta.abs(),12,'sum')),
  128:100*div(p14,p14+n14),
  133:5*(age(l,20,False)-age(h,20,True)),
  144:div(roll(torch.where(torch.isfinite(r),torch.where(r<0,div(r.abs(),a),0.),torch.nan),20,'sum'),roll(down,20,'sum')),
  145:100*div(roll(v,9)-roll(v,26),roll(v,12)),
  168:-div(v,roll(v,20)),
  172:roll(100*div((pdi-mdi).abs(),pdi+mdi),6)
 }
 # All factors share 120 consecutive valid 4h bars, stabilizing recursive SMA.
 valid=torch.isfinite(c)&torch.isfinite(v)
 ready=roll(valid.double(),120,'sum')==120
 return torch.stack([torch.where(ready&torch.isfinite(x[int(n[-3:])]),x[int(n[-3:])],torch.nan) for n in NAMES],-1)

def compute():
 bars=pd.read_parquet('/workspace/bars.parquet');labels=pd.read_parquet('/workspace/labels.parquet')
 output=[];ics=[];deciles=[]
 for year in range(2021,2027):
  q=labels.loc[labels.data_year.eq(year)].sort_values(['entry_time_utc','symbol']).copy()
  syms=sorted(q.symbol.unique());dates=sorted(q.entry_time_utc.unique())
  first=pd.Timestamp(dates[0]).value//10**6;last=pd.Timestamp(dates[-1]).value//10**6
  start=first-240*BAR;stamps=np.arange(start,last,BAR,dtype=np.int64)
  b=bars.loc[bars.symbol.isin(syms)&bars.timestamp.between(start,last-BAR)].copy()
  rr=torch.tensor(pd.Index(stamps).get_indexer(b.timestamp),device=DEV);cc=torch.tensor(pd.Index(syms).get_indexer(b.symbol),device=DEV)
  a=torch.full((len(stamps),len(syms),6),torch.nan,device=DEV)
  a[rr,cc]=torch.tensor(b[['open','high','low','close','volume','amount']].to_numpy(),device=DEV)
  good=torch.isfinite(a).all(-1)&(a>0).all(-1)
  a=torch.where(good[:,:,None],a,torch.nan)
  x=factors(*a.unbind(-1))
  indices=pd.Index(stamps+BAR).get_indexer(pd.DatetimeIndex(dates).as_unit('ms').asi8)
  assert (indices>=0).all();x=x[torch.tensor(indices,device=DEV)]
  D,N,F=x.shape;assert len(q)==D*N
  known=torch.ones((D,N),device=DEV,dtype=torch.bool) if year==2021 else torch.tensor(q.pool_known_at_decision.to_numpy().reshape(D,N),device=DEV)
  x=torch.where(known[:,:,None],x,torch.nan)
  q[NAMES]=x.reshape(-1,F).cpu().numpy();output.append(q)
  y=torch.tensor(q.realized_return_24h.to_numpy().reshape(D,N),device=DEV)
  bench=torch.tensor(q.benchmark_return_24h.to_numpy().reshape(D,N)[:,0],device=DEV)
  ds=q.drop_duplicates('entry_time_utc').decision_date_beijing.to_numpy()
  for j,name in enumerate(NAMES):
   v=x[:,:,j];eligible=torch.isfinite(v);m=eligible&torch.isfinite(y);count=m.sum(1)
   xr=rank(v,m);yr=rank(y,m)
   dx=torch.where(m,xr-torch.nanmean(xr,1)[:,None],0.);dy=torch.where(m,yr-torch.nanmean(yr,1)[:,None],0.)
   ic=div((dx*dy).sum(1),(dx.square().sum(1)*dy.square().sum(1)).sqrt())
   ok=(count>=20)&torch.isfinite(ic)
   ics.append(pd.DataFrame(dict(year=year,factor=name,date=ds,rank_ic=torch.where(ok,ic,torch.nan).cpu().numpy(),members=count.cpu().numpy())))
   order=torch.argsort(torch.where(eligible,v,torch.inf),dim=1,stable=True);cnt=eligible.sum(1)
   # Group membership uses no future return. Unknown outcomes omit that diagnostic day.
   valid=(cnt>=10)&((~eligible)|torch.isfinite(y)).all(1)&torch.isfinite(bench)
   sy=torch.nan_to_num(y,nan=0).gather(1,order)
   prefix=torch.cat([torch.zeros((D,1),device=DEV),sy.cumsum(1)],1)
   for g in range(10):
    size=cnt//10+(cnt%10>g);st=g*(cnt//10)+torch.minimum(cnt%10,torch.full_like(cnt,g))
    gr=div(prefix.gather(1,(st+size)[:,None]).squeeze(1)-prefix.gather(1,st[:,None]).squeeze(1),size)
    deciles.append(pd.DataFrame(dict(year=year,factor=name,date=ds,decile=g+1,excess_return=torch.where(valid,gr-bench,torch.nan).cpu().numpy(),group_return=torch.where(valid,gr,torch.nan).cpu().numpy(),members=size.cpu().numpy())))
  print('GPU factors / IC / deciles',year,'rows',len(q),flush=True)
 q=pd.concat(output,ignore_index=True);q.to_parquet(ROOT/'factor_values.parquet',index=False)
 ic=pd.concat(ics,ignore_index=True);ic.to_csv(ROOT/'daily_rank_ic.csv',index=False)
 annual=ic.groupby(['year','factor']).rank_ic.agg(['mean','std','count']).reset_index().rename(columns={'mean':'RankIC','count':'days'})
 annual['RankICIR']=annual.RankIC/annual['std'];annual.to_csv(ROOT/'annual_rank_ic.csv',index=False)
 overall=ic.groupby('factor').rank_ic.agg(['mean','std','count']);overall['RankICIR']=overall['mean']/overall['std'];overall.to_csv(ROOT/'overall_rank_ic.csv')
 pd.concat(deciles,ignore_index=True).to_parquet(ROOT/'decile_daily.parquet',index=False)
 return q

def portfolio(q):
 syms=sorted(q.symbol.unique());dates=pd.date_range(q.decision_date_beijing.min(),'2026-10-28',tz='Asia/Shanghai');stamps=dates.tz_convert('UTC').as_unit('ms').asi8
 D,N,F=len(dates),len(syms),len(NAMES)
 ri=torch.tensor(pd.Index(stamps).get_indexer(q.entry_time_utc.dt.as_unit('ms').astype('int64')),device=DEV)
 ci=torch.tensor(pd.Index(syms).get_indexer(q.symbol),device=DEV)
 x=torch.full((D,N,F),torch.nan,device=DEV);y=torch.full((D,N),torch.nan,device=DEV);px=torch.full_like(y,torch.nan)
 x[ri,ci]=torch.tensor(q[NAMES].to_numpy(),device=DEV);y[ri,ci]=torch.tensor(q.realized_return_24h.to_numpy(),device=DEV);px[ri,ci]=torch.tensor(q.entry_open.to_numpy(),device=DEV)
 ex=q[['symbol','exit_time_utc','exit_open']].dropna();er=pd.Index(stamps).get_indexer(ex.exit_time_utc.dt.as_unit('ms').astype('int64'));ec=pd.Index(syms).get_indexer(ex.symbol);valid=er>=0
 er=torch.tensor(er[valid],device=DEV);ec=torch.tensor(ec[valid],device=DEV)
 px[er,ec]=torch.where(torch.isfinite(px[er,ec]),px[er,ec],torch.tensor(ex.exit_open.to_numpy()[valid],device=DEV))
 eligible=torch.isfinite(x)&torch.isfinite(px)[:,:,None]&(px>0)[:,:,None]
 count=eligible.sum(1);eligible&=(count>=10)[:,None,:];count=eligible.sum(1)
 order=torch.argsort(torch.where(eligible,x,torch.inf),dim=1,stable=True)
 ranks=torch.empty_like(order).scatter_(1,order,torch.arange(N,device=DEV)[None,:,None].expand(D,N,F))
 highrank=count[:,None,:]-1-ranks
 keep=torch.ceil(count*.3).long();lowN=count//10+(count%10>0);highN=count//10
 low=torch.zeros_like(eligible);high=torch.zeros_like(eligible)
 for target,rk,size in [(low,ranks,lowN),(high,highrank,highN)]:
  previous=torch.zeros((N,F),device=DEV,dtype=torch.bool)
  for i in range(D):
   retain=previous&eligible[i]&(rk[i]<keep[i][None,:])
   priority=torch.where(retain,rk[i],rk[i]+N)
   priority=torch.where(eligible[i],priority,3*N)
   ord=torch.argsort(priority,dim=0,stable=True)
   pos=torch.empty_like(ord).scatter_(0,ord,torch.arange(N,device=DEV)[:,None].expand(N,F))
   # Fill from leading10%; remaining capacity only relevant for impossible/degenerate inputs.
   chosen=(pos<size[i][None,:])&eligible[i]&(retain|(rk[i]<size[i][None,:]))
   target[i]=chosen;previous=chosen
 weights=(high.double()/high.sum(1).clamp(min=1)[:,None,:]-low.double()/low.sum(1).clamp(min=1)[:,None,:])*.5
 annual=[];daily=[]
 for direction,sign in [('high_long_low_short',1),('low_long_high_short',-1)]:
  w=weights*sign;g=(w*torch.nan_to_num(y,nan=0)[:,:,None]).sum(1)
  old=torch.cat([torch.zeros_like(w[:1]),w[:-1]]);pg=torch.cat([torch.zeros_like(g[:1]),g[:-1]])
  pp=torch.cat([px[:1],px[:-1]]);good=torch.isfinite(px)&torch.isfinite(pp)&(px>0)&(pp>0)
  ratio=torch.where(good,px/pp,1.);before=old*ratio[:,:,None]/(1+pg[:,None,:])
  fee=torch.zeros_like(g)
  for _ in range(20):fee=.0005*((1-fee[:,None,:])*w-before).abs().sum(1)
  net=(1-fee)*(1+g)-1
  assert (1+net>0).all(),'Portfolio insolvency requires separate handling'
  missing=(w.ne(0)&~torch.isfinite(y)[:,:,None]).any(1);marks=(old.ne(0)&~good[:,:,None]).any(1)
  for year in range(2021,2027):
   m=torch.tensor(dates.year==year,device=DEV);r=net[m];eq=torch.cat([torch.ones((1,F),device=DEV),(1+r).cumprod(0)])
   dd=(1-eq/eq.cummax(0).values).max(0).values;ret=eq[-1]-1;sh=r.mean(0)/r.std(0,correction=1)*365**.5
   cal=(eq[-1]**(365/len(r))-1)/dd
   for j,name in enumerate(NAMES):annual.append(dict(year=year,factor=name,direction=direction,net_return=ret[j].item(),max_drawdown=dd[j].item(),sharpe=sh[j].item(),calmar=cal[j].item(),oneway_turnover=(fee[m,j].mean()/.001).item(),estimated_return_days=int(missing[m,j].sum()),estimated_fee_days=int(marks[m,j].sum())))
  for j,name in enumerate(NAMES):daily.append(pd.DataFrame(dict(date=dates.strftime('%Y-%m-%d'),factor=name,direction=direction,net_return=net[:,j].cpu().numpy(),fee=fee[:,j].cpu().numpy(),long_members=(high if sign==1 else low)[:,:,j].sum(1).cpu().numpy(),short_members=(low if sign==1 else high)[:,:,j].sum(1).cpu().numpy(),estimated_return=missing[:,j].cpu().numpy(),estimated_fee=marks[:,j].cpu().numpy())))
 pd.DataFrame(annual).to_csv(ROOT/'annual_net_metrics.csv',index=False);pd.concat(daily,ignore_index=True).to_parquet(ROOT/'daily_net.parquet',index=False)
 print('GPU buffered portfolios complete',flush=True)

if __name__=='__main__':
 print('GPU',torch.cuda.get_device_name(),flush=True)
 q=compute();portfolio(q)
 method=dict(gpu=torch.cuda.get_device_name(),factors=NAMES,source='JoinQuant/jqdatasdk alpha191.py formulas; 4h adaptation, not platform-verified bitwise replication',
  periods='4h bars; daily signals Beijing00; only bars closed by signal',warmup='120 consecutive positive OHLCV/amount bars; 240 bars prehistory loaded per year; SMA seed first finite and reset on missing',
  ranks='RANK cross-section of fixed annual pool at each4h timestamp; TSRANK time-series; average ties; STD sample ddof1; extrema ages choose latest tie',
  corrections='005 YSRANK interpreted TSRANK; 133 age counts current bar0; 172 standard HD=high-lag(high), LD=lag(low)-low, TR; 128 ratio algebra avoids infinity when negative money flow zero',
  years='2021 retrospective2022pool;2022-2026 same-year pools. 2026 final signal Oct27, return/exit Oct28',
  IC='Daily Spearman using individual finite factor and realized-price rows; minimum20; ICIR not annualized',
  deciles='G1 low G10 high; stable symbol ties array_split sizes; full-pool supplied equalweight benchmark; simple gross excess; incomplete eligible outcomes omit entire diagnostic day',
  portfolio='Both fixed directions reported; each side50% postfee equity, top/bottom10%,30% retention, daily equalweight, compounded; no annual hindsight flips',
  fee='0.0005 per actual traded notional including drift, initial/year changes/final exits; oneway turnover=traded notional/2',
  missing='Holdings never selected by future labels; missing held return0 and missing mark unchanged, explicitly flagged; no funding or slippage',
  disclaimer='Research screen on already examined years, not new out-of-sample proof; no XGBoost training')
 (ROOT/'method.json').write_text(json.dumps(method,indent=2))
