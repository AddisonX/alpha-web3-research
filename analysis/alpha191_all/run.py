from pathlib import Path
import json,time
import numpy as np,pandas as pd,torch
import formulas_gpu as f
ROOT=Path('/tmp/alpha191_all_pct');ROOT.mkdir(exist_ok=True)
OLD=Path('/tmp/alpha191_16');BAR=14400000
NAMES=[f'alpha191_{n:03d}' for n in range(1,192) if n not in f.EXCLUDED]
REUSE=[]
assert torch.cuda.is_available();torch.set_default_dtype(torch.float64)
print('GPU',torch.cuda.get_device_name(),'179 factors; unified internal percentile ranks',flush=True)
bars=pd.read_parquet('/workspace/bars.parquet');labels=pd.read_parquet('/workspace/labels.parquet')
old=pd.read_parquet(OLD/'factor_values.parquet');ics=[];deciles=[];cover=[]
for year in range(2021,2026):
 q=labels.loc[labels.data_year.eq(year)].sort_values(['entry_time_utc','symbol']).copy().reset_index(drop=True)
 syms=sorted(q.symbol.unique());dates=sorted(q.entry_time_utc.unique());D=len(dates);N=len(syms);assert len(q)==D*N
 first=pd.Timestamp(dates[0]).value//10**6;last=pd.Timestamp(dates[-1]).value//10**6;start=first-360*BAR;stamps=np.arange(start,last,BAR,dtype=np.int64)
 bb=bars.loc[bars.symbol.isin(syms)&bars.timestamp.between(start,last-BAR)]
 rr=torch.tensor(pd.Index(stamps).get_indexer(bb.timestamp),device='cuda');cc=torch.tensor(pd.Index(syms).get_indexer(bb.symbol),device='cuda')
 a=torch.full((len(stamps),N,6),torch.nan,device='cuda');a[rr,cc]=torch.tensor(bb[['open','high','low','close','volume','amount']].to_numpy(),device='cuda')
 good=torch.isfinite(a).all(-1)&(a>0).all(-1);a=torch.where(good[:,:,None],a,torch.nan)
 ready=f.summ(good.double(),120)==120
 idx=pd.Index(stamps+BAR).get_indexer(pd.DatetimeIndex(dates).as_unit('ms').asi8);assert (idx>=0).all();idx=torch.tensor(idx,device='cuda')
 known=torch.ones((D,N),device='cuda',dtype=torch.bool) if year==2021 else torch.tensor(q.pool_known_at_decision.to_numpy().reshape(D,N),device='cuda')
 calc=f.Formulas(*a.unbind(-1));year_values={};ds=q.drop_duplicates('entry_time_utc').decision_date_beijing.to_numpy()
 y=torch.tensor(q.realized_return_24h.to_numpy().reshape(D,N),device='cuda');bench=torch.tensor(q.benchmark_return_24h.to_numpy().reshape(D,N)[:,0],device='cuda')
 oldy=old.loc[old.data_year.eq(year)].set_index(['entry_time_utc','symbol']).reindex(pd.MultiIndex.from_frame(q[['entry_time_utc','symbol']]))
 t=time.time()
 for n,fn in sorted(calc.functions.items()):
  name=f'alpha191_{n:03d}'
  if n in REUSE:x=torch.tensor(oldy[name].to_numpy().reshape(D,N),device='cuda')
  else:
   z=fn();assert z.shape==ready.shape,(n,z.shape,ready.shape)
   x=torch.where(ready&torch.isfinite(z),z,torch.nan)[idx];x=torch.where(known,x,torch.nan)
  year_values[name]=x.reshape(-1).cpu().numpy()
  eligible=torch.isfinite(x);m=eligible&torch.isfinite(y);count=m.sum(1);xr=f.rank(x,m);yr=f.rank(y,m)
  dx=torch.where(m,xr-torch.nanmean(xr,1)[:,None],0.);dy=torch.where(m,yr-torch.nanmean(yr,1)[:,None],0.)
  ic=f.div((dx*dy).sum(1),(dx.square().sum(1)*dy.square().sum(1)).sqrt());ok=(count>=20)&torch.isfinite(ic)
  ics.append(pd.DataFrame(dict(year=year,factor=name,date=ds,rank_ic=torch.where(ok,ic,torch.nan).cpu().numpy(),members=count.cpu().numpy())))
  order=torch.argsort(torch.where(eligible,x,torch.inf),dim=1,stable=True);cnt=eligible.sum(1)
  valid=(cnt>=10)&((~eligible)|torch.isfinite(y)).all(1)&torch.isfinite(bench)
  sy=torch.nan_to_num(y,nan=0).gather(1,order);prefix=torch.cat([torch.zeros((D,1),device='cuda'),sy.cumsum(1)],1)
  sortedx=x.gather(1,order);unique=(sortedx[:,1:]!=sortedx[:,:-1])&torch.isfinite(sortedx[:,1:]);unique=unique.sum(1)+(cnt>0)
  cover.append(dict(year=year,factor=name,IC_days=int(ok.sum()),decile_days=int(valid.sum()),mean_factor_members=float(cnt.double().mean()),mean_unique_values=float(unique.double().mean()),reused=n in REUSE))
  for g in range(10):
   size=cnt//10+(cnt%10>g);st=g*(cnt//10)+torch.minimum(cnt%10,torch.full_like(cnt,g))
   gr=f.div(prefix.gather(1,(st+size)[:,None]).squeeze(1)-prefix.gather(1,st[:,None]).squeeze(1),size)
   deciles.append(pd.DataFrame(dict(year=year,factor=name,date=ds,decile=g+1,excess_return=torch.where(valid,gr-bench,torch.nan).cpu().numpy(),group_return=torch.where(valid,gr,torch.nan).cpu().numpy(),members=size.cpu().numpy())))
  if n%20==0:print(year,'through',n,'seconds',round(time.time()-t,1),flush=True)
 pd.concat([q,pd.DataFrame(year_values)],axis=1).to_parquet(ROOT/f'factor_values_{year}.parquet',index=False,compression='zstd')
 print('YEAR COMPLETE',year,round(time.time()-t,1),flush=True)
 del calc,a
ic=pd.concat(ics,ignore_index=True);ic.to_parquet(ROOT/'daily_rank_ic.parquet',index=False)
annual=ic.groupby(['factor','year']).rank_ic.agg(['mean','std','count']).reset_index().rename(columns={'mean':'RankIC','count':'days'});annual['RankICIR']=annual.RankIC/annual['std'];annual.to_csv(ROOT/'annual_rank_ic.csv',index=False)
dec=pd.concat(deciles,ignore_index=True);dec.to_parquet(ROOT/'decile_daily.parquet',index=False,compression='zstd')
ad=dec.groupby(['factor','year','decile']).excess_return.agg(['sum','count']).reset_index().rename(columns={'sum':'excess_simple_return','count':'days'});ad.loc[ad.days.eq(0),'excess_simple_return']=np.nan;ad.to_csv(ROOT/'annual_decile_excess.csv',index=False)
overall=ic.groupby('factor').rank_ic.agg(['mean','std','count']).rename(columns={'mean':'RankIC','count':'days'});overall['RankICIR']=overall.RankIC/overall['std'];overall.to_csv(ROOT/'overall_rank_ic.csv')
rows=[]
for name,r in overall.iterrows():
 z=ad.loc[ad.factor.eq(name)].pivot(index='year',columns='decile',values='excess_simple_return').reindex(range(2021,2026));long=10 if r.RankIC>0 else 1;short=1 if r.RankIC>0 else 10
 rec=dict(factor=name,RankIC=r.RankIC,RankICIR=r.RankICIR,IC_days=int(r.days),long_group=long,short_group=short,long_positive_years=int((z[long]>0).sum()),short_negative_years=int((z[short]<0).sum()))
 for yr in range(2021,2026):
  rec[f'G1_{yr}']=z.loc[yr,1];rec[f'G10_{yr}']=z.loc[yr,10];rec[f'IC_{yr}']=annual.loc[annual.factor.eq(name)&annual.year.eq(yr),'RankIC'].iloc[0]
 rec['G1_positive_years']=int((z[1]>0).sum());rec['G1_negative_years']=int((z[1]<0).sum());rec['G10_positive_years']=int((z[10]>0).sum());rec['G10_negative_years']=int((z[10]<0).sum())
 rec['pass_IC']=bool(abs(r.RankIC)>.03);rec['pass_endpoint']=rec['long_positive_years']>=4 or rec['short_negative_years']>=4;rec['selected']=rec['pass_IC'] and rec['pass_endpoint']
 rec['any_fixed_endpoint_selected']=rec['pass_IC'] and max(rec['G1_positive_years'],rec['G1_negative_years'],rec['G10_positive_years'],rec['G10_negative_years'])>=4
 rec['selected_ic_aligned']=rec['selected'];rec['selected']=rec['any_fixed_endpoint_selected']
 rows.append(rec)
screen=pd.DataFrame(rows).sort_values('factor');screen.to_csv(ROOT/'all_factor_screen.csv',index=False);screen.loc[screen.selected].to_csv(ROOT/'selected_factors.csv',index=False)
pd.DataFrame(cover).to_csv(ROOT/'coverage_and_ties.csv',index=False)
pd.DataFrame([dict(factor=f'alpha191_{n:03d}',reason=s) for n,s in sorted(f.EXCLUDED.items())]).to_csv(ROOT/'not_reproduced.csv',index=False)
(ROOT/'method.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),source='https://github.com/JoinQuant/jqdatasdk/blob/master/jqdatasdk/alpha191.py',computed=len(NAMES),reused=REUSE,excluded=f.EXCLUDED,formula_interpretations=f.NOTES,periods='4h bars; daily Beijing00 closed-bar signals; original annual pools;2021 retrospectively2022pool',warmup='120 consecutive valid bars;360 prehistory for all179 factors',RANK='Average ordinal divided by valid cross-sectional count; TSRANK average ordinal divided by window; ties stable symbol for deciles',WMA='Normalized geometrically decaying weights0.9^age; DECAYLINEAR weights1..n; STD/cov sample',IC='Overall mean daily cross-sectional Spearman2021-2025; minimum20 paired observations;ICIR mean/sampleSTD unannualized',selection='abs(overall RankIC)>0.03; test four fixed legs G1-long/G1-short/G10-long/G10-short; require at least one same leg with correct excess sign in>=4years; no requirement that endpoint direction agrees with overallIC sign; IC-aligned flag also retained',excess='G1 lowest G10 highest; each equalweight top/bottom10%; daily return minus full-pool supplied equalweight benchmark; annual simple sum; gross before fee/funding/slippage; no retentionbuffer',missing='No future labels used to form groups; omit diagnostic date if any factor-eligible outcome missing; coverage reported',caveat='In-sample research selection; formula adaptations not authenticated JoinQuant bitwise replication'),indent=2),encoding='utf-8')
print('COUNTS',len(screen),'IC',int(screen.pass_IC.sum()),'SELECTED',int(screen.selected.sum()),'ANY_ENDPOINT',int(screen.any_fixed_endpoint_selected.sum()),flush=True)
print(screen.loc[screen.selected,['factor','RankIC','long_group','long_positive_years','short_group','short_negative_years']].to_string(index=False),flush=True)
