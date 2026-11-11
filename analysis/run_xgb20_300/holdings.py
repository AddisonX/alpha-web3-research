"""CUDA theoretical target-holdings turnover, marking old positions at current open."""
import sys

import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

MODEL=Path('/tmp/xgb20_gpu_expanding_300_depth4_lr005_2022_2026')
BASE=Path('/tmp/xgb22run/data/model_tables/factors22_2021_2026')
OUT=Path('/tmp/xgb20_300_lr005_holdings_turnover');OUT.mkdir(exist_ok=True)
predictions=[];prices=[]
scores=pd.read_parquet(MODEL/'all_years_decile_daily.parquet')
end=int(scores.decision_timestamp.max())+86400000
for year in range(2022,2027):
    p=pd.read_parquet(MODEL/str(year)/'predictions_with_realized_labels.parquet')
    p=p.loc[p.entry_time_utc.dt.as_unit('ms').astype('int64')<end]
    predictions.append(p[['symbol','entry_time_utc','prediction_decile']])
    q=pd.read_parquet(BASE/f'predict_table_2022-2026/predict_table_{year}.parquet',columns=['symbol','entry_time_utc','exit_time_utc','entry_open','exit_open'])
    prices.extend([q[['symbol','entry_time_utc','entry_open']].rename(columns={'entry_time_utc':'time','entry_open':'price'}),
                   q[['symbol','exit_time_utc','exit_open']].rename(columns={'exit_time_utc':'time','exit_open':'price'})])
pred=pd.concat(predictions,ignore_index=True)
start=int(pred.entry_time_utc.dt.as_unit('ms').astype('int64').min())
stamps=np.arange(start,end+1,86400000,dtype=np.int64)
dates=pd.to_datetime(stamps,unit='ms',utc=True)
symbols=sorted(pred.symbol.unique());D=len(stamps);N=len(symbols)
ri=pd.Index(stamps).get_indexer(pred.entry_time_utc.dt.as_unit('ms').astype('int64'))
ci=pd.Index(symbols).get_indexer(pred.symbol)
price=pd.concat(prices,ignore_index=True).dropna(subset='price').drop_duplicates(['symbol','time'])
price=price.pivot(index='time',columns='symbol',values='price').reindex(index=dates,columns=symbols)
assert not pred.duplicated(['symbol','entry_time_utc']).any()
torch.cuda.synchronize();begin=time.perf_counter()
device='cuda';assert torch.cuda.is_available()
weights=torch.zeros((D,N,11),dtype=torch.float64,device=device)
groups=torch.tensor(pred.prediction_decile.to_numpy()-1,device=device,dtype=torch.int64)
rr=torch.tensor(ri,device=device);cc=torch.tensor(ci,device=device)
weights[rr,cc,groups]=1.
counts=weights[:,:,:10].sum(dim=1)
weights[:,:,:10]/=counts.clamp(min=1)[:,None,:]
weights[:,:,10]=.5*weights[:,:,9]-.5*weights[:,:,0]
px=torch.tensor(price.to_numpy(),dtype=torch.float64,device=device)
old=torch.cat([torch.zeros_like(weights[:1]),weights[:-1]],dim=0)
prior_price=torch.cat([px[:1],px[:-1]],dim=0)
good_px=torch.isfinite(px)&px.gt(0)
good_prior=torch.isfinite(prior_price)&prior_price.gt(0)
needed_old=old.ne(0);needed_new=weights.ne(0)
observable=((~needed_old)|(good_px&good_prior)[:,:,None]).all(dim=1)&((~needed_new)|good_px[:,:,None]).all(dim=1)
ratio=torch.where(good_px&good_prior,px/prior_price,torch.zeros_like(px))
marked=old*ratio[:,:,None]
traded=(weights-marked).abs().sum(dim=1)
oneway=.5*traded
oneway=torch.where(observable,oneway,torch.nan)
target_only=.5*(weights-old).abs().sum(dim=1)
long_count=counts[:,9];short_count=counts[:,0];total_count=long_count+short_count
years=dates.tz_convert('Asia/Shanghai').year.to_numpy()
rows=[];legrows=[]
for year in range(2022,2027):
    ym=torch.tensor(years==year,device=device)
    active=ym&total_count.gt(0)
    rec={'year':year,'active_holding_days':int(active.sum().item()),
         'long_min':int(long_count[active].min().item()),'long_max':int(long_count[active].max().item()),
         'short_min':int(short_count[active].min().item()),'short_max':int(short_count[active].max().item()),
         'combined_min':int(total_count[active].min().item()),'combined_max':int(total_count[active].max().item()),
         'mean_long_members':float(long_count[active].mean().item()),'mean_short_members':float(short_count[active].mean().item()),
         'mean_daily_oneway_turnover':float(oneway[ym,10].nanmean().item()),
         'mean_daily_two_way_traded_over_gross':float((2*oneway[ym,10]).nanmean().item()),
         'mean_daily_long_oneway_turnover':float(oneway[ym,9].nanmean().item()),
         'mean_daily_short_oneway_turnover':float(oneway[ym,0].nanmean().item()),
         'mean_daily_target_only_oneway_turnover':float(target_only[ym,10].mean().item()),
         'unpriced_turnover_days':int((ym&~observable[:,10]).sum().item()),
         'flat_target_days':int((ym&total_count.eq(0)).sum().item()),'calendar_days':int(ym.sum().item())}
    rows.append(rec)
    for g in range(10):
        ga=ym&counts[:,g].gt(0)
        legrows.append({'year':year,'decile':g+1,'members_min':int(counts[ga,g].min().item()),
                        'members_max':int(counts[ga,g].max().item()),
                        'mean_daily_oneway_turnover':float(oneway[ym,g].nanmean().item())})
torch.cuda.synchronize();seconds=time.perf_counter()-begin
daily=pd.DataFrame({'date_beijing':dates.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d'),
                    'long_members':long_count.cpu().numpy(),'short_members':short_count.cpu().numpy(),
                    'combined_members':total_count.cpu().numpy(),
                    'oneway_turnover':oneway[:,10].cpu().numpy(),'two_way_traded_over_gross':traded[:,10].cpu().numpy(),
                    'target_only_oneway_turnover':target_only[:,10].cpu().numpy(),
                    'priced':observable[:,10].cpu().numpy()})
daily.loc[~daily.priced,'two_way_traded_over_gross']=np.nan
daily.to_csv(OUT/'daily_holdings_turnover.csv',index=False)
result=pd.DataFrame(rows);result.to_csv(OUT/'annual_holdings_turnover.csv',index=False)
pd.DataFrame(legrows).to_csv(OUT/'all_decile_holdings_turnover.csv',index=False)
all_mean=float(oneway[:,10].nanmean().item())
meta={'GPU':torch.cuda.get_device_name(),'gpu_seconds':seconds,'model':str(MODEL),
      'portfolio':'long G10, short G1, equal weight within each side, each side50% of target gross; scale invariant gross-normalized turnover',
      'oneway_turnover':'sum(abs(target signed notional minus previous target shares marked at current open)) / (2 * constant target gross notional)',
      'two_way_turnover':'sum absolute traded notional / target gross, twice one-way; full replacement100% one-way /200% two-way',
      'constant_size':'fixed gross notional, not compounded NAV; price drift and cross-side switches included; no costs or fill simulation',
      'calendar':'all calendar days from first forecast to final scored exit; forecasts remain active even if missing outcome prevents daily scoring',
      'no_signal_policy':'flat target on dates without forecasts, including Jan1 before annual pool is known; year-end liquidation/re-entry and terminal liquidation included',
      'holdings_count':'min/max over nonempty target days; flat dates reported separately; long/short counts differ because deciles have unequal integer sizes',
      'missing_prices':'unpriceable transitions are excluded from priced turnover average and counted, not silently priced at zero; target-only turnover remains available; theoretical execution intent only',
      'overall_mean_daily_oneway':all_mean,'overall_mean_daily_two_way':all_mean*2,
      'last_scored_decision':str(pd.to_datetime(end-86400000,unit='ms',utc=True).tz_convert('Asia/Shanghai').date()),
      'terminal_liquidation':str(dates[-1].tz_convert('Asia/Shanghai').date())}
(OUT/'method.json').write_text(json.dumps(meta,indent=2))
print(result.to_string(index=False),flush=True);print(json.dumps(meta,indent=2),flush=True)
