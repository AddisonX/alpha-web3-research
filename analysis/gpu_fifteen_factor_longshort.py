"""CUDA sorts and decile aggregation for individual percentile factors, no direction fitting."""
import sys
sys.path.append('/workspace/alpha-web3-research/.venv/lib/python3.12/site-packages')
import json,time
from pathlib import Path
import torch
import pandas as pd

BASE=Path('/tmp/alpha15-run/data/model_tables/factors15_2021_2026')
OUT=Path('/tmp/individual_15factor_longshort_2025_2026');OUT.mkdir(exist_ok=True)
names=json.loads((BASE/'feature_columns.json').read_text());nf=len(names);assert nf==15
assert torch.cuda.is_available()
rows=[];daily=[];durations=[]
for year in [2025,2026]:
    frame=pd.read_parquet(BASE/f'predict_table_2022-2026/predict_table_{year}.parquet').sort_values(['entry_time_utc','symbol'])
    n=frame.symbol.nunique();d=frame.entry_time_utc.nunique();assert len(frame)==n*d
    torch.cuda.synchronize();begin=time.perf_counter()
    x=torch.tensor(frame[names].to_numpy().reshape(d,n,nf),device='cuda',dtype=torch.float64)
    y=torch.tensor(frame.excess_return_24h.to_numpy().reshape(d,n),device='cuda',dtype=torch.float64)
    known=torch.tensor(frame.pool_known_at_decision.to_numpy().reshape(d,n),device='cuda')
    mask=torch.isfinite(x).all(dim=2)&known
    members=mask.sum(dim=1)
    # Establish factor ordering before consulting outcome availability.
    scores=torch.where(mask[:,:,None],x,torch.inf)
    ordering=torch.argsort(scores,dim=1,stable=True)
    evaluate=(members>=20)&((~mask)|torch.isfinite(y)).all(dim=1)
    clean_y=torch.nan_to_num(y,nan=0.,posinf=0.,neginf=0.)
    sorted_y=clean_y[:,:,None].expand(-1,-1,nf).gather(1,ordering)
    prefix=torch.cat([torch.zeros((d,1,nf),dtype=torch.float64,device='cuda'),sorted_y.cumsum(dim=1)],dim=1)
    q=members//10;rem=members%10;parts=[]
    for g in range(10):
        start=g*q+torch.minimum(rem,torch.full_like(rem,g))
        count=q+(rem>g).to(torch.int64);end=start+count
        lo=prefix.gather(1,start[:,None,None].expand(-1,1,nf)).squeeze(1)
        hi=prefix.gather(1,end[:,None,None].expand(-1,1,nf)).squeeze(1)
        parts.append(((hi-lo)/count[:,None])[evaluate])
    groups=torch.stack(parts,dim=1)
    totals=groups.sum(dim=0)
    spread=totals[9]-totals[0]
    torch.cuda.synchronize();durations.append(time.perf_counter()-begin)
    totals_cpu=totals.cpu().numpy();ls=spread.cpu().numpy();group_cpu=groups.cpu().numpy()
    dates=frame.drop_duplicates('entry_time_utc').decision_date_beijing.to_numpy()[evaluate.cpu().numpy()]
    for j,name in enumerate(names):
        rows.append({'factor':name,'year':year,'G1_excess_simple':float(totals_cpu[0,j]),
                     'G10_excess_simple':float(totals_cpu[9,j]),'long_short_G10_minus_G1':float(ls[j]),'days':len(dates)})
        for k,day in enumerate(dates):daily.append({'factor':name,'year':year,'date_beijing':day,
             'G1_excess':float(group_cpu[k,0,j]),'G10_excess':float(group_cpu[k,9,j]),
             'long_short_return':float(group_cpu[k,9,j]-group_cpu[k,0,j])})
    print('YEAR',year,'DAYS',len(dates),'GPU_SECONDS',durations[-1],flush=True)
result=pd.DataFrame(rows);result.to_csv(OUT/'annual_longshort_with_legs.csv',index=False)
pivot=result.pivot(index='factor',columns='year',values='long_short_G10_minus_G1').reindex(names)
pivot.to_csv(OUT/'annual_longshort.csv')
pd.DataFrame(daily).to_csv(OUT/'daily_longshort.csv',index=False)
meta={'device':'cuda','GPU':torch.cuda.get_device_name(),'gpu_compute_seconds':sum(durations),
      'sample':'same all15-feature-complete, known-pool and realized-label-complete scoring sample as current XGBoost runs',
      'factor_direction':'original percentile ascending; no sign reversal; stable symbol ties',
      'portfolio':'long G10 at100% notional, short G1 at100%; total gross200%; 1x total gross equal split halves these returns',
      'return':'sum of daily G10 excess minus G1 excess, equivalent to raw group return difference; gross before fees/funding/slippage',
      '2026':'through last scored date 2026-11-04; partial year',
      'ranking':'CUDA stable argsort; equal-count numpy.array_split-equivalent sizes calculated on CUDA',
      'limitation':'retrospective selected factors; not an independent out-of-sample test'}
(OUT/'method.json').write_text(json.dumps(meta,indent=2))
show=pivot.map(lambda v:f'{v:.2%}')
doc='<meta charset="utf-8"><title>15因子单因子多空收益</title><style>body{font:15px system-ui;max-width:1200px;margin:30px auto}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}p{line-height:1.8}</style><h1>15因子 · 2025 / 2026 单因子多空收益</h1><p>原始百分位方向，G10−G1，不根据年度表现反转。与15因子XGBoost回测相同样本。累计单利毛收益、总名义敞口200%，不扣手续费、资金费和滑点。2026截至11月4日。</p>'+show.to_html()+'<h2>G1/G10两端</h2>'+result.to_html(index=False,float_format=lambda v:f'{v:.5f}')
(OUT/'15因子单因子多空收益_2025-2026.html').write_text(doc,encoding='utf-8')
print(pivot.to_string(),flush=True)
