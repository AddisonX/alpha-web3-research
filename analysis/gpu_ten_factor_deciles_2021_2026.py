import sys
sys.path.append('/workspace/alpha-web3-research/.venv/lib/python3.12/site-packages')
from pathlib import Path
import pandas as pd,numpy as np,torch,json
BASE=Path('/tmp/alpha15-run/data/model_tables/factors15_2021_2026')
OUT=Path('/tmp/ten_factor_deciles_2021_2026');OUT.mkdir(exist_ok=True)
names=json.loads((BASE/'feature_columns.json').read_text())[:10]
rows=[]
assert torch.cuda.is_available()
for year in range(2021,2027):
    file=BASE/('train_table_2021-2025/train_table_2021.parquet' if year==2021 else f'predict_table_2022-2026/predict_table_{year}.parquet')
    df=pd.read_parquet(file);df=df.loc[df.data_year.eq(year)]
    if year==2026:df=df.loc[df.decision_date_beijing.le('2026-10-27')]
    df=df.sort_values(['entry_time_utc','symbol'])
    n=df.symbol.nunique();d=df.entry_time_utc.nunique();assert len(df)==n*d
    x=torch.tensor(df[names].to_numpy().reshape(d,n,10),device='cuda',dtype=torch.float64)
    y=torch.tensor(df.realized_return_24h.to_numpy().reshape(d,n),device='cuda',dtype=torch.float64)
    bench=torch.tensor(df.benchmark_return_24h.to_numpy().reshape(d,n)[:,0],device='cuda',dtype=torch.float64)
    known=torch.ones((d,n),device='cuda',dtype=torch.bool) if year==2021 else torch.tensor(df.pool_known_at_decision.to_numpy().reshape(d,n),device='cuda')
    mask=torch.isfinite(x)&known[:,:,None]
    count=mask.sum(1)
    order=torch.argsort(torch.where(mask,x,torch.inf),dim=1,stable=True)
    valid=(count>=10)&((~mask)|torch.isfinite(y)[:,:,None]).all(1)&torch.isfinite(bench[:,None])
    sorted_y=torch.nan_to_num(y,nan=0)[:,:,None].expand(-1,-1,10).gather(1,order)
    prefix=torch.cat([torch.zeros((d,1,10),device='cuda',dtype=torch.float64),sorted_y.cumsum(1)],1)
    q=count//10;rem=count%10
    dates=df.drop_duplicates('entry_time_utc').decision_date_beijing.to_numpy()
    for group in range(10):
        start=group*q+torch.minimum(rem,torch.full_like(rem,group));size=q+(rem>group)
        gr=(prefix.gather(1,(start+size)[:,None,:]).squeeze(1)-prefix.gather(1,start[:,None,:]).squeeze(1))/size.clamp(min=1)
        excess=gr-bench[:,None]
        for j,name in enumerate(names):
            ok=valid[:,j].cpu().numpy()
            rows.append(pd.DataFrame(dict(year=year,pool_year=2022 if year==2021 else year,factor=name,decile=group+1,
                date_beijing=dates[ok],group_return=gr[:,j].cpu().numpy()[ok],excess_return=excess[:,j].cpu().numpy()[ok],
                members=size[:,j].cpu().numpy()[ok])))
    print('Year complete',year,flush=True)
pd.concat(rows,ignore_index=True).to_parquet(OUT/'decile_daily.parquet',index=False)
(OUT/'method.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),factors=names,
    returns='Gross cumulative excess simple return relative to supplied full-pool equalweight benchmark. No fees, funding, slippage; original diagnostic chart convention.',
    cohorts='2021 uses2022 pool (retrospective first-year training data); 2022-2026 use same-year predict pools. 2021 cohort is not known point-in-time.',
    grouping='Each factor independently finite; stable symbol-order ties; original numpy.array_split-equivalent10 groups; G1 lowest G10 highest.',
    missing='Complete outcomes required for all eligible members on each factor/day; omitted days not imputed.',
    horizon='2021-01-01 to2026-10-27 decisions, 2026 return endpointOct28'),indent=2))
