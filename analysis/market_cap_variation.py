"""Descriptive market-cap dispersion and calendar-day rank stability."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/market_cap_variation'
OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_parquet(ROOT/'data/fundamentals/cmc_2021_2025/contract_daily.parquet')
df['date']=pd.to_datetime(df.date)
df.loc[(df.marketcap_identity_error > .01) | (df.market_cap_usd <= 0),'market_cap_usd']=np.nan
rows=[]
for year in range(2021,2026):
    part=df.loc[df.in_research_pool & (df.date.dt.year==year)]
    cap=part.pivot(index='date',columns='contract',values='market_cap_usd').sort_index()
    ids=part.pivot(index='date',columns='contract',values='cmc_id').reindex_like(cap)
    q=cap.quantile([.1,.5,.9],axis=1)
    row={'data_year':year,'pool_year':year+1,
         'median_daily_members':float(cap.notna().sum(axis=1).median()),
         'median_daily_p90_p10_ratio':float((q.loc[.9]/q.loc[.1]).median()),
         'median_daily_max_min_ratio':float((cap.max(axis=1)/cap.min(axis=1)).median()),
         'median_daily_max_median_ratio':float((cap.max(axis=1)/q.loc[.5]).median())}
    for lag in [1,7,30]:
        past=cap.shift(lag)
        stable_id=ids.eq(ids.shift(lag))
        pct=(cap/past-1).where(stable_id)
        a=pct.to_numpy(); a=a[np.isfinite(a)]
        row[f'median_absolute_{lag}d_cap_change']=float(np.median(np.abs(a)))
        row[f'p90_absolute_{lag}d_cap_change']=float(np.quantile(np.abs(a),.9))
        correlations, switches, rank_moves, overlaps=[] ,[],[],[]
        for day in cap.index[lag:]:
            current=cap.loc[day]
            previous=past.loc[day]
            good=current.notna() & previous.notna() & stable_id.loc[day]
            if good.sum()<20:
                continue
            x=current[good].rank(method='average')
            y=previous[good].rank(method='average')
            correlations.append(float(x.corr(y)))
            # Exact equal-count ten-way allocation, ascending symbol tie break.
            def groups(values):
                result=pd.Series(index=values.index,dtype='int64')
                ordered=values.sort_values(kind='stable').index.to_numpy()
                for g,names in enumerate(np.array_split(ordered,10)):
                    result.loc[names]=g
                return result
            switches.append(float(groups(current[good]).ne(groups(previous[good])).mean()))
            rank_moves.append(float((x-y).abs().div(len(x)).median()))
            overlaps.append(int(good.sum()))
        row[f'median_{lag}d_rank_correlation']=float(np.median(correlations))
        row[f'median_{lag}d_decile_switch_fraction']=float(np.median(switches))
        row[f'median_{lag}d_rank_percentile_move']=float(np.median(rank_moves))
        row[f'median_{lag}d_common_members']=float(np.median(overlaps))
    rows.append(row)
result=pd.DataFrame(rows)
result.to_csv(OUT/'summary.csv',index=False)
(OUT/'method.json').write_text(json.dumps({
    'source':'local CMC daily historical market caps; no new downloads',
    'scope':'data year y uses already-selected pool y+1, retrospective fixed membership',
    'dispersion':'median across daily cross-sections; P90/P10 uses pandas linear quantiles',
    'change':'absolute market-cap percentage change, pooled asset-day observations by year',
    'stability':'daily cross-sectional rank correlations and decile switch fractions, median across dates',
    'pairing':'exact 1/7/30 calendar days; require positive cap both dates, unchanged provider asset id; minimum 20 common assets',
    'missing':'no fill; provider market-cap identity error >1% excluded; index contracts excluded by missing caps',
    'deciles':'re-sort both dates on the identical common-asset subset; equal-count numpy.array_split, ascending symbols as tie break',
    'caveats':'revised historical observations; rank similarity is descriptive and does not establish profitability; end-year pools are retrospective'
},indent=2))
print(result.to_json(orient='records',indent=2))
