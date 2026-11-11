"""Independent reconciliation against the previously published factor-decile curves."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/model_tables/factors10_2021_2026'
features=pd.read_parquet(OUT/'six_price_factors_raw.parquet')
page=(ROOT/'reports/达标60因子_2021-2025整体汇总.html').read_text(encoding='utf-8')
i=page.index('const factors=')+len('const factors=');curves,_=json.JSONDecoder().raw_decode(page[i:])
original={}
for f in curves:
    if f['name'] not in features.columns:continue
    c=pd.DataFrame(np.array(f['curves'],dtype=float).T,index=pd.to_datetime(f['dates']),columns=range(1,11))
    r=c.ffill().diff().where(c.notna());r.index-=pd.Timedelta(days=1)
    original[f['name']]=r.dropna().loc['2021-01-01':'2025-12-31']
records=[]
for year in range(2021,2026):
    data=pd.read_parquet(ROOT/f'data/returns/annual_pools_2021_to_t/pool_{year+1}/daily_returns.parquet')
    data=data.loc[data.data_year.eq(year)].copy()
    data['decision_timestamp']=data.entry_time_utc.dt.as_unit('ms').astype('int64')
    data=data.merge(features,on=['symbol','decision_timestamp'],how='left',validate='one_to_one').sort_values(['entry_time_utc','symbol'])
    errors={name:[] for name in original}
    for stamp,frame in data.groupby('entry_time_utc'):
        day=stamp.tz_convert('Asia/Shanghai').tz_localize(None)
        for name,old in original.items():
            if day not in old.index:continue
            assert frame[name].notna().all(),(year,day,name,'missing original-valid factor')
            assert frame.return_24h.notna().all()
            x=frame[name].to_numpy();returns=frame.return_24h.to_numpy();bench=returns.mean()
            groups=np.array_split(np.argsort(x,kind='stable'),10)
            calculated=np.array([returns[g].mean()-bench for g in groups])
            err=np.max(np.abs(calculated-old.loc[day].to_numpy()))
            errors[name].append(float(err))
    for name,values in errors.items():
        records.append({'year':year,'factor':name,'comparison_days':len(values),
                        'maximum_decile_daily_difference':max(values) if values else None,
                        'days_difference_over_1e_10':sum(v>1e-10 for v in values)})
result={'checks':records,'comparison':'new six raw factors sorted into original ten equal-count deciles; same original valid days and full-pool benchmark',
        'max_absolute_difference':max(r['maximum_decile_daily_difference'] for r in records if r['comparison_days'])}
(OUT/'factor_reconciliation.json').write_text(json.dumps(result,indent=2))
print(pd.DataFrame(records).to_string(index=False))
print('max_difference',result['max_absolute_difference'])
