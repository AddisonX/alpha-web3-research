from pathlib import Path
import pandas as pd,json
O=Path('/tmp/yearmask250_results');O.mkdir(exist_ok=True)
rows=[];totals=[];audits=[]
for count in [19,15]:
 m=Path(f'/tmp/xgb{count}_yearmask_gpu_expanding_250_depth4_lr005_2022_2026');n=Path(f'/tmp/xgb{count}_yearmask_250_lr005_hysteresis_10_30')
 a=pd.read_csv(n/'annual_net_metrics.csv');b=pd.read_csv(m/'annual_metrics.csv')
 a=a.merge(b[['prediction_year','RankIC','RankICIR']],left_on='year',right_on='prediction_year');a['factor_count']=count;rows.append(a)
 d=pd.read_csv(n/'daily_net_equity.csv');totals.append(dict(factor_count=count,total_net_return=d.net_equity.iloc[-1]-1))
 for y in range(2022,2027):
  z=pd.read_csv(m/str(y)/'year_factor_mask_audit.csv');z['factor_count']=count;audits.append(z)
pd.concat(rows,ignore_index=True).to_csv(O/'annual_comparison.csv',index=False)
pd.concat(audits,ignore_index=True).to_csv(O/'all_mask_audits.csv',index=False)
pd.DataFrame(totals).to_csv(O/'totals.csv',index=False)
print(pd.concat(rows).to_string(index=False),flush=True);print(pd.DataFrame(totals).to_string(index=False),flush=True)
