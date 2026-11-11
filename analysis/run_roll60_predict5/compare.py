from pathlib import Path
import pandas as pd
O=Path('/tmp/roll60_predict5_results');O.mkdir(exist_ok=True)
rows=[];totals=[]
for count in [15,19]:
 m=Path(f'/tmp/xgb{count}_roll60_predict5_100_lr005');n=Path(str(m)+'_hysteresis_10_30')
 a=pd.read_csv(n/'annual_net_metrics.csv');b=pd.read_csv(m/'annual_metrics.csv')
 a=a.merge(b[['prediction_year','RankIC','RankICIR']],left_on='year',right_on='prediction_year');a['factor_count']=count;rows.append(a)
 d=pd.read_csv(n/'daily_net_equity.csv');totals.append(dict(factor_count=count,total_net_return=d.net_equity.iloc[-1]-1))
pd.concat(rows,ignore_index=True).to_csv(O/'annual_comparison.csv',index=False)
pd.DataFrame(totals).to_csv(O/'totals.csv',index=False)
print(pd.concat(rows).to_string(index=False),flush=True);print(pd.DataFrame(totals).to_string(index=False),flush=True)
