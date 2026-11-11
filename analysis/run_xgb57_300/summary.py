from pathlib import Path
import pandas as pd,json
m=Path('/tmp/xgb57alpha_gpu_expanding_300_depth4_lr005_2022_2026');n=Path('/tmp/xgb57alpha_300_lr005_hysteresis_10_30')
a=pd.read_csv(n/'annual_net_metrics.csv');b=pd.read_csv(m/'annual_metrics.csv')
c=a.merge(b[['prediction_year','RankIC','RankICIR']],left_on='year',right_on='prediction_year')
c.to_csv(n/'annual_summary.csv',index=False)
d=pd.read_csv(n/'daily_net_equity.csv');print(c.to_string(index=False));print('TOTAL',d.net_equity.iloc[-1]-1)
