from pathlib import Path
import pandas as pd,tarfile
rows=[]
for trees in [300,500]:
 n=Path(f'/tmp/xgb57alpha_native_{trees}_lr005_short100_hysteresis_10_30');m=Path(f'/tmp/xgb57alpha_native_gpu_expanding_{trees}_depth4_lr005_2022_2026')
 a=pd.read_csv(n/'annual_net_metrics.csv');b=pd.read_csv(m/'annual_metrics.csv');a=a.merge(b[['prediction_year','RankIC','RankICIR']],left_on='year',right_on='prediction_year');a['trees']=trees;rows.append(a)
 d=pd.read_csv(n/'daily_net_equity.csv');print('TOTAL',trees,d.net_equity.iloc[-1]-1)
s=pd.concat(rows,ignore_index=True);out=Path('/tmp/xgb57_short_300_500_comparison');out.mkdir(exist_ok=True);s.to_csv(out/'annual_comparison.csv',index=False);print(s.to_string(index=False))
with tarfile.open('/tmp/xgb57_500_results.tar.gz','w:gz',compresslevel=3) as z:
 for name in ['xgb57alpha_native_gpu_expanding_500_depth4_lr005_2022_2026','xgb57alpha_native_500_lr005_short100_hysteresis_10_30','xgb57_short_300_500_comparison']:
  z.add('/tmp/'+name,arcname=name)
