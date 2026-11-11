from pathlib import Path
import pandas as pd,tarfile
rows=[];totals=[]
for trees in [500,200]:
 m=Path(f'/tmp/xgb42screen_native_gpu_expanding_{trees}_depth4_lr004_2022_2026');n=Path(f'/tmp/xgb42screen_native_{trees}_lr004_hysteresis_10_30')
 a=pd.read_csv(n/'annual_net_metrics.csv');b=pd.read_csv(m/'annual_metrics.csv');a=a.merge(b[['prediction_year','RankIC','RankICIR']],left_on='year',right_on='prediction_year');a['trees']=trees;rows.append(a)
 d=pd.read_csv(n/'daily_net_equity.csv');totals.append(dict(trees=trees,total=d.net_equity.iloc[-1]-1))
o=Path('/tmp/xgb42screen_200_vs500');o.mkdir(exist_ok=True);pd.concat(rows).to_csv(o/'annual_comparison.csv',index=False);pd.DataFrame(totals).to_csv(o/'totals.csv',index=False)
print(pd.concat(rows).to_string(index=False));print(pd.DataFrame(totals).to_string(index=False))
with tarfile.open('/tmp/xgb42screen200_metrics.tar.gz','w:gz') as z:
 z.add(o,arcname=o.name)
 m=Path('/tmp/xgb42screen_native_gpu_expanding_200_depth4_lr004_2022_2026')
 for name in ['manifest.json','annual_metrics.csv','feature_importance_gain.csv']:z.add(m/name,arcname=m.name+'/'+name)
 n=Path('/tmp/xgb42screen_native_200_lr004_hysteresis_10_30');z.add(n,arcname=n.name)
