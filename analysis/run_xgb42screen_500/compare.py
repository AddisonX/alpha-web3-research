from pathlib import Path
import pandas as pd,tarfile
rows=[];totals=[]
for tag,model,net in [('57_lr005','xgb57alpha_native_gpu_expanding_500_depth4_lr005_2022_2026','xgb57alpha_native_500_lr005_hysteresis_10_30'),('42_lr004','xgb42screen_native_gpu_expanding_500_depth4_lr004_2022_2026','xgb42screen_native_500_lr004_hysteresis_10_30')]:
 m=Path('/tmp')/model;n=Path('/tmp')/net
 a=pd.read_csv(n/'annual_net_metrics.csv');b=pd.read_csv(m/'annual_metrics.csv');a=a.merge(b[['prediction_year','RankIC','RankICIR']],left_on='year',right_on='prediction_year');a['case']=tag;rows.append(a)
 d=pd.read_csv(n/'daily_net_equity.csv');totals.append(dict(case=tag,total=d.net_equity.iloc[-1]-1))
o=Path('/tmp/xgb42screen_vs57');o.mkdir(exist_ok=True);pd.concat(rows).to_csv(o/'annual_comparison.csv',index=False);pd.DataFrame(totals).to_csv(o/'totals.csv',index=False)
print(pd.concat(rows).to_string(index=False));print(pd.DataFrame(totals).to_string(index=False))
with tarfile.open('/tmp/xgb42screen_metrics.tar.gz','w:gz') as z:
 z.add(o,arcname=o.name)
 m=Path('/tmp/xgb42screen_native_gpu_expanding_500_depth4_lr004_2022_2026')
 for name in ['manifest.json','annual_metrics.csv','feature_importance_gain.csv']:
  z.add(m/name,arcname=m.name+'/'+name)
 n=Path('/tmp/xgb42screen_native_500_lr004_hysteresis_10_30');z.add(n,arcname=n.name)
