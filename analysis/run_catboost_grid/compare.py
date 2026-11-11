from pathlib import Path
import pandas as pd,tarfile,json
O=Path('/tmp/catboost_grid_results');O.mkdir(exist_ok=True);rows=[];totals=[]
for count in [15,42]:
 for trees in [120,400]:
  tag=f'catboost{count}_{trees}';m=Path(f'/tmp/{tag}_gpu_expanding_depth4_lr004_2022_2026');n=Path(f'/tmp/{tag}_lr004_hysteresis_10_30')
  a=pd.read_csv(n/'annual_net_metrics.csv');b=pd.read_csv(m/'annual_metrics.csv');a=a.merge(b[['prediction_year','RankIC','RankICIR']],left_on='year',right_on='prediction_year');a['factor_count']=count;a['trees']=trees;rows.append(a)
  d=pd.read_csv(n/'daily_net_equity.csv');totals.append(dict(factor_count=count,trees=trees,total_net_return=d.net_equity.iloc[-1]-1))
  cfg=json.loads((m/'manifest.json').read_text());assert cfg['params']['task_type']=='GPU' and cfg['params']['iterations']==trees and len(cfg['features'])==count
s=pd.concat(rows,ignore_index=True);s.to_csv(O/'annual_comparison.csv',index=False);t=pd.DataFrame(totals);t.to_csv(O/'totals.csv',index=False)
print(s.to_string(index=False));print(t.to_string(index=False))
with tarfile.open('/tmp/catboost_grid_results.tar.gz','w:gz') as z:
 z.add(O,arcname=O.name)
 for count in [15,42]:
  for trees in [120,400]:
   tag=f'catboost{count}_{trees}';m=Path(f'/tmp/{tag}_gpu_expanding_depth4_lr004_2022_2026');n=Path(f'/tmp/{tag}_lr004_hysteresis_10_30')
   for file in ['manifest.json','annual_metrics.csv','feature_importance.csv']:z.add(m/file,arcname=m.name+'/'+file)
   for year in range(2022,2027):
    for file in ['model.cbm','booster_config.json','run.json','training_rmse.json']:z.add(m/str(year)/file,arcname=f'{m.name}/{year}/{file}')
   z.add(n,arcname=n.name)
