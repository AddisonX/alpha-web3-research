from pathlib import Path
import tarfile
m=Path('/tmp/catboost15_120_gpu_expanding_depth4_lr005_2022_2026');n=Path('/tmp/catboost15_120_lr005_hysteresis_10_30')
print((n/'annual_net_metrics.csv').read_text())
with tarfile.open('/tmp/catboost15_120_lr005_results.tar.gz','w:gz') as z:
 z.add(n,arcname=n.name)
 for file in ['manifest.json','annual_metrics.csv','feature_importance.csv']:z.add(m/file,arcname=m.name+'/'+file)
 for year in range(2022,2027):
  for file in ['model.cbm','booster_config.json','run.json','training_rmse.json']:z.add(m/str(year)/file,arcname=f'{m.name}/{year}/{file}')
