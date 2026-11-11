from pathlib import Path
import tarfile
with tarfile.open('/tmp/xgb57_results.tar.gz','w:gz',compresslevel=3) as z:
 for name in ['xgb57alpha_native_gpu_expanding_300_depth4_lr005_2022_2026','xgb57alpha_native_300_lr005_hysteresis_10_30']:
  z.add('/tmp/'+name,arcname='reports/'+name)
 z.add('/tmp/xgb22run/data/model_tables/factors57alpha_2021_2026',arcname='data/model_tables/factors57alpha_2021_2026')
 for name in ['xgb57_prepare.log','xgb57_native_train.log','xgb57_native_buffer.log']:
  z.add('/tmp/'+name,arcname='reports/xgb57_logs/'+name)
print('Archive ready',flush=True)
