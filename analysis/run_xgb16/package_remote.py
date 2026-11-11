from pathlib import Path
import tarfile
r=Path('/workspace/xgb16run')
with tarfile.open(r/'results.tar.gz','w:gz') as z:
 for name in ['xgb16_gpu_expanding_120_depth4_lr005_2022_2026','xgb16_120_lr005_net_compound_fee0005','xgb16_120_lr005_hysteresis_10_15']:
  z.add(Path('/tmp')/name,arcname=name)
 for name in ['table_coverage.csv','funding_method.json','funding_acquisition.json']:
  z.add(r/name,arcname='xgb16_preparation/'+name)
print((r/'results.tar.gz').stat().st_size)
