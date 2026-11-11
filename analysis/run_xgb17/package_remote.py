from pathlib import Path
import tarfile
r=Path('/workspace/xgb17run')
with tarfile.open(r/'results.tar.gz','w:gz') as z:
 for name in ['xgb17_gpu_expanding_120_depth4_lr005_2022_2026','xgb17_120_lr005_net_compound_fee0005','xgb17_120_lr005_hysteresis_10_15']:
  z.add(Path('/tmp')/name,arcname=name)
 for name in ['setup.json']:
  z.add(r/name,arcname='xgb17_preparation/'+name)
print((r/'results.tar.gz').stat().st_size)
