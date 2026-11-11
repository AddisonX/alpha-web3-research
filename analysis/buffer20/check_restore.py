import pandas as pd,numpy as np,json
from pathlib import Path
r=Path('/workspace/buffer20');a=pd.read_csv(r/'original15_metrics.csv').set_index('prediction_year');b=pd.read_csv('/tmp/xgb15restore_gpu_expanding_120_depth4_lr005_2022_2026/annual_metrics.csv').set_index('prediction_year')
cols=['RankIC','RankICIR']+[f'G{i}_excess_simple' for i in range(1,11)]
diff=float(np.max(np.abs(a[cols].to_numpy()-b[cols].to_numpy())))
(r/'restore_check.json').write_text(json.dumps({'maximum_absolute_metric_difference':diff,'matches':bool(np.allclose(a[cols],b[cols],rtol=1e-8,atol=1e-10))},indent=2))
assert np.allclose(a[cols],b[cols],rtol=1e-8,atol=1e-10),diff
print('Original15 restored metrics match',diff,flush=True)
