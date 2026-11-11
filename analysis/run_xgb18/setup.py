from pathlib import Path
import json,pandas as pd
root=Path('/workspace/xgb18run');src=Path('/workspace/xgb17deltarun/data/model_tables/factors17delta_2021_2026');add=Path('/workspace/xgb16run/data/model_tables/factors16_2021_2026');dst=root/'data/model_tables/factors18_2021_2026';dst.mkdir(parents=True,exist_ok=True)
names=json.loads((src/'feature_columns.json').read_text())+['funding_sum_3d'];assert len(set(names))==18
for p in src.rglob('*.parquet'):
 f=pd.read_parquet(p);g=pd.read_parquet(add/p.relative_to(src),columns=['symbol','entry_time_utc','funding_sum_3d']);n=len(f)
 f=f.merge(g,on=['symbol','entry_time_utc'],how='left',validate='one_to_one');assert len(f)==n
 target=dst/p.relative_to(src);target.parent.mkdir(exist_ok=True);f.to_parquet(target,index=False)
(dst/'feature_columns.json').write_text(json.dumps(names,indent=2));(root/'setup.json').write_text(json.dumps(dict(features=names,change='Add existing3day cumulative funding percentiles to previous17delta tables. All other inputs and training parameters unchanged.',source=str(src),added_source=str(add)),indent=2));print('18-factor tables ready',flush=True)
