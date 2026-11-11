from pathlib import Path
import json,pyarrow.parquet as pq
src=Path('/workspace/xgb16run/data/model_tables/factors16_2021_2026');dst=Path('/workspace/xgb17run/data/model_tables/factors17_2021_2026');dst.mkdir(parents=True,exist_ok=True)
names=json.loads(Path('/workspace/xgb16run/data/model_tables/factors15_2021_2026/feature_columns.json').read_text())+['funding_sum_3d','funding_sum_7d']
assert len(names)==17
for p in src.rglob('*.parquet'):
 assert set(names)<=set(pq.read_schema(p).names)
 target=dst/p.relative_to(src);target.parent.mkdir(exist_ok=True)
 if not target.exists():target.symlink_to(p)
(dst/'feature_columns.json').write_text(json.dumps(names,indent=2))
Path('/workspace/xgb17run/setup.json').write_text(json.dumps(dict(features=names,data_source=str(src),change='Restore range_to_quote_volume_2; reuse all16-run table values and funding percentiles unchanged',trees=120,depth=4,eta=.05,training='Expanding2021 onwards'),indent=2))
print('17-feature tables ready')
