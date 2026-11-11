from pathlib import Path
import json,pyarrow.parquet as pq
root=Path('/workspace/xgb14run');src=Path('/workspace/xgb16run/data/model_tables/factors15_2021_2026');dst=root/'data/model_tables/factors14_2021_2026';dst.mkdir(parents=True,exist_ok=True)
names=json.loads((src/'feature_columns.json').read_text());names.remove('range_to_quote_volume_2');assert len(names)==14
for p in src.rglob('*.parquet'):
 assert set(names)<=set(pq.read_schema(p).names)
 target=dst/p.relative_to(src);target.parent.mkdir(exist_ok=True)
 if not target.exists():target.symlink_to(p)
(dst/'feature_columns.json').write_text(json.dumps(names,indent=2));(root/'setup.json').write_text(json.dumps(dict(features=names,source=str(src),change='Original15 minus range_to_quote_volume_2; no funding features or funding availability filters',trees=120,depth=4,eta=.05,training='Expanding2021 onwards'),indent=2));print('Original14 tables ready',flush=True)
