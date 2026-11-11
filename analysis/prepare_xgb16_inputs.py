from pathlib import Path
import pandas as pd,numpy as np,json,tarfile
base=Path('data/model_tables/factors15_2021_2026');out=Path('reports/xgb16_preparation');out.mkdir(exist_ok=True)
features=json.loads(Path('analysis/active_feature_columns.json').read_text());features=[f for f in features if not f.startswith('funding_')]
parts=[]
for p in list(base.glob('train_table_2021-2025/train_table_*.parquet'))+list(base.glob('predict_table_2022-2026/predict_table_*.parquet')):
 f=pd.read_parquet(p);m=np.isfinite(f[features]).all(axis=1)
 parts.append(f.loc[m,['symbol','entry_time_utc']])
need=pd.concat(parts).drop_duplicates().sort_values(['symbol','entry_time_utc']);need=need.loc[need.entry_time_utc.lt(pd.Timestamp('2026-10-28',tz='Asia/Shanghai').tz_convert('UTC'))]
seed=pd.concat([pd.read_parquet(p,columns=['symbol','entry_time_utc','funding_sum_3d','funding_sum_7d']) for p in Path('reports/funding_four_factors').glob('funding_table_202[2-6].parquet')]).drop_duplicates(['symbol','entry_time_utc'])
need.to_parquet(out/'needed.parquet',index=False);seed.to_parquet(out/'seed.parquet',index=False)
with tarfile.open(out/'inputs.tar.gz','w:gz') as t:
 for p in list(base.glob('train_table_2021-2025/train_table_*.parquet'))+list(base.glob('predict_table_2022-2026/predict_table_*.parquet')):t.add(p,arcname='data/model_tables/factors15_2021_2026/'+p.relative_to(base).as_posix())
 t.add(base/'feature_columns.json',arcname='data/model_tables/factors15_2021_2026/feature_columns.json')
 t.add(out/'needed.parquet',arcname='needed.parquet');t.add(out/'seed.parquet',arcname='seed.parquet')
print('Required symbol-days',len(need),'seed',len(seed),'archive bytes',(out/'inputs.tar.gz').stat().st_size)
