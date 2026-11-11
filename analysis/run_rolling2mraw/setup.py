from pathlib import Path
import pandas as pd,numpy as np,json
src=Path('/workspace/xgb16run/data/model_tables/factors15_2021_2026');root=Path('/workspace/rolling2mraw');dst=root/'data/model_tables/factors15raw_2021_2026';dst.mkdir(parents=True,exist_ok=True)
names=json.loads((src/'feature_columns.json').read_text());raw=pd.read_parquet('/workspace/raw15.parquet');keys=['symbol','entry_time_utc'];audit=[]
for p in src.rglob('*.parquet'):
 old=pd.read_parquet(p);f=old.drop(columns=names).merge(raw,on=keys,how='left',validate='one_to_one');assert len(f)==len(old)
 assert f[keys].equals(old[keys].reset_index(drop=True))
 a=np.isfinite(old[names]).to_numpy();b=np.isfinite(f[names]).to_numpy();assert np.array_equal(a,b),(p.name,int((a!=b).sum()))
 assert f.excess_return_24h.equals(old.excess_return_24h.reset_index(drop=True))
 f['features_complete']=b.all(axis=1);f['prediction_eligible_on_features']=f.features_complete&f.pool_known_at_decision
 target=dst/p.relative_to(src);target.parent.mkdir(exist_ok=True);f.to_parquet(target,index=False)
 audit.append(dict(table=p.name,rows=len(f),complete=int(b.all(axis=1).sum()),same_eligibility_and_labels=True))
(dst/'feature_columns.json').write_text(json.dumps(names,indent=2));(root/'raw_input_audit.json').write_text(json.dumps(audit,indent=2));print('RAW TABLES READY same sample masks and excess labels',flush=True)
