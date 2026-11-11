from pathlib import Path
import json,importlib.util,pandas as pd,numpy as np
R=Path(__file__).resolve().parents[2];B=R/'data/model_tables/factors15_2021_2026';O=R/'reports/rolling2m_raw_preparation';O.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('builder',R/'analysis/build_train_predict_tables.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
names=json.loads((B/'feature_columns.json').read_text());keys=['symbol','decision_timestamp']
a=pd.read_parquet(R/'data/model_tables/factors10_2021_2026/six_price_factors_raw.parquet');c=pd.read_parquet(B/'five_new_factors_raw.parquet');raw=a.merge(c,on=keys,validate='one_to_one')
parts=[]
for p in list(B.glob('train_table_2021-2025/*.parquet'))+list(B.glob('predict_table_2022-2026/*.parquet')):
 f=pd.read_parquet(p,columns=['symbol','entry_time_utc','market_cap_source_last_date_utc']);parts.append(f)
q=pd.concat(parts).drop_duplicates(['symbol','entry_time_utc']);q['decision_timestamp']=q.entry_time_utc.dt.as_unit('ms').astype('int64');q=q.merge(raw,on=keys,how='left',validate='one_to_one')
for name,panel in b.cap_features().items():
 ri=panel.index.get_indexer(q.market_cap_source_last_date_utc);ci=panel.columns.get_indexer(q.symbol);v=np.full(len(q),np.nan);good=(ri>=0)&(ci>=0);v[good]=panel.to_numpy()[ri[good],ci[good]];q[name]=v
q[['symbol','entry_time_utc',*names]].to_parquet(O/'raw15.parquet',index=False)
print('Raw15 rows',len(q),'bytes',(O/'raw15.parquet').stat().st_size)
