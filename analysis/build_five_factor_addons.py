"""Add five original-formula percentile factors to existing tables; no changes to labels."""
import importlib.util,json,zipfile,hashlib
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('table_builder',ROOT/'analysis/build_train_predict_tables.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
ADDED=['avg_range_4','distance_to_low_6','avg_range_3','distance_to_low_4','volatility_4']
OUT=ROOT/'data/model_tables/factors15_2021_2026'
OUT.mkdir(parents=True,exist_ok=True)
b.PRICE_FEATURES=ADDED
b.FEATURE_ROOT=OUT/'raw_new_factors_by_symbol';b.FEATURE_ROOT.mkdir(exist_ok=True)
keys=b.sources();frames=[]
with ThreadPoolExecutor(max_workers=6) as ex:
    fs={ex.submit(b.symbol_features,n,k):n for n,k in sorted(keys.items())}
    for i,f in enumerate(as_completed(fs),1):
        frames.append(f.result())
        if i%60==0 or i==len(fs):print(f'New raw features {i}/{len(fs)}',flush=True)
raw=pd.concat(frames,ignore_index=True)[['symbol','decision_timestamp',*ADDED]]
raw.to_parquet(OUT/'five_new_factors_raw.parquet',index=False)
original=ROOT/'data/model_tables/factors10_2021_2026'
features=json.loads((original/'feature_columns.json').read_text())+ADDED
assert len(features)==15
(OUT/'feature_columns.json').write_text(json.dumps(features,indent=2))
summary=[]
# Each cohort's cross-sectional denominator is independent of future outcomes.
for split,years in [('train',range(2021,2026)),('predict',range(2022,2027))]:
    dirname=f'{split}_table_'+('2021-2025' if split=='train' else '2022-2026')
    destination=OUT/dirname;destination.mkdir(exist_ok=True)
    for year in years:
        name=f'{split}_table_{year}'
        source=original/dirname/f'{name}.parquet'
        data=pd.read_parquet(source)
        data['decision_timestamp']=data.entry_time_utc.dt.as_unit('ms').astype('int64')
        data=data.merge(raw,on=['symbol','decision_timestamp'],how='left',validate='one_to_one')
        for col in ADDED:data[col]=data.groupby('decision_timestamp')[col].rank(method='average',pct=True)
        data['features_complete']=np.isfinite(data[features]).all(axis=1)
        data['prediction_eligible_on_features']=data.features_complete&data.pool_known_at_decision
        assert data[features].stack().dropna().between(0,1,inclusive='right').all()
        data.drop(columns='decision_timestamp').to_parquet(destination/f'{name}.parquet',index=False)
        summary.append({'table':name,'pool_year':int(data.pool_year.iloc[0]),'rows':len(data),
                        'complete_feature_rows':int(data.features_complete.sum()),
                        'source_table_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
        print(name,'ready',len(data),flush=True)
pd.DataFrame(summary).to_csv(OUT/'table_summary.csv',index=False)
(OUT/'addon_method.json').write_text(json.dumps({'features':features,'added':ADDED,
 'data':'all original labels, original ten percentile columns and all member-date rows preserved',
 'new_columns':'original formulas on positive-volume contiguous 4h candles, legacy 25-bar warmup, pool/date percentile ranking independently per new factor',
 'training':'rolling one calendar year for the requested model; cumulative tables retained for audit'},indent=2))
archive=ROOT/'reports/xgb15_gpu_inputs.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for p in [OUT/'feature_columns.json',OUT/'addon_method.json',*sorted(OUT.glob('*_table_*/*.parquet'))]:
        z.write(p,p.relative_to(OUT.parent).as_posix())
print('INPUTS READY',round(archive.stat().st_size/1024**2,2),'MB',flush=True)
