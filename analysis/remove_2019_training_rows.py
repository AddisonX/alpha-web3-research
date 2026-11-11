import sys,json,hashlib
from pathlib import Path
import pandas as pd
root=Path(sys.argv[1])
old=root/'factors15_2019_2026';new=root/'factors15_2020_2026'
if old.exists():
    assert not new.exists()
    old.rename(new)
records=[]
for p in sorted((new/'train_table_2021-2025').glob('*.parquet')):
    df=pd.read_parquet(p);removed=int(df.data_year.eq(2019).sum())
    df=df.loc[df.data_year.ge(2020)].copy()
    df.to_parquet(p,index=False)
    records.append(dict(table=p.name,rows=len(df),removed_2019_rows=removed,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    print(p.name,'removed',removed,'remaining',len(df),flush=True)
for p in new.rglob('*coverage*.csv'):
    df=pd.read_csv(p)
    if 'data_year' in df:df=df.loc[df.data_year.ge(2020)]
    elif 'decision_date_beijing' in df:df=df.loc[df.decision_date_beijing.ge('2020-01-01')]
    df.to_csv(p,index=False)
for p in [new/'raw_features_2019_2020.parquet',*sorted((new/'raw_features_by_symbol').glob('*.parquet'))]:
    if p.exists():
        df=pd.read_parquet(p)
        df=df.loc[df.decision_timestamp>=1577808000000]
        df.to_parquet(p,index=False)
        if p.name=='raw_features_2019_2020.parquet':p.rename(p.with_name('raw_features_2020.parquet'))
m=json.loads((new/'manifest.json').read_text());m['start']='2020-01-01';m['tables']=records
m['removed_2019']=True;m['existing_model_results']='Prior from2019 model results are historical; not retrained after removal.'
(new/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
