from pathlib import Path
import tarfile,pandas as pd
r=Path(__file__).resolve().parents[2]/'reports'
with tarfile.open(r/'buffer30_results.tar.gz') as z:z.extractall(r,filter='data')
o=r/'buffer30_vs25_14_vs15';o.mkdir(exist_ok=True);ret=pd.DataFrame();turn=pd.DataFrame();rows=[]
for n in [15,14]:
 for b in [25,30]:
  f=pd.read_csv(r/f'xgb{n}_120_lr005_hysteresis_10_{b}/annual_net_metrics.csv').set_index('year');ret[f'{n}_buffer{b}']=f.net_compound_return;turn[f'{n}_buffer{b}']=f.mean_daily_oneway_turnover
  q=f.reset_index();q['factors']=n;q['buffer_pct']=b;rows.append(q)
ret.index.name='year';turn.index.name='year';ret.to_csv(o/'returns.csv');turn.to_csv(o/'turnover.csv');pd.concat(rows,ignore_index=True).to_csv(o/'all_metrics.csv',index=False)
print('RETURNS');print(ret.map(lambda v:f'{v:.2%}').to_string());print('TOTAL',((1+ret).prod()-1).to_dict());print('TURNOVER');print(turn.map(lambda v:f'{v:.2%}').to_string());print('MISSING');print(pd.concat(rows).query('buffer_pct==30')[['year','factors','estimated_return_days','estimated_fee_days']].to_string(index=False))
