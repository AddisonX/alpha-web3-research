from pathlib import Path
import tarfile,pandas as pd
R=Path(__file__).resolve().parents[2]/'reports'
with tarfile.open(R/'rolling2mraw_results.tar.gz') as z:z.extractall(R,filter='data')
O=R/'rolling2m_raw_vs_percentile';O.mkdir(exist_ok=True);ret=pd.DataFrame();stats=[];ic=pd.DataFrame()
for name,label in [('rolling2m','percentile'),('rolling2mraw','raw')]:
 f=pd.read_csv(R/f'xgb15_{name}_120_lr005_hysteresis_10_30/annual_net_metrics.csv').set_index('year');ret[label]=f.net_compound_return
 a=pd.read_csv(R/f'xgb15_{name}_120_lr005/annual_metrics.csv').set_index('prediction_year');ic[label+'_IC']=a.IC;ic[label+'_ICIR']=a.ICIR
 q=f.reset_index();q['input']=label;stats.append(q)
ret.index.name='year';ic.index.name='year';ret.to_csv(O/'returns.csv');ic.to_csv(O/'rank_ic.csv');pd.concat(stats).to_csv(O/'all_metrics.csv',index=False)
print(ret.map(lambda v:f'{v:.2%}').to_string());print('TOTAL',((1+ret).prod()-1).to_dict());print(ic.round(4).to_string());print('RAW',stats[1].to_string(index=False))
