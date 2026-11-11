from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parents[2]/'reports';O=R/'xgb18_vs17delta';O.mkdir(exist_ok=True)
ret=pd.DataFrame();ic=pd.DataFrame();detail=[]
for case,label in [('17delta','old17'),('18','new18')]:
 a=pd.read_csv(R/f'xgb{case}_gpu_expanding_120_depth4_lr005_2022_2026/annual_metrics.csv').set_index('prediction_year')
 ic[f'{label}_IC']=a.RankIC;ic[f'{label}_ICIR']=a.RankICIR
 for kind,suffix in [('direct','net_compound_fee0005'),('buffer','hysteresis_10_15')]:
  f=pd.read_csv(R/f'xgb{case}_120_lr005_{suffix}/annual_net_metrics.csv').set_index('year');ret[f'{label}_{kind}']=f.net_compound_return
  q=f.reset_index();q['case']=label;q['strategy']=kind;detail.append(q)
ret.index.name='year';ic.index.name='year';ret.to_csv(O/'annual_returns.csv');ic.to_csv(O/'rank_ic.csv');pd.concat(detail,ignore_index=True).to_csv(O/'all_metrics.csv',index=False)
print(ret.map(lambda x:f'{x:.2%}').to_string());print('TOTAL',((1+ret).prod()-1).to_dict());print(ic.round(4).to_string())
