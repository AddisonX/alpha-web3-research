from pathlib import Path
import pandas as pd,json
R=Path(__file__).resolve().parents[2]/'reports';O=R/'xgb16_vs_original15';O.mkdir(exist_ok=True)
old='xgb15_gpu_expanding_120_depth4_lr005_2022_2026';new='xgb16_gpu_expanding_120_depth4_lr005_2022_2026'
a=pd.read_csv(R/old/'annual_metrics.csv').set_index('prediction_year');b=pd.read_csv(R/new/'annual_metrics.csv').set_index('prediction_year')
ic=pd.DataFrame({'15_IC':a.RankIC,'16_IC':b.RankIC,'15_ICIR':a.RankICIR,'16_ICIR':b.RankICIR});ic.index.name='year';ic.to_csv(O/'rank_ic_comparison.csv')
ret=pd.DataFrame(index=a.index);detail=[]
for count in [15,16]:
 for kind,suffix in [('direct','net_compound_fee0005'),('buffer','hysteresis_10_15')]:
  f=pd.read_csv(R/f'xgb{count}_120_lr005_{suffix}'/'annual_net_metrics.csv').set_index('year');ret[f'{count}_{kind}']=f.net_compound_return
  q=f.reset_index();q['factors']=count;q['strategy']=kind;detail.append(q)
ret.index.name='year';ret.to_csv(O/'annual_returns_comparison.csv');pd.concat(detail,ignore_index=True).to_csv(O/'all_metrics_comparison.csv',index=False)
print('RETURNS');print(ret.map(lambda x:f'{x:.2%}').to_string());print('RANK_IC');print(ic.round(5).to_string())
print('COMPOUND_TOTAL',((1+ret).prod()-1).to_dict())
