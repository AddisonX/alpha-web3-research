from pathlib import Path
import pandas as pd,tarfile
R=Path(__file__).resolve().parents[2]/'reports'
with tarfile.open(R/'rolling2m_results.tar.gz') as z:z.extractall(R,filter='data')
O=R/'rolling2m_vs_expanding30';O.mkdir(exist_ok=True)
a=pd.read_csv(R/'xgb15_120_lr005_hysteresis_10_30/annual_net_metrics.csv').set_index('year');b=pd.read_csv(R/'xgb15_rolling2m_120_lr005_hysteresis_10_30/annual_net_metrics.csv').set_index('year');ic=pd.read_csv(R/'xgb15_rolling2m_120_lr005/annual_metrics.csv').set_index('prediction_year')
result=b.copy();result['IC']=ic.IC;result['ICIR']=ic.ICIR;result['expanding_net_return']=a.net_compound_return;result.to_csv(O/'annual_comparison.csv')
print(result.to_string());print('TOTAL',{'expanding':float((1+a.net_compound_return).prod()-1),'rolling2m':float((1+b.net_compound_return).prod()-1)})
models=pd.read_csv(R/'xgb15_rolling2m_120_lr005/monthly_training_records.csv');print('MODELS',len(models),'MIN_TRAIN',models.training_rows.min(),'MAX_TRAIN',models.training_rows.max());print(models.iloc[[0,-1]].to_string(index=False))
