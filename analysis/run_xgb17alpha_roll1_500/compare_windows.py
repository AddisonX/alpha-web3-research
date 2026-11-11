from pathlib import Path
import pandas as pd,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/tmp/xgb17alpha_gpu_roll1_500_depth4_lr005_2022_2026')
rows=[];totals=[];fig,ax=plt.subplots(figsize=(13,5))
for window,model,net in [('expanding2021','xgb17alpha_gpu_expanding_500_depth4_lr005_2022_2026','xgb17alpha_500_lr005_hysteresis_10_30'),('rolling1year','xgb17alpha_gpu_roll1_500_depth4_lr005_2022_2026','xgb17alpha_roll1_500_lr005_hysteresis_10_30')]:
 m=Path('/tmp')/model;n=Path('/tmp')/net
 a=pd.read_csv(n/'annual_net_metrics.csv');ic=pd.read_csv(m/'annual_metrics.csv')
 a=a.merge(ic[['prediction_year','RankIC','RankICIR']],left_on='year',right_on='prediction_year');a['window']=window;rows.append(a)
 d=pd.read_csv(n/'daily_net_equity.csv');totals.append(dict(window=window,total_net_return=d.net_equity.iloc[-1]-1))
 ax.plot(pd.to_datetime(d.date_beijing)+pd.Timedelta(days=1),d.net_equity,label=window)
pd.concat(rows,ignore_index=True).to_csv(O/'window_comparison.csv',index=False);pd.DataFrame(totals).to_csv(O/'window_totals.csv',index=False)
ax.set_title('17 factors | 500 trees | eta0.05 | depth4 | 30% buffer | 0.05% fee');ax.set_ylabel('Net compounded equity');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(O/'window_net_equity_comparison.png',dpi=140);plt.close(fig)
print(pd.concat(rows).pivot(index='year',columns='window',values='net_compound_return').to_string(),flush=True)
print(pd.DataFrame(totals).to_string(index=False),flush=True)
