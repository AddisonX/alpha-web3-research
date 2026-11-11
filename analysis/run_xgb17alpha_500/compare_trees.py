from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/tmp/xgb17alpha_gpu_expanding_500_depth4_lr005_2022_2026')
rows=[];total=[]
fig,ax=plt.subplots(figsize=(13,5))
for trees in [150,300,500]:
 m=Path(f'/tmp/xgb17alpha_gpu_expanding_{trees}_depth4_lr005_2022_2026')
 n=Path(f'/tmp/xgb17alpha_{trees}_lr005_hysteresis_10_30')
 a=pd.read_csv(n/'annual_net_metrics.csv');ic=pd.read_csv(m/'annual_metrics.csv')
 a=a.merge(ic[['prediction_year','RankIC','RankICIR']],left_on='year',right_on='prediction_year');a['trees']=trees;rows.append(a)
 d=pd.read_csv(n/'daily_net_equity.csv');total.append(dict(trees=trees,total_net_return=d.net_equity.iloc[-1]-1))
 ax.plot(pd.to_datetime(d.date_beijing)+pd.Timedelta(days=1),d.net_equity,label=f'{trees} trees')
pd.concat(rows,ignore_index=True).to_csv(O/'tree_comparison.csv',index=False)
pd.DataFrame(total).to_csv(O/'tree_totals.csv',index=False)
ax.set_title('17 factors | eta0.05 | depth4 | expanding2021 | 30% buffer | 0.05% fee')
ax.set_ylabel('Net compounded equity');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(O/'tree_net_equity_comparison.png',dpi=140);plt.close(fig)
print(pd.concat(rows).pivot(index='year',columns='trees',values='net_compound_return').to_string(),flush=True)
print(pd.DataFrame(total).to_string(index=False),flush=True)
