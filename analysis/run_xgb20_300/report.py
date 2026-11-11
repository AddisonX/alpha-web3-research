from pathlib import Path
import pandas as pd,numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.dates import MonthLocator,DateFormatter
M=Path('/tmp/xgb20_gpu_expanding_300_depth4_lr005_2022_2026');N=Path('/tmp/xgb20_300_lr005_hysteresis_10_30')
ic=pd.read_csv(M/'annual_metrics.csv');net=pd.read_csv(N/'annual_net_metrics.csv')
old=pd.read_csv('/tmp/old15_net.csv');oi=pd.read_csv('/tmp/old15_ic.csv')
result=net.merge(ic[['prediction_year','RankIC','RankICIR','mean_predicted_members']],left_on='year',right_on='prediction_year')
result=result.merge(old[['year','net_compound_return','max_drawdown','sharpe','mean_daily_oneway_turnover']].add_prefix('old15_'),left_on='year',right_on='old15_year')
result=result.merge(oi[['prediction_year','RankIC','RankICIR']].add_prefix('old15_'),left_on='year',right_on='old15_prediction_year')
result.to_csv(M/'comparison.csv',index=False)
d=pd.read_parquet(M/'all_years_decile_daily.parquet');colors=plt.cm.turbo(np.linspace(.05,.95,10))
fig,axes=plt.subplots(3,2,figsize=(15,13))
for year,ax in zip(range(2022,2027),axes.flat):
 p=d.loc[d.prediction_year.eq(year)];grid=pd.date_range(f'{year}-01-01',p.date_beijing.max())
 for g,color in zip(range(1,11),colors):
  s=p.loc[p.decile.eq(g)].sort_values('date_beijing');v=pd.Series(s.excess_return.cumsum().to_numpy(),index=pd.to_datetime(s.date_beijing)).reindex(grid)
  ax.plot(grid+pd.Timedelta(days=1),v,color=color,label=f'G{g}',lw=1.2)
 z=ic.loc[ic.prediction_year.eq(year)].iloc[0]
 ax.set_title(f'{year} | Rank IC {z.RankIC:.4f} | ICIR {z.RankICIR:.3f}')
 ax.yaxis.set_major_formatter(PercentFormatter(1));ax.xaxis.set_major_locator(MonthLocator(interval=3));ax.xaxis.set_major_formatter(DateFormatter('%b'));ax.grid(alpha=.2);ax.axhline(0,color='gray',lw=.6)
axes.flat[-1].axis('off')
axes.flat[-1].text(.05,.8,'20 percentile factors\nExpanding from2021; annual refit\n300 trees | depth4 | eta0.05\nGPU RTX4090\n\nCharts: gross cumulative SIMPLE excess\nBenchmark: full-pool equalweight\nG1 low score; G10 high score\nNo fees, funding or slippage in these charts',va='top',fontsize=12)
fig.suptitle('XGBoost20 | Annual decile excess returns',fontsize=19)
fig.legend(*axes.flat[0].get_legend_handles_labels(),ncol=10,loc='upper center',bbox_to_anchor=(.5,.954))
fig.tight_layout(rect=[0,.03,1,.92]);fig.savefig(M/'annual_deciles.png',dpi=130);plt.close(fig)
daily=pd.read_csv(N/'daily_net_equity.csv');baseline=pd.read_csv('/tmp/old15_daily.csv')
fig,ax=plt.subplots(figsize=(13,5))
ax.plot(pd.to_datetime(baseline.date_beijing)+pd.Timedelta(days=1),baseline.net_equity,label='Original15 | 120 trees | eta0.05')
ax.plot(pd.to_datetime(daily.date_beijing)+pd.Timedelta(days=1),daily.net_equity,label='New20 | 300 trees | eta0.05')
ax.set_title('30% buffer | 50% long + 50% short | daily equalweight | 0.05% fee per traded notional');ax.set_ylabel('Net compounded equity');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(M/'net_equity_comparison.png',dpi=140);plt.close(fig)
print(result.to_string(index=False),flush=True)
print('NEW TOTAL',daily.net_equity.iloc[-1]-1,'OLD TOTAL',baseline.net_equity.iloc[-1]-1,flush=True)
