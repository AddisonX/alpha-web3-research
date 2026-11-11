from pathlib import Path
import zipfile
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.dates import YearLocator,DateFormatter
R=Path('/tmp/alpha191_all_pct');O=R/'charts42';O.mkdir(exist_ok=True)
s=pd.read_csv(R/'selected_factors.csv').sort_values('factor');assert len(s)==42
d=pd.read_parquet(R/'decile_daily.parquet');colors=plt.cm.turbo(np.linspace(.04,.96,10))
for row in s.itertuples():
 q=d.loc[d.factor.eq(row.factor)&d.year.between(2021,2025)].copy()
 fig,ax=plt.subplots(figsize=(12,5.8))
 for g,col in zip(range(1,11),colors):
  z=q.loc[q.decile.eq(g)].sort_values('date');dates=pd.to_datetime(z.date)+pd.Timedelta(days=1)
  curve=pd.Series(z.excess_return.cumsum().to_numpy(),index=dates).reindex(pd.date_range('2021-01-01','2026-01-01'))
  curve.iloc[0]=0
  ax.plot(curve.index,curve.values,label=f'G{g}',color=col,lw=1.9 if g in [1,10] else 1.15,alpha=1 if g in [1,10] else .85)
 ax.set_title(f"Alpha{row.factor[-3:]} | 2021-2025 | Rank IC {row.RankIC:+.5f}",fontsize=15,pad=14)
 ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('Cumulative simple excess return')
 ax.xaxis.set_major_locator(YearLocator());ax.xaxis.set_major_formatter(DateFormatter('%Y'));ax.set_xlim(pd.Timestamp('2021-01-01'),pd.Timestamp('2026-01-01'))
 ax.axhline(0,color='#333333',lw=.8);ax.grid(alpha=.18)
 ax.legend(ncol=10,loc='upper center',bbox_to_anchor=(.5,-.09),frameon=False,fontsize=9)
 fig.text(.5,.025,'G1 lowest / G10 highest. Benchmark: full-pool equal weight. Gross, before fees/funding/slippage.\nDaily Beijing 00:00 signals; plotted on return realization date. Missing diagnostic days are gaps; 2021 uses retrospective 2022 pool.',ha='center',fontsize=8,color='#555555')
 fig.tight_layout(rect=[0,.11,1,1]);fig.savefig(O/f'{row.factor}.png',dpi=130,facecolor='white');plt.close(fig)
with zipfile.ZipFile(R/'alpha191_selected42_charts.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(O.glob('*.png')):z.write(p,'charts42/'+p.name)
print('Saved',len(list(O.glob('*.png'))),'charts',flush=True)
