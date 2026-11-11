from pathlib import Path
import json,zipfile
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
R=Path('/tmp/alpha191_16')
a=pd.read_csv(R/'annual_rank_ic.csv');n=pd.read_csv(R/'annual_net_metrics.csv')
d=pd.read_parquet(R/'decile_daily.parquet');ic=pd.read_csv(R/'daily_rank_ic.csv')
colors=plt.cm.turbo(np.linspace(.05,.95,10))
for name,p in d.groupby('factor',sort=True):
 fig,axes=plt.subplots(3,2,figsize=(15,12))
 for year,ax in zip(range(2021,2027),axes.flat):
  for g,c in zip(range(1,11),colors):
   s=p.loc[p.year.eq(year)&p.decile.eq(g)].sort_values('date')
   ax.plot(pd.to_datetime(s.date)+pd.Timedelta(days=1),s.excess_return.cumsum(),color=c,label=f'G{g}',lw=1.1)
  z=a.loc[a.year.eq(year)&a.factor.eq(name)].iloc[0]
  ax.set_title(f'{year} | Rank IC {z.RankIC:.4f} | ICIR {z.RankICIR:.3f}')
  ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2);ax.axhline(0,color='gray',lw=.6)
 fig.suptitle(name+' | 4h inputs, daily Beijing 00:00',fontsize=17)
 fig.legend(*axes.flat[0].get_legend_handles_labels(),ncol=10,loc='upper center',bbox_to_anchor=(.5,.95))
 fig.text(.5,.015,'Gross cumulative SIMPLE excess vs full-pool equal weight. G1 lowest; G10 highest. No fees/funding/slippage.\n2021 uses retrospective 2022 pool; 2026 through Oct28 realization. Incomplete diagnostic dates shown as gaps.',ha='center',fontsize=9)
 fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(R/f'{name}.png',dpi=120);plt.close(fig)
p=a.pivot(index='factor',columns='year',values='RankIC').sort_index()
fig,ax=plt.subplots(figsize=(10,9));v=max(abs(p.to_numpy()).max(),.01)
im=ax.imshow(p.to_numpy(),cmap='RdBu',vmin=-v,vmax=v,aspect='auto')
ax.set_xticks(range(6),p.columns);ax.set_yticks(range(len(p)),p.index)
for i in range(len(p)):
 for j in range(6):ax.text(j,i,f'{p.iloc[i,j]:.4f}',ha='center',va='center',color='white' if abs(p.iloc[i,j])>.65*v else 'black',fontsize=10)
ax.set_title('Alpha191 subset: annual Rank IC (Spearman)\n2021 retrospective; 2026 partial through Oct27')
fig.colorbar(im,ax=ax,label='Mean daily Rank IC');fig.tight_layout();fig.savefig(R/'annual_rank_ic_heatmap.png',dpi=150);plt.close(fig)
rows=[]
for factor,g in ic.groupby('factor'):
 v=g.loc[g.year.between(2022,2026),'rank_ic'].dropna()
 rows.append(dict(factor=factor,rank_ic_2022_2026=v.mean(),rank_icir_2022_2026=v.mean()/v.std(),days=len(v)))
s=pd.DataFrame(rows)
for direction,tag in [('high_long_low_short','high_long'),('low_long_high_short','low_long')]:
 p=n.loc[n.direction.eq(direction)&n.year.ge(2022)].pivot(index='factor',columns='year',values='net_return')
 p.columns=[f'{tag}_{c}' for c in p.columns]
 s=s.merge(p,on='factor')
 s[f'{tag}_total']=s[[f'{tag}_{y}' for y in range(2022,2027)]].add(1).prod(axis=1)-1
s.to_csv(R/'summary.csv',index=False)
with zipfile.ZipFile(R/'alpha191_16_charts.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in R.glob('*.png'):z.write(f,f.name)
print(s.to_string(index=False),flush=True)
