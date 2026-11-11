from pathlib import Path
import pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.dates import MonthLocator,DateFormatter
import zipfile
ROOT=Path(__file__).resolve().parents[1]/'reports/funding_four_factors'
d=pd.read_parquet(ROOT/'decile_daily.parquet')
colors=['#30123b','#4145ab','#4675ed','#39a2fc','#1bcfd4','#24b99d','#56bd69','#a9b82e','#ef9d20','#d92b16']
for name,p in d.groupby('factor',sort=True):
    fig,axes=plt.subplots(5,2,figsize=(16,20))
    for i,year in enumerate(range(2022,2027)):
        q=p[p.year==year]
        grid=pd.date_range(f'{year}-01-01',f'{year}-12-31' if year<2026 else '2026-10-27')
        for j,col in enumerate(['price_excess','funding_adjusted_excess']):
            ax=axes[i,j]
            for g in range(1,11):
                a=q[q.decile==g].sort_values('date_beijing')
                s=pd.Series(a[col].cumsum().to_numpy(),index=pd.to_datetime(a.date_beijing)).reindex(grid)
                ax.plot(grid+pd.Timedelta(days=1),s,label=f'G{g}',color=colors[g-1],linewidth=1.2)
            ax.set_title(f'{year} | '+('Price excess' if j==0 else 'Price + funding excess'))
            ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2);ax.axhline(0,color='#555',linewidth=.6)
            ax.xaxis.set_major_locator(MonthLocator(interval=2));ax.xaxis.set_major_formatter(DateFormatter('%b'))
    h,l=axes[0,0].get_legend_handles_labels()
    fig.suptitle(name,fontsize=19,y=.985)
    fig.legend(h,l,ncol=10,loc='upper center',bbox_to_anchor=(.5,.966),frameon=False)
    fig.text(.5,.943,'G1 lowest / G10 highest | Cumulative excess simple returns | Before transaction fees',ha='center')
    fig.text(.5,.014,'Beijing00:00 decisions; simultaneous funding excluded from signals. Missing evaluation days shown as gaps. 2026 throughOct28.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.03,1,.932],h_pad=2,w_pad=2)
    fig.savefig(ROOT/f'{name}.png',dpi=120);plt.close(fig)
    print(name,flush=True)
with zipfile.ZipFile(ROOT/'four_funding_factor_charts.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.glob('*.png'):z.write(p,p.name)
