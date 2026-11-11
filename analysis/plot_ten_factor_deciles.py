from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.dates import MonthLocator,DateFormatter
import zipfile
root=Path(__file__).resolve().parents[1]/'reports/ten_factor_deciles_2021_2026'
df=pd.read_parquet(root/'decile_daily.parquet')
colors=['#30123b','#4145ab','#4675ed','#39a2fc','#1bcfd4','#24b99d','#56bd69','#a9b82e','#ef9d20','#d92b16']
plt.rcParams.update({'font.size':10,'axes.titlesize':13,'axes.labelsize':10})
for name,part in df.groupby('factor',sort=True):
    fig,axes=plt.subplots(3,2,figsize=(16,14))
    for year,ax in zip(range(2021,2027),axes.flat):
        annual=part.loc[part.year.eq(year)]
        grid=pd.date_range(f'{year}-01-01',f'{year}-12-31' if year<2026 else '2026-10-27')
        for g in range(1,11):
            p=annual.loc[annual.decile.eq(g)].sort_values('date_beijing')
            s=pd.Series(p.excess_return.cumsum().to_numpy(),index=pd.to_datetime(p.date_beijing))
            line=s.reindex(grid)
            ax.plot(grid+pd.Timedelta(days=1),line,label=f'G{g}',color=colors[g-1],linewidth=1.4)
        ax.set_title(str(year)+(' (through Oct 28)' if year==2026 else ''))
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2);ax.axhline(0,color='#555',linewidth=.7)
        ax.xaxis.set_major_locator(MonthLocator(interval=2));ax.xaxis.set_major_formatter(DateFormatter('%b'))
        ax.set_ylabel('Cumulative excess simple return')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=10,loc='upper center',bbox_to_anchor=(.5,.94),frameon=False)
    fig.suptitle(name,fontsize=20,y=.985)
    fig.text(.5,.958,'G1 = lowest factor values | G10 = highest | Full-pool equal-weight benchmark | Gross, before costs',ha='center',fontsize=11)
    fig.text(.5,.017,'Daily Beijing 00:00; dates show return realization. 2021: retrospective 2022 pool. Later years: same-year pools. Missing days shown as gaps.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.915],h_pad=2,w_pad=2)
    fig.savefig(root/f'{name}.png',dpi=130,facecolor='white');plt.close(fig)
    print(name,flush=True)
with zipfile.ZipFile(root/'ten_factor_annual_decile_charts.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.glob('*.png')):z.write(p,p.name)
