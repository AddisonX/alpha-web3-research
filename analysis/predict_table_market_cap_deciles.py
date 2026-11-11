"""Direct size deciles from predict_table percentiles; no model and no factor rebuild."""
import json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

ROOT=Path(__file__).resolve().parents[1]
TABLES=ROOT/'data/model_tables/factors10_2021_2026/predict_table_2022-2026'
OUT=ROOT/'reports/predict_table_market_cap_2022_2026'
OUT.mkdir(parents=True,exist_ok=True)
NAMES=[f'market_cap_mean_{w}d' for w in [10,20,30,40]]
COLORS=['#30123b','#4145ab','#4675ed','#39a2fc','#1bcfd4','#24eca6','#61fc6c','#b2f235','#faba39','#df2e16']
DAY=86400000


def summary(part,period):
    days=part.drop_duplicates('stamp')
    result={'factor':part.factor.iloc[0],'period':str(period),'days':len(days),
            'mean_members':days.factor_members.mean(),'percentile_IC':days.ic.mean(),'RankIC':days.rank_ic.mean()}
    for g in range(1,11):result[f'G{g}_excess_simple']=part.loc[part.decile.eq(g),'excess_return'].sum()
    result['G1_minus_G10']=result['G1_excess_simple']-result['G10_excess_simple']
    return result


def plot(part,name,period,path):
    start=pd.Timestamp('2022-01-01' if period=='2022-2026' else f'{period}-01-01')
    end=pd.to_datetime(part.date_beijing).max()
    dates=pd.date_range(start,end)
    stamps=dates.tz_localize('Asia/Shanghai').tz_convert('UTC').as_unit('ms').asi8
    fig,ax=plt.subplots(figsize=(13,6),layout='constrained');chart=go.Figure()
    for g,color in enumerate(COLORS,1):
        p=part.loc[part.decile.eq(g)].sort_values('stamp').copy()
        values=pd.Series(p.excess_return.cumsum().to_numpy(),index=p.stamp).reindex(stamps).to_numpy()
        ax.plot(dates+pd.Timedelta(days=1),values,label=f'G{g}',color=color,linewidth=1.3)
        chart.add_scatter(x=dates+pd.Timedelta(days=1),y=values,name=f'G{g}',mode='lines',connectgaps=False,line={'color':color})
    w=name.split('_')[-1]
    ax.set_title(f'{w} mean circulating market cap | predict_table {period}\nG1 smallest / G10 largest | cumulative excess simple return | gross')
    ax.set_xlabel('Exit date, Beijing time | daily 00:00 | supplied full-pool benchmark')
    ax.set_ylabel('Cumulative excess simple return')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2);ax.axhline(0,color='gray',linewidth=.6)
    ax.legend(ncol=10,fontsize=8);fig.savefig(path,dpi=140);plt.close(fig)
    chart.update_layout(height=580,title=f'{w} 平均市值 · {period}',yaxis_tickformat='.0%',yaxis_title='累计超额单利收益',xaxis_title='收益兑现日期（北京时间）',legend_orientation='h')
    return chart.to_html(full_html=False,include_plotlyjs=False)


rows=[]
for year in range(2022,2027):
    data=pd.read_parquet(TABLES/f'predict_table_{year}.parquet')
    # Same contemporaneously observable market-cap sample for all four horizons.
    # Do not require the other six factors or use future outcomes to choose members.
    current_market=data.factor_bar_close_utc.eq(data.entry_time_utc)
    eligible=data.loc[data.pool_known_at_decision&current_market&np.isfinite(data[NAMES]).all(axis=1)].sort_values(['entry_time_utc','symbol'])
    for stamp,frame in eligible.groupby('entry_time_utc',sort=True):
        if len(frame)<20:continue
        groups={name:np.array_split(np.argsort(frame[name].to_numpy(),kind='stable'),10) for name in NAMES}
        if frame.realized_return_24h.isna().any() or frame.benchmark_return_24h.isna().any():continue
        assert frame.benchmark_return_24h.nunique()==1
        r=frame.realized_return_24h.to_numpy();benchmark=float(frame.benchmark_return_24h.iloc[0]);y=r-benchmark
        for name in NAMES:
            x=frame[name].to_numpy()
            ic=float(np.corrcoef(x,y)[0,1])
            rank_ic=float(pd.Series(x).rank().corr(pd.Series(y).rank()))
            recent=[]
            for g,ix in enumerate(groups[name],1):
                recent.append({'factor':name,'year':year,'date_beijing':stamp.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d'),
                               'stamp':int(stamp.timestamp()*1000),'decile':g,'members':len(ix),'factor_members':len(frame),
                               'group_return':float(r[ix].mean()),'benchmark_return':benchmark,
                               'excess_return':float(r[ix].mean()-benchmark),'ic':ic,'rank_ic':rank_ic})
            assert np.isclose(sum(v['group_return']*v['members'] for v in recent)/len(frame),r.mean())
            rows.extend(recent)
    print('processed',year,flush=True)
daily=pd.DataFrame(rows).sort_values(['factor','decile','stamp'])
daily.to_parquet(OUT/'decile_daily.parquet',index=False)
metrics=[]
for name,p in daily.groupby('factor'):
    metrics.append(summary(p,'2022-2026'))
    metrics.extend(summary(q,y) for y,q in p.groupby('year'))
metrics=pd.DataFrame(metrics);metrics.to_csv(OUT/'metrics.csv',index=False)
doc='<meta charset="utf-8"><title>predict_table 市值十档分层</title><style>body{font:15px system-ui;max-width:1500px;margin:25px auto;padding:0 20px;color:#203047}p{line-height:1.8}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}.wide{overflow:auto}section{scroll-margin-top:60px}nav{position:sticky;top:0;padding:12px;background:white;z-index:100}a{margin-right:20px}</style><h1>predict_table · 2022—2026 · 平均市值十档分层</h1>'
doc+='<p>直接使用predict_table_2022—2026中的四个平均市值百分位，不经过XGBoost。G1为市值最小档，G10最大。四因子使用共同有效市值样本，并要求零点刚收盘的历史行情存在，避免把已无合约行情但仍有代币市值的币纳入；不要求其余六个行情因子值完整。北京时间零点调仓，因子时点与原表一致；年度币池已知后开始，首日为1月2日。</p><p>基准直接取表中的benchmark_return_24h：该年固定币池当天有完整价格收益的成员等权。分层样本与基准成员可以不同。每日超额逐日相加、不复利、不扣成本。先按因子分组，再评价收益；预测成员缺真实收益的日期不评分。2026只计已有真实收益的区间。IC为百分位因子Pearson相关，与之前原始市值IC口径不同。整体曲线跨年连续，年度图每年归零。</p>'
doc+='<nav>'+''.join(f'<a href="#w{w}">{w}天</a>' for w in [10,20,30,40])+'</nav><script>'+get_plotlyjs()+'</script>'
for name,p in daily.groupby('factor',sort=True):
    w=name.split('_')[-1][:-1]
    doc+=f'<section id="w{w}"><h2>{w}天平均市值</h2>'+plot(p,name,'2022-2026',OUT/f'{name}_2022_2026.png')
    show=metrics.loc[metrics.factor.eq(name)].drop(columns='factor').copy()
    for c in show.columns:
        if 'excess_simple' in c or c=='G1_minus_G10':show[c]=show[c].map(lambda x:f'{x:.2%}')
    doc+='<div class="wide">'+show.to_html(index=False,float_format=lambda x:f'{x:.4f}')+'</div><details><summary>展开2022—2026逐年图</summary>'
    for year,q in p.groupby('year'):
        doc+=plot(q,name,str(year),OUT/f'{name}_{year}.png')
    doc+='</details></section>'
doc+='<p>历史市值可能修订；因子及币池存在事后筛选历史；零点成交和滑点尚未模拟。</p>'
(OUT/'predict_table市值分层_2022-2026.html').write_text(doc,encoding='utf-8')
(OUT/'method.json').write_text(json.dumps({'input':'predict_table_2022-2026 only','factors':NAMES,'sample':'same four-cap-complete members; last positive-volume historical candle closes at decision; pool known at decision; no six-factor completeness condition',
 'benchmark':'provided benchmark_return_24h, not recomputed on factor subset','g1':'smallest mean cap','g10':'largest mean cap','sign_flipping':False,
 'aggregation':'sum of daily excess, gross, not compounded','annual':'reset each year','overall':'continuous 2022-2026'},indent=2))
archive=OUT.parent/'predict_table_市值十档_2022-2026.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in OUT.iterdir():
        if p.is_file():z.write(p,'predict_table_market_cap_2022_2026/'+p.name)
print(metrics.loc[metrics.period.eq('2022-2026')].to_string(index=False),flush=True)
print('COMPLETE',flush=True)
