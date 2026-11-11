"""Screen fixed G1/G10 legs for persistent extreme position and temporal trend."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/persistent_extreme_factors'
OUT.mkdir(parents=True,exist_ok=True)
source=ROOT/'reports/达标60因子_2021-2025整体汇总.html'
page=source.read_text(encoding='utf-8'); start=page.index('const factors=')+len('const factors=')
payload,_=json.JSONDecoder().raw_decode(page[start:])
frames={}
for f in payload:
    curve=pd.DataFrame(np.array(f['curves'],dtype=float).T,index=pd.to_datetime(f['dates']),columns=range(1,11))
    daily=curve.ffill().diff().where(curve.notna())
    daily.index=daily.index-pd.Timedelta(days=1)
    frames[f['name']]=daily.loc['2021-01-01':'2025-12-31']
new=pd.read_parquet(ROOT/'reports/average_market_cap_2021_2025/decile_daily.parquet')
for name,part in new.groupby('factor'):
    frames[name]=part.pivot(index='decision_date_beijing',columns='decile',values='excess_return')
    frames[name].index=pd.to_datetime(frames[name].index)
assert len(frames)==64
profiles={
 '宽松':{'top2':.50,'top1':.20,'rolling':.65,'years':3,'rho':.65,'drawdown':.80,'recent':False},
 '基准':{'top2':.65,'top1':.30,'rolling':.75,'years':4,'rho':.80,'drawdown':.50,'recent':True},
 '严格':{'top2':.80,'top1':.50,'rolling':.85,'years':5,'rho':.90,'drawdown':.30,'recent':True}}
records=[]
for name,raw in sorted(frames.items()):
    valid=raw.dropna()
    assert len(valid)>1500
    cum=valid.cumsum()
    for g in [1,10]:
        for sign,label in [(1,'做多超额'),(-1,'做空超额')]:
            r=valid[g]*sign
            c=r.cumsum()
            signed=cum*sign
            occupied=signed.iloc[180:]
            rank=occupied.rank(axis=1,ascending=False,method='min')[g]
            annual=r.groupby(r.index.year).sum()
            calendar=r.reindex(pd.date_range('2021-01-01','2025-12-31'))
            roll=calendar.rolling(365,min_periods=330).sum().dropna()
            total=float(c.iloc[-1])
            peak=np.maximum.accumulate(np.r_[0,c.to_numpy()])
            dd=float(np.max(peak-np.r_[0,c.to_numpy()]))
            rec={'factor':name,'endpoint':f'G{g}','direction':label,'sign':sign,'valid_days':len(valid),
                 'total_excess_simple':total,'extreme_top1_fraction':float(rank.eq(1).mean()),
                 'extreme_top2_fraction':float(rank.le(2).mean()),
                 'rolling365_positive_fraction':float(roll.gt(0).mean()),
                 'positive_years':int(annual.gt(0).sum()),'positive_2025':bool(annual.loc[2025]>0),
                 'time_rank_correlation':float(c.rank().corr(pd.Series(np.arange(len(c)),index=c.index))),
                 'max_drawdown_simple':dd,'drawdown_over_gain':dd/total if total>0 else np.inf}
            rec.update({f'return_{y}':float(annual.loc[y]) for y in range(2021,2026)})
            for profile,p in profiles.items():
                tests={'累计为正':total>0,'长期前二':rec['extreme_top2_fraction']>=p['top2'],
                       '长期极值':rec['extreme_top1_fraction']>=p['top1'],
                       '滚动一年向上':rec['rolling365_positive_fraction']>=p['rolling'],
                       '年度持续':rec['positive_years']>=p['years'],
                       '时间趋势':rec['time_rank_correlation']>=p['rho'],
                       '回撤受控':rec['drawdown_over_gain']<=p['drawdown'],
                       '最近一年':not p['recent'] or rec['positive_2025']}
                rec[f'pass_{profile}']=all(tests.values())
                if profile=='基准':
                    rec['failure_reasons']='、'.join(k for k,v in tests.items() if not v)
                    rec['failure_count']=sum(not v for v in tests.values())
            records.append(rec)
legs=pd.DataFrame(records)
legs.to_csv(OUT/'all_256_endpoint_checks.csv',index=False)
best=legs.sort_values(['factor','pass_基准','failure_count','extreme_top2_fraction','rolling365_positive_fraction'],ascending=[True,False,True,False,False]).drop_duplicates('factor').sort_values('factor')
best.to_csv(OUT/'all_64_factors.csv',index=False)
kept=best.loc[best['pass_基准']]
kept.to_csv(OUT/'retained_factors.csv',index=False)
best.loc[~best['pass_基准']].to_csv(OUT/'excluded_factors.csv',index=False)
summary={'candidate_count':64,'candidate_scope':'previous 60 screened factors + four arithmetic average market-cap factors; no new IC filter',
         'retained':len(kept),'names':kept.factor.tolist(),
         'sensitivity':{p:int(legs.loc[legs[f'pass_{p}'],'factor'].nunique()) for p in profiles},
         'profiles':profiles,'definition':'fixed G1/G10, long excess or negative excess (short relative contribution), never change direction by year',
         'occupancy':'after first 180 valid observations; count days in best 2 of signed ten curves and best 1; signed means lowest for original short end',
         'trend':'Spearman correlation of cumulative signed excess and time; NOT strict daily monotonicity',
         'rolling':'365 calendar days, minimum 330 valid returns; all years 2021-2025',
         'returns':'same full-pool equal-weight benchmark and gross simple returns as prior reports; short leg means benchmark return minus group return, not naked short absolute PnL',
         'missing':'omit incomplete ten-decile days; never fill missing returns with zero for rolling metrics',
         'limitations':['in-sample descriptive screening; four possible fixed legs per factor examined','past-selected pools and factors; not independent out-of-sample validation','no costs, funding or liquidity constraints'],
         'validation':'old report cumulative curves are converted to daily returns with the anchor removed; dates shifted from exit to decision day'}
(OUT/'method_and_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
doc='<meta charset="utf-8"><title>因子端点持续性筛选</title><style>body{font:14px system-ui;margin:25px;color:#203047}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}p{line-height:1.8}section{scroll-margin-top:60px}nav{position:sticky;top:0;background:white;padding:15px;z-index:100}.wide{overflow:auto}</style>'
doc+=f'<h1>64因子端点持续性筛选 · 保留{len(kept)}个</h1><p>之前60因子加4个平均市值。逐因子检查固定G1/G10的做多超额及做空超额贡献，不逐年反转。做空超额是全池收益减该档收益，非裸空绝对盈亏。判断曲线长期位于最高或最低，并持续向上或向下，允许正常日波动。</p>'
doc+='<p>基准规则：跳过最初180个有效日后，累计曲线处于极值前二的时间≥65%，恰为最高/最低≥30%；滚动一年方向正确比例≥75%；五年至少四年方向正确且2025为正；累计曲线与时间的秩相关≥0.8；最大回撤不超过最终累计收益的50%。任意一个固定端点全部满足即保留。规则是本次描述性筛选约定，非统计显著性检验。</p>'
doc+='<p>宽松/基准/严格保留数：'+str(summary['sensitivity'])+'。历史曲线使用各自原报告的有效日，四个平均市值因子从2021-02-12开始；有效日数量列明。未计手续费、滑点及资金费；既有因子和币池的事后选择偏差仍然存在。</p>'
doc+='<nav>搜索定位 <input id="q"><button onclick="go()">定位</button> <label><input type="checkbox" id="all" onchange="toggle()">显示剔除因子的图</label></nav>'
display=best[['factor','endpoint','direction','pass_基准','positive_years','extreme_top1_fraction','extreme_top2_fraction','rolling365_positive_fraction','time_rank_correlation','drawdown_over_gain','failure_reasons']]
doc+='<h2>全部64个判定</h2><div class="wide">'+display.to_html(index=False,float_format=lambda v:f'{v:.3f}')+'</div><script>'+get_plotlyjs()+'</script>'
for row in best.itertuples(index=False,name=None):
    r=dict(zip(best.columns,row)); name=r['factor']; valid=frames[name].dropna(); c=valid.cumsum(); chart=go.Figure()
    for g in range(1,11):
        chart.add_scatter(x=c.index,y=c[g],name=f'G{g}',mode='lines',line={'width':3 if f'G{g}'==r['endpoint'] else 1})
    chart.update_layout(height=420,title=f'{name} | {r["endpoint"]} {r["direction"]}',yaxis_tickformat='.0%',yaxis_title='原始累计超额单利收益',legend_orientation='h')
    cls='keep' if r['pass_基准'] else 'exclude'
    doc+=f'<section id="{name}" class="{cls}" style="display:{"block" if cls=="keep" else "none"}"><h2>{name} · {"保留" if cls=="keep" else "剔除"}</h2>'+chart.to_html(full_html=False,include_plotlyjs=False)+'</section>'
doc+='''<script>function toggle(){document.querySelectorAll('.exclude').forEach(x=>x.style.display=document.getElementById('all').checked?'block':'none')}function go(){let q=document.getElementById('q').value.trim().toLowerCase();let e=[...document.querySelectorAll('section')].find(e=>e.id.toLowerCase().includes(q));if(e){e.style.display='block';e.scrollIntoView({behavior:'smooth'})}}document.getElementById('q').onkeydown=e=>{if(e.key==='Enter')go()};</script>'''
(OUT/'因子端点持续性筛选.html').write_text(doc,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
