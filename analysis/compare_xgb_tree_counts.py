"""Compare the existing 100/300/500-tree GPU runs without refitting models."""
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/xgb_100_300_500_comparison'
OUT.mkdir(parents=True,exist_ok=True)
rows=[];curves=[]
for trees in [100,300,500]:
    source=ROOT/f'reports/xgb10_gpu_{trees}_depth4_2022_2026'
    metrics=pd.read_csv(source/'annual_metrics.csv')
    daily=pd.read_parquet(source/'all_years_decile_daily.parquet')
    d=daily.drop_duplicates(['prediction_year','decision_timestamp'])
    for r in metrics.to_dict('records'):
        rows.append({'trees':trees,'period':str(int(r['prediction_year'])),'long_short_simple':r['G10_minus_G1'],
                     'IC':r['IC'],'ICIR':r['ICIR'],'days':int(r['valid_days'])})
    rows.append({'trees':trees,'period':'2022-2026','long_short_simple':float(metrics.G10_minus_G1.sum()),
                 'IC':float(d.ic.mean()),'ICIR':float(d.ic.mean()/d.ic.std(ddof=1)),'days':len(d)})
    ends=daily.loc[daily.decile.isin([1,10])].pivot(index=['prediction_year','decision_timestamp'],columns='decile',values='excess_return')
    p=(ends[10]-ends[1]).rename('long_short_daily').reset_index();p['trees']=trees
    curves.append(p)
result=pd.DataFrame(rows);result.to_csv(OUT/'comparison.csv',index=False)
fig=go.Figure()
for p in curves:
    p=p.sort_values('decision_timestamp')
    fig.add_scatter(x=pd.to_datetime(p.decision_timestamp+86400000+8*3600000,unit='ms'),y=p.long_short_daily.cumsum(),name=str(int(p.trees.iloc[0]))+' trees',mode='lines')
fig.update_layout(height=620,title='100 / 300 / 500 棵树 · G10−G1 累计多空单利收益',yaxis_tickformat='.0%',xaxis_title='收益兑现日期（北京时间）',yaxis_title='累计多空毛收益')
doc='<meta charset="utf-8"><title>XGBoost树数对比</title><style>body{font:15px system-ui;max-width:1400px;margin:30px auto;padding:0 20px;color:#203047}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccc;text-align:right}p{line-height:1.8}</style><h1>100 / 300 / 500 棵树 · GPU XGBoost 对比</h1><p>深度均为4，学习率0.05，seed42，不早停；使用相同年度训练表、预测表和评估口径。G10预测最高、G1最低，多空=G10−G1，未复利、不扣手续费、资金费和滑点。2026截至11月4日。</p><p>多头100%加空头100%，总名义敞口200%；若总名义敞口限制为100%，等权多空的该项毛收益应除以2。ICIR不年化；整体IC和ICIR按全部有效交易日重新计算，不取年度IR均值。因子筛选及参数比较均有历史回看影响，不能视为严格独立样本外检验。</p>'
for key,label in [('long_short_simple','累计多空收益：G10−G1'),('IC','年度平均IC'),('ICIR','年度ICIR')]:
    t=result.pivot(index='period',columns='trees',values=key).reindex(['2022','2023','2024','2025','2026','2022-2026'])
    t.to_csv(OUT/f'{key}.csv')
    rendered=t.map(lambda x:f'{x:.2%}' if key=='long_short_simple' else f'{x:.5f}')
    doc+='<h2>'+label+'</h2>'+rendered.to_html()
doc+='<script>'+get_plotlyjs()+'</script>'+fig.to_html(full_html=False,include_plotlyjs=False)
(OUT/'100_300_500棵树对比.html').write_text(doc,encoding='utf-8')
print(result.to_json(orient='records',indent=2))
