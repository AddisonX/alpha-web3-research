"""Compare 10/20/30/40-calendar-day arithmetic average circulating market caps."""
from __future__ import annotations
import hashlib
import html
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/average_market_cap_2021_2025'
FUND=ROOT/'data/fundamentals/cmc_2021_2025/contract_daily.parquet'
RETURNS=ROOT/'data/returns/annual_pools_2021_to_t'
WINDOWS=[10,20,30,40]
DAY=86400000
LAG=24
OUT.mkdir(parents=True,exist_ok=True)


def corr(x,y):
    if np.ptp(x)==0 or np.ptp(y)==0:
        return np.nan
    return float(np.corrcoef(x,y)[0,1])


def metrics(part,period):
    days=part.drop_duplicates('decision_timestamp')
    row={'factor':part.factor.iloc[0],'period':period,'valid_days':len(days),
         'mean_factor_members':float(days.factor_members.mean())}
    for name,label in [('ic','IC'),('rank_ic','RankIC')]:
        vals=days[name].dropna()
        row[label]=float(vals.mean())
        row[label+'IR']=float(vals.mean()/vals.std(ddof=1)) if vals.std(ddof=1)>0 else None
    for g in range(1,11):
        row[f'G{g}_excess_simple']=float(part.loc[part.decile==g,'excess_return'].sum())
    row['G1_minus_G10']=row['G1_excess_simple']-row['G10_excess_simple']
    return row


def main():
    fund=pd.read_parquet(FUND)
    fund['date']=pd.to_datetime(fund.date)
    fund.loc[(fund.market_cap_usd<=0)|(fund.marketcap_identity_error>.01),'market_cap_usd']=np.nan
    cap=fund.pivot(index='date',columns='contract',values='market_cap_usd').sort_index()
    ids=fund.pivot(index='date',columns='contract',values='cmc_id').reindex_like(cap)
    # The source grid is daily; missing values invalidate the complete window.
    assert cap.index.equals(pd.date_range('2021-01-01','2025-12-31'))
    factors={}
    for w in WINDOWS:
        roll=cap.rolling(w,min_periods=w).mean()
        stable=ids.rolling(w,min_periods=w).min().eq(ids.rolling(w,min_periods=w).max())
        factors[w]=roll.where(stable)
    rows,coverage,aligned,manifest_years=[],[],[],[]
    scope=json.loads((FUND.parent/'scope.json').read_text())
    indexes={a['contract'] for a in scope if a['instrument_type']=='index'}
    for year in range(2021,2026):
        t=year+1
        expected_symbols=sorted(a['contract'] for a in scope if year in a['research_years'])
        data=pd.read_parquet(RETURNS/f'pool_{t}/daily_returns.parquet')
        data=data.loc[data.data_year==year].copy().sort_values(['entry_time_utc','symbol'])
        assert sorted(data.symbol.unique())==expected_symbols
        assert not data.duplicated(['symbol','entry_time_utc']).any()
        stamps=pd.to_datetime(data.entry_time_utc,utc=True)
        source_date=(stamps-pd.Timedelta(hours=LAG)).dt.floor('D').dt.tz_localize(None)-pd.Timedelta(days=1)
        data['factor_source_last_date_utc']=source_date
        data['factor_source_usable_not_before_utc']=(source_date+pd.Timedelta(days=1)).dt.tz_localize('UTC')
        assert (data.factor_source_usable_not_before_utc+pd.Timedelta(hours=LAG)<=stamps).all()
        for w in WINDOWS:
            panel=factors[w].reindex(index=source_date,columns=expected_symbols)
            positions=pd.Index(expected_symbols).get_indexer(data.symbol)
            values=panel.to_numpy()[np.arange(len(data)),positions]
            data[f'market_cap_mean_{w}d']=values
            # Independently verify one actual window against the raw source series.
            valid_indices=np.flatnonzero(np.isfinite(values))
            sample=valid_indices[len(valid_indices)//2]
            symbol=data.symbol.iloc[sample]
            end=source_date.iloc[sample]
            raw=cap.loc[end-pd.Timedelta(days=w-1):end,symbol]
            assert len(raw)==w and raw.notna().all()
            assert np.isclose(values[sample],raw.mean(),rtol=1e-12)
        names=[f'market_cap_mean_{w}d' for w in WINDOWS]
        data['all_four_factors_available']=np.isfinite(data[names]).all(axis=1)
        token_count=len(set(expected_symbols)-indexes)
        included_dates=[]
        for stamp,frame in data.groupby('entry_time_utc',sort=True):
            complete_returns=frame.return_24h.notna().all()
            eligible=frame.loc[frame.all_four_factors_available & frame.return_24h.notna()]
            included=bool(complete_returns and len(eligible)>=max(20,int(np.ceil(token_count*.90))))
            day=stamp.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d')
            coverage.append({'data_year':year,'pool_year':t,'decision_date_beijing':day,
                             'pool_members':len(frame),'token_members':token_count,
                             'valid_return_members':int(frame.return_24h.notna().sum()),
                             'common_factor_members':len(eligible),'included':included})
            if not included:
                continue
            included_dates.append(day)
            benchmark=float(frame.return_24h.mean())
            covered=float(eligible.return_24h.mean())
            returns=eligible.return_24h.to_numpy()
            for name in names:
                x=eligible[name].to_numpy()
                ic=corr(x,returns-benchmark)
                rank_ic=corr(pd.Series(x).rank().to_numpy(),pd.Series(returns).rank().to_numpy())
                order=np.argsort(x,kind='stable')
                recent=[]
                for g,indices in enumerate(np.array_split(order,10),1):
                    r=float(returns[indices].mean())
                    row={'factor':name,'data_year':year,'pool_year':t,'decision_date_beijing':day,
                         'decision_timestamp':int(stamp.timestamp()*1000),'exit_timestamp':int(stamp.timestamp()*1000)+DAY,
                         'decile':g,'members':len(indices),'factor_members':len(eligible),
                         'ic':ic,'rank_ic':rank_ic,'group_return':r,'benchmark_return':benchmark,
                         'covered_benchmark_return':covered,'excess_return':r-benchmark}
                    recent.append(row)
                assert np.isclose(sum(r['members']*r['group_return'] for r in recent)/len(eligible),covered,atol=1e-12)
                rows.extend(recent)
        data.to_parquet(OUT/f'aligned_dataset_{year}.parquet',index=False)
        manifest_years.append({'data_year':year,'pool_year':t,'pool_members':len(expected_symbols),
                               'valid_days':len(included_dates),'first_valid_date':included_dates[0],
                               'last_valid_date':included_dates[-1],
                               'excluded_dates':[r['decision_date_beijing'] for r in coverage if r['data_year']==year and not r['included']]})
        print(f'{year}: {len(included_dates)} valid days, first {included_dates[0]}',flush=True)
    daily=pd.DataFrame(rows).sort_values(['factor','decile','decision_timestamp'])
    daily['cumulative_excess_simple_return']=daily.groupby(['factor','decile']).excess_return.cumsum()
    assert not daily.duplicated(['factor','decile','decision_timestamp']).any()
    counts=daily.groupby(['factor','decile']).size()
    assert counts.nunique()==1
    tables=[]
    for factor,part in daily.groupby('factor'):
        tables.append(metrics(part,'2021-2025'))
        tables.extend(metrics(p,str(y)) for y,p in part.groupby('data_year'))
    table=pd.DataFrame(tables)
    daily.to_parquet(OUT/'decile_daily.parquet',index=False)
    table.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'date_coverage.csv',index=False)
    manifest={'formulas':{f'market_cap_mean_{w}d':f'arithmetic mean of {w} consecutive UTC daily circulating market caps in USD' for w in WINDOWS},
              'years':manifest_years,'source_additional_lag_hours':LAG,
              'IC':'daily cross-sectional Pearson correlation of arithmetic mean cap and next 24h excess return; raw units, not logarithm',
              'RankIC':'Spearman; average ranks for ties','ICIR':'daily mean / sample standard deviation; not annualized',
              'decision_time':'00:00 Asia/Shanghai','holding_hours':24,
              'sample':'same dates and asset subset for all four windows; complete 40d history generally sets eligibility; >=90% token pool coverage',
              'benchmark':'entire fixed pool equal weight, including index contracts and missing-factor names; require returns for every member',
              'grouping':'ascending arithmetic average cap, G1 smallest / G10 largest; ascending-symbol stable tie break; numpy.array_split',
              'costs':'gross before fees, slippage, funding','aggregation':'sum daily excess; not compounding; does not reset at year end',
              'missing':'no fill; full rolling windows; source identity discrepancy >1% excluded; provider id must remain constant over window',
              'fundamentals_sha256':hashlib.sha256(FUND.read_bytes()).hexdigest(),
              'returns_validation':str(RETURNS/'validation.json'),
              'validations':['factor availability +24h <= decision','sample raw arithmetic mean matches rolling factor',
                             'same asset/day sample across four factors','member-weighted deciles reconcile to covered universe'],
              'limitations':['retrospective year+1 pools use analyzed full year for selection',
                             'historical provider revisions and original publication vintage unverified',
                             'no 2020 cap history; 2021 initial warmup excluded',
                             'covered factor universe differs from full benchmark, so ten-decile excess need not average to zero']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    render(daily,table,manifest_years)
    bundle=OUT.parent/'平均市值10_20_30_40天_2021-2025.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():
                z.write(p,'average_market_cap_2021_2025/'+p.name)
    print(table.loc[table.period=='2021-2025'].to_string(index=False),flush=True)
    print('COMPLETE',flush=True)


def render(daily,table,years):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    colors=['#30123b','#4145ab','#4675ed','#39a2fc','#1bcfd4','#24eca6','#61fc6c','#b2f235','#faba39','#df2e16']
    style='body{font:15px system-ui;margin:28px auto;max-width:1500px;padding:0 22px;color:#203047}p{line-height:1.8}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}.tables{overflow:auto}section{margin:40px 0;scroll-margin-top:65px}nav{position:sticky;top:0;background:white;padding:12px;border-bottom:1px solid #ccc;z-index:100}a{margin-right:15px}input{padding:7px}'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>平均市值因子 2021—2025</title><style>'+style+'</style>'
    page+='<h1>10 / 20 / 30 / 40 天平均流通市值因子</h1><p>因子是最近 N 个 UTC 日终流通市值的算术平均，单位美元；不取对数、不翻转方向。G1 最小、G10 最大。北京时间每天 00:00 调仓，持有 24 小时，日终市值额外延迟 24 小时使用。IC 为原始平均市值的 Pearson 相关，RankIC 为 Spearman，两类 IR 均不年化。</p>'
    page+='<p>数据年 2021—2025，对应币池年 2022—2026。四因子使用完全相同的有效日期和币种；因子样本至少覆盖当年 90% 的代币成员。基准是固定全币池等权，含缺市值币和指数合约，全池任一币缺收益则整日剔除。收益是每日超额相加，不复利、不扣手续费、滑点及资金费。年间曲线连续累计；图中缺少收益的日期留空。</p>'
    page+='<p>没有 2020 年市值，因此统一受 40 天窗口预热约束；缺值不填充，窗口内供应商资产版本变化亦剔除。市值样本与全币池基准不同，十档超额平均不一定为零。币池有事后筛选偏差，历史市值可能修订，24 小时延迟不代表已验证当时的首次发布时间。</p>'
    page+='<nav><label>搜索定位 <input id="search" placeholder="10 / 20 / 30 / 40"></label><button onclick="locate()">定位</button> '+''.join(f'<a href="#factor-{w}">{w} 天</a>' for w in WINDOWS)+'</nav>'
    coverage=pd.DataFrame([{k:v for k,v in y.items() if k!='excluded_dates'} for y in years])
    page+='<h2>共同有效样本</h2>'+coverage.to_html(index=False)
    page+='<script>'+get_plotlyjs()+'</script>'
    full_stamps=np.arange(pd.Timestamp('2021-01-01',tz='Asia/Shanghai').timestamp()*1000,
                          pd.Timestamp('2026-01-01',tz='Asia/Shanghai').timestamp()*1000,DAY,dtype='int64')
    full_dates=pd.to_datetime(full_stamps+DAY+8*3600000,unit='ms')
    for w in WINDOWS:
        name=f'market_cap_mean_{w}d'
        part=daily.loc[daily.factor==name]
        overall=table.loc[(table.factor==name)&(table.period=='2021-2025')].iloc[0]
        fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
        chart=go.Figure()
        for g,color in enumerate(colors,1):
            p=part.loc[part.decile==g].set_index('decision_timestamp')
            values=p.cumulative_excess_simple_return.reindex(full_stamps).to_numpy()
            ax.plot(full_dates,values,color=color,label=f'G{g}',linewidth=1.3)
            chart.add_scatter(x=full_dates,y=values,name=f'G{g}',mode='lines',connectgaps=False,line={'color':color})
        title=f'{w}-day arithmetic mean circulating market cap | 2021-2025\nIC {overall.IC:+.4f} | ICIR {overall.ICIR:+.4f} | RankIC {overall.RankIC:+.4f} | G1 small / G10 large'
        ax.set_title(title)
        ax.set_ylabel('Cumulative excess simple return vs full-pool equal weight')
        ax.set_xlabel('Exit date, Beijing time | rebalance 00:00 | extra 24h cap delay | gross')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.axhline(0,color='gray',linewidth=.7)
        ax.grid(alpha=.2)
        ax.legend(ncol=10,fontsize=8)
        fig.savefig(OUT/f'{name}.png',dpi=145)
        plt.close(fig)
        chart.update_layout(height=620,title=f'{w} 天平均市值 · IC {overall.IC:+.4f} · ICIR {overall.ICIR:+.4f}',yaxis_title='累计超额单利收益',yaxis_tickformat='.0%',xaxis_title='收益兑现日期（北京时间）',legend_orientation='h',margin={'l':75,'r':25,'t':60,'b':60})
        page+=f'<section id="factor-{w}"><h2>{w} 天平均市值</h2>'+chart.to_html(full_html=False,include_plotlyjs=False)
        t=table.loc[table.factor==name].drop(columns='factor').copy()
        for col in [*[f'G{g}_excess_simple' for g in range(1,11)],'G1_minus_G10']:
            t[col]=t[col].map(lambda v:f'{v:.2%}')
        page+='<div class="tables">'+t.to_html(index=False,float_format=lambda v:f'{v:.5f}')+'</div></section>'
    page+='''<script>function locate(){const q=document.getElementById('search').value.trim();const w=['10','20','30','40'].find(x=>q.includes(x));if(w){document.getElementById('factor-'+w).scrollIntoView({behavior:'smooth'});history.replaceState(null,'','#factor-'+w)}}document.getElementById('search').addEventListener('keydown',e=>{if(e.key==='Enter')locate()});</script></html>'''
    (OUT/'平均市值因子_2021-2025.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    main()
