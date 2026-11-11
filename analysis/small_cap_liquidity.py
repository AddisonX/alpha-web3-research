"""Historical execution-capacity screen of the four G1 size portfolios; not an LOB slippage model."""
from __future__ import annotations
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FACTORS=ROOT/'reports/average_market_cap_2021_2025'
RETURNS=ROOT/'data/returns/annual_pools_2021_to_t'
CACHE=ROOT/'data/cache/binance_market_cap_4h'
OUT=ROOT/'reports/small_cap_liquidity_100k_cny'
OUT.mkdir(parents=True,exist_ok=True)
CNY=100000
FX=7.0  # Explicit fixed scenario conversion, not a historical exchange-rate observation.
CAPITAL=CNY/FX
BAR=14400000
DAY=86400000


def market(native):
    dest=OUT/'market_cache'/f'{native}.parquet'
    if dest.exists():
        return pd.read_parquet(dest)
    keys=set()
    for listing in ['acquisition.json','supplemental_acquisition.json']:
        record=json.loads((RETURNS/listing).read_text())
        sources=record['sources'] if isinstance(record,dict) else record
        for item in sources:
            key=item['key']
            if f'/klines/{native}/4h/' in key and item.get('status','available')=='available':
                filename=key.rsplit('/',1)[-1]
                day=filename[len(native)+4:-4]
                if '2020-12' <= day <= '2025-12-31':
                    keys.add(key)
    frames=[]
    # Prefer monthly records; supplemental daily files only fill missing timestamps.
    for key in sorted(keys,key=lambda k:('/daily/' in k,k)):
        path=CACHE/key.rsplit('/',1)[-1]
        with zipfile.ZipFile(path) as z:
            raw=pd.read_csv(z.open(z.namelist()[0]),header=None,usecols=[0,1,2,3,4,5,7,8],dtype=str)
        raw.columns=['stamp','open','high','low','close','volume','quote_volume','trades']
        numeric=pd.to_numeric(raw.stamp,errors='coerce')
        raw=raw.loc[numeric.notna()].apply(pd.to_numeric,errors='raise')
        raw.loc[raw.stamp>1e14,'stamp']//=1000
        frames.append(raw)
    bars=pd.concat(frames).drop_duplicates('stamp',keep='first').sort_values('stamp').set_index('stamp')
    full_index=np.arange(int(bars.index.min()),int(bars.index.max())+BAR,BAR)
    bars=bars.reindex(full_index)
    bars.index.name='stamp'
    bars['prev4h_quote']=bars.quote_volume.shift(1)
    bars['prev24h_quote']=bars.quote_volume.shift(1).rolling(6,min_periods=6).sum()
    bars['prev7d_min4h_quote']=bars.quote_volume.shift(1).rolling(42,min_periods=42).min()
    bars['prev7d_median4h_quote']=bars.quote_volume.shift(1).rolling(42,min_periods=42).median()
    result=bars.loc[(bars.index+8*3600000)%DAY==0].reset_index()
    result.insert(0,'symbol',native[:-4]+'/USDT:USDT')
    result.to_parquet(dest,index=False)
    return result


def main():
    (OUT/'market_cache').mkdir(exist_ok=True)
    cover=pd.read_csv(FACTORS/'date_coverage.csv')
    included=set(cover.loc[cover.included,'decision_date_beijing'])
    holdings=[]
    for year in range(2021,2026):
        data=pd.read_parquet(FACTORS/f'aligned_dataset_{year}.parquet')
        data=data.loc[data.decision_date_beijing.isin(included)&data.all_four_factors_available&data.return_24h.notna()].sort_values(['entry_time_utc','symbol'])
        for stamp,frame in data.groupby('entry_time_utc',sort=True):
            n=int(np.ceil(len(frame)/10))
            for w in [10,20,30,40]:
                name=f'market_cap_mean_{w}d'
                chosen=frame.sort_values(name,kind='stable').iloc[:n]
                for row in chosen.itertuples():
                    holdings.append({'factor':name,'window':w,'year':year,'stamp':int(stamp.timestamp()*1000),
                                     'date_beijing':stamp.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d'),
                                     'symbol':row.symbol,'members':n,'target_usdt':CAPITAL/n,
                                     'return_24h':row.return_24h})
    held=pd.DataFrame(holdings)
    original=pd.read_parquet(FACTORS/'decile_daily.parquet')
    original=original.loc[original.decile==1]
    check=held.groupby(['factor','stamp']).agg(members=('symbol','size'),mean_return=('return_24h','mean')).join(
        original.rename(columns={'decision_timestamp':'stamp'}).set_index(['factor','stamp'])[['members','group_return']],rsuffix='_original')
    assert check.members.eq(check.members_original).all()
    assert np.allclose(check.mean_return,check.group_return,atol=1e-12)
    natives=sorted(s.split('/')[0]+'USDT' for s in held.symbol.unique())
    markets=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        fs={ex.submit(market,n):n for n in natives}
        for i,f in enumerate(as_completed(fs),1):
            markets.append(f.result())
            if i%20==0 or i==len(fs):
                print(f'Market data {i}/{len(fs)}',flush=True)
    prices=pd.concat(markets,ignore_index=True)
    lookup=prices.set_index(['symbol','stamp'])
    assert not lookup.index.duplicated().any()
    orders=[]
    for factor,group in held.groupby('factor',sort=True):
        previous={}
        dates=sorted(group.stamp.unique())
        for stamp in [*dates,dates[-1]+DAY]:
            targets=group.loc[group.stamp==stamp].set_index('symbol').target_usdt.to_dict()
            for symbol in sorted(set(targets)|set(previous)):
                market_row=lookup.loc[(symbol,stamp)]
                price=float(market_row.open)
                assert price>0 and np.isfinite(price),(symbol,stamp)
                old_usdt=previous.get(symbol,0)*price
                target=targets.get(symbol,0)
                delta=target-old_usdt
                if abs(delta)>1e-6:
                    orders.append({'factor':factor,'stamp':stamp,
                                   'date_beijing':(pd.Timestamp(stamp,unit='ms',tz='UTC')+pd.Timedelta(hours=8)).strftime('%Y-%m-%d'),
                                   'year':int((pd.Timestamp(stamp,unit='ms',tz='UTC')+pd.Timedelta(hours=8)).year),
                                   'symbol':symbol,'side':'buy' if delta>0 else 'sell',
                                   'trade_type':'initial_or_new_entry' if symbol not in previous else ('full_exit' if symbol not in targets else 'rebalance'),
                                   'trade_usdt':abs(delta),'target_usdt':target,'previous_marked_usdt':old_usdt})
            previous={s:target/float(lookup.loc[(s,stamp),'open']) for s,target in targets.items()}
        assert not previous
    orders=pd.DataFrame(orders)
    needed=['quote_volume','trades','prev4h_quote','prev24h_quote','prev7d_min4h_quote','prev7d_median4h_quote']
    orders=orders.merge(prices[['symbol','stamp',*needed]],on=['symbol','stamp'],validate='many_to_one',how='left')
    held=held.merge(prices[['symbol','stamp',*needed]],on=['symbol','stamp'],validate='many_to_one',how='left')
    for frame,amount in [(orders,'trade_usdt'),(held,'target_usdt')]:
        frame['participation_actual4h']=frame[amount]/frame.quote_volume
        frame['participation_prev4h']=frame[amount]/frame.prev4h_quote
        frame['participation_prev24h']=frame[amount]/frame.prev24h_quote
        frame['participation_5min_uniform_proxy']=frame.participation_actual4h*48
        frame['participation_15min_uniform_proxy']=frame.participation_actual4h*16
        frame['capacity_cny_at_1pct_of_actual4h']=frame.quote_volume*.01*FX
    assert held.quote_volume.gt(0).all()
    assert orders.quote_volume.gt(0).all()
    held.to_parquet(OUT/'g1_holdings_liquidity.parquet',index=False)
    orders.to_parquet(OUT/'rebalance_orders.parquet',index=False)
    summaries=[]
    for factor,part in held.groupby('factor'):
        for year in [*range(2021,2026),'2021-2025']:
            p=part if year=='2021-2025' else part.loc[part.year==year]
            o=orders.loc[orders.factor==factor]
            if year!='2021-2025':
                o=o.loc[o.year==year]
            row={'factor':factor,'period':str(year),'holding_days':p.stamp.nunique(),'unique_symbols':p.symbol.nunique(),
                 'members_min':p.members.min(),'members_max':p.members.max(),
                 'target_cny_min':p.target_usdt.min()*FX,'target_cny_max':p.target_usdt.max()*FX,
                 'median_4h_quote_usdt':p.quote_volume.median(),'p01_4h_quote_usdt':p.quote_volume.quantile(.01),
                 'min_4h_quote_usdt':p.quote_volume.min(),
                 'full_position_4h_participation_p99':p.participation_actual4h.quantile(.99),
                 'full_position_4h_participation_max':p.participation_actual4h.max(),
                 'full_position_4h_over_1pct_fraction':p.participation_actual4h.gt(.01).mean(),
                 'full_position_5min_uniform_over_1pct_fraction':p.participation_5min_uniform_proxy.gt(.01).mean(),
                 'full_position_5min_uniform_over_5pct_fraction':p.participation_5min_uniform_proxy.gt(.05).mean(),
                 'order_count':len(o),'median_order_usdt':o.trade_usdt.median(),'max_order_usdt':o.trade_usdt.max(),
                 'actual_order_4h_participation_p99':o.participation_actual4h.quantile(.99),
                 'actual_order_4h_participation_max':o.participation_actual4h.max(),
                 'actual_order_4h_over_1pct_count':int(o.participation_actual4h.gt(.01).sum()),
                 'actual_order_5min_uniform_over_1pct_count':int(o.participation_5min_uniform_proxy.gt(.01).sum()),
                 'actual_order_5min_uniform_over_5pct_count':int(o.participation_5min_uniform_proxy.gt(.05).sum()),
                 'total_traded_notional_over_capital':o.trade_usdt.sum()/CAPITAL}
            gross=original.loc[original.factor==factor]
            if year!='2021-2025':
                gross=gross.loc[gross.data_year==year]
            row['gross_g1_excess_simple']=gross.excess_return.sum()
            for bps in [5,10,20,50]:
                row[f'g1_excess_minus_{bps}bp_one_way_cost']=row['gross_g1_excess_simple']-row['total_traded_notional_over_capital']*bps/10000
            summaries.append(row)
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'summary.csv',index=False)
    ranked=orders.sort_values('participation_actual4h',ascending=False)
    ranked.head(200).to_csv(OUT/'worst_200_orders.csv',index=False)
    held.sort_values('participation_actual4h',ascending=False).head(200).to_csv(OUT/'worst_200_full_positions.csv',index=False)
    coin=held.groupby('symbol').agg(holding_observations=('stamp','size'),holding_dates=('stamp','nunique'),
                                 median_4h_quote_usdt=('quote_volume','median'),min_4h_quote_usdt=('quote_volume','min'),
                                 max_full_position_4h_participation=('participation_actual4h','max')).sort_values('max_full_position_4h_participation',ascending=False)
    coin.to_csv(OUT/'coin_liquidity.csv')
    meta={'capital_cny':CNY,'gross_exposure_multiple':1,'assumed_cny_per_usdt':FX,
          'fx_note':'fixed scenario conversion only; not measured current or historical FX',
          'portfolio':'entire capital equally allocated to G1; constant USDT notional, target shares rebalanced daily; gains withdrawn/losses topped up for fixed-size capacity comparison',
          'orders':'target notional less previous target shares marked at current entry open; exits and new entries included; terminal liquidation included',
          'liquidity':'Binance USD-M contract quote_volume in USDT at decision 4h bar; prior 4h/24h/7d fields use only closed bars',
          'horizon':'actual4h is realized future 4h turnover, ex-post diagnostic, not a valid contemporaneous filter; uniform 5/15min proxies are not actual short-interval liquidity',
          'missing':'G1 membership exactly matches previous report; no missing prices or zero execution-bar volumes in evaluated observations',
          'thresholds':'1%/5% are descriptive participation screens, not calibrated slippage guarantees',
          'cost_sensitivity':'constant one-way all-in bps assumptions, not measured slippage; costs charged on actual modeled traded notional; benchmark remains gross',
          'limitations':['no historical order-book spreads or depth; cannot measure instant-market-order slippage',
                         'constant notional scenario, not compounded account NAV','old pools are retrospective',
                         'year-end exits may use next-year liquidity; entire-period costs include terminal exit'],
          'validations':['G1 holdings counts and mean returns equal original four-factor report','all required open prices present','all execution 4h quote volumes positive']}
    (OUT/'method.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    page='<meta charset="utf-8"><title>小市值G1交易容量</title><style>body{font:14px system-ui;margin:25px;color:#203047}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ccc}p{max-width:1300px;line-height:1.8}.table{overflow:auto}</style><h1>平均市值因子 G1 · 10万元人民币 · 1倍名义仓位</h1>'
    page+='<p>固定 7 元人民币 = 1 USDT 仅作容量换算。全资金等权投入 G1，不复利；日常交易额计入价格漂移后的再平衡、换币和最终平仓。收益和持仓已对照原报告一致。</p><p>成交额取实际 Binance 合约 quote_volume；成交时段为北京时间 00:00—04:00。4小时成交额是事后容量统计，不是零点即可知道的数据，也不等于盘口深度。5/15分钟指标仅把4小时成交额均匀折算，不能代表实际短时成交，更不能据此确定滑点。精确滑点需要历史盘口和成交时序。</p><p>1%/5% 为参与率筛查线，无保证含义。成本表假设每笔单边综合成本5/10/20/50bp，未测量实际滑点；G1扣成本后仍对比全池毛收益基准。历史币池有事后筛选偏差。</p>'
    page+='<h2>各窗口整体容量与成本敏感性</h2><div class="table">'+summary.loc[summary.period=='2021-2025'].to_html(index=False,float_format=lambda x:f'{x:.6g}')+'</div><h2>成交参与率最高的订单</h2><div class="table">'+ranked.head(30).to_html(index=False,float_format=lambda x:f'{x:.6g}')+'</div><h2>逐币流动性</h2><div class="table">'+coin.to_html(float_format=lambda x:f'{x:.6g}')+'</div>'
    (OUT/'小市值G1流动性检查.html').write_text(page,encoding='utf-8')
    print(summary.loc[summary.period=='2021-2025'].to_json(orient='records',indent=2),flush=True)
    print('WORST ORDERS\n'+ranked.head(15).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
