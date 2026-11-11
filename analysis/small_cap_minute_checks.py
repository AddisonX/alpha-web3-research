"""Targeted minute-volume checks, not historical order-book slippage estimates."""
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/small_cap_liquidity_100k_cny'
CACHE=OUT/'minute_archive_cache'
CACHE.mkdir(exist_ok=True)
orders=pd.read_parquet(OUT/'rebalance_orders.parquet')
holdings=pd.read_parquet(OUT/'g1_holdings_liquidity.parquet')
weak_orders=orders.sort_values('participation_actual4h',ascending=False).drop_duplicates(['symbol','stamp']).head(20)
weak_positions=holdings.sort_values('participation_actual4h',ascending=False).drop_duplicates(['symbol','stamp']).head(10)
selected=pd.concat([weak_orders[['symbol','stamp']],weak_positions[['symbol','stamp']]]).drop_duplicates()


def fetch(row):
    symbol,stamp=row
    native=symbol.split('/')[0]+'USDT'
    day=pd.Timestamp(stamp,unit='ms',tz='UTC').strftime('%Y-%m-%d')
    name=f'{native}-1m-{day}.zip'
    path=CACHE/name
    url=f'https://data.binance.vision/data/futures/um/daily/klines/{native}/1m/{name}'
    result={'symbol':symbol,'stamp':stamp,'date_beijing':(pd.Timestamp(stamp,unit='ms',tz='UTC')+pd.Timedelta(hours=8)).strftime('%Y-%m-%d'),'source_url':url}
    for attempt in range(3):
        try:
            if not path.exists():
                response=requests.get(url,timeout=(15,45))
                response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    assert z.testzip() is None
                path.write_bytes(response.content)
            break
        except requests.HTTPError as e:
            if e.response.status_code==404:
                return {**result,'status':'minute_archive_not_found'}
            if attempt==2:
                raise
            time.sleep(attempt+1)
        except requests.RequestException:
            if attempt==2:
                raise
            time.sleep(attempt+1)
    with zipfile.ZipFile(path) as z:
        raw=pd.read_csv(z.open(z.namelist()[0]),header=None,usecols=[0,1,2,3,4,5,7],dtype=str)
    raw.columns=['stamp','open','high','low','close','volume','quote_volume']
    raw=raw.loc[pd.to_numeric(raw.stamp,errors='coerce').notna()].apply(pd.to_numeric,errors='raise')
    raw.loc[raw.stamp>1e14,'stamp']//=1000
    raw=raw.set_index('stamp').sort_index()
    result.update({'status':'available','sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for minutes in [1,5,15,240]:
        part=raw.reindex(stamp+np.arange(minutes)*60000)
        result[f'quote_{minutes}min_usdt']=float(part.quote_volume.sum()) if part.quote_volume.notna().all() else None
    return result


results=[]
with ThreadPoolExecutor(max_workers=6) as ex:
    fs={ex.submit(fetch,row):row for row in selected.itertuples(index=False,name=None)}
    for f in as_completed(fs):
        results.append(f.result())
minute=pd.DataFrame(results)
minute.to_csv(OUT/'targeted_minute_volume.csv',index=False)
pd.Series(minute.status.value_counts()).to_csv(OUT/'minute_acquisition_status.csv')
keys=minute.loc[minute.status=='available',['symbol','stamp',*[f'quote_{m}min_usdt' for m in [1,5,15,240]]]]
checked=orders.merge(keys,on=['symbol','stamp'],how='inner',validate='many_to_one')
for m in [1,5,15]:
    checked[f'participation_actual_{m}min']=checked.trade_usdt/checked[f'quote_{m}min_usdt']
assert np.allclose(checked.quote_volume,checked.quote_240min_usdt,rtol=1e-7,atol=.02)
checked.sort_values('participation_actual_5min',ascending=False).to_csv(OUT/'targeted_minute_order_checks.csv',index=False)
position=holdings.merge(keys,on=['symbol','stamp'],how='inner',validate='many_to_one')
for m in [1,5,15]:
    position[f'full_position_participation_actual_{m}min']=position.target_usdt/position[f'quote_{m}min_usdt']
assert np.allclose(position.quote_volume,position.quote_240min_usdt,rtol=1e-7,atol=.02)
position.sort_values('full_position_participation_actual_5min',ascending=False).to_csv(OUT/'targeted_minute_full_position_checks.csv',index=False)
meta={'selection':'20 distinct symbol-date worst actual 4h order participation + 10 worst full-position symbol-dates; deduplicated',
      'requested_symbol_dates':len(selected),'available_symbol_dates':int(minute.status.eq('available').sum()),
      'missing_symbol_dates':int(minute.status.ne('available').sum()),'scope':'purposive worst-case sample; not representative of all orders',
      'validation':'sum of 240 minute quote volumes matches original 4h quote volume',
      'limitations':'realized turnover after decision; not order-book depth; cannot infer exact market-order slippage or average future short-interval volume'}
(OUT/'minute_method.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
print(checked.sort_values('participation_actual_5min',ascending=False).drop_duplicates(['symbol','stamp']).head(12)[['date_beijing','symbol','trade_usdt','quote_1min_usdt','quote_5min_usdt','quote_15min_usdt','participation_actual_1min','participation_actual_5min','participation_actual_15min']].to_string(index=False))
