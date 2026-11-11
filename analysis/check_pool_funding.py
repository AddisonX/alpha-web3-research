import sys
sys.path.append('/workspace/alpha-web3-research/.venv/lib/python3.12/site-packages')
import requests,json,time
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
OUT=Path('/tmp/pool2026_funding');OUT.mkdir(exist_ok=True)
base='https://fapi.binance.com'
def get(path,params=None):
    r=requests.get(base+path,params=params,timeout=30);r.raise_for_status();return r.json()
now=get('/fapi/v1/time')['serverTime'];start=now-30*86400000
prem=get('/fapi/v1/premiumIndex');info=get('/fapi/v1/fundingInfo');exchange=get('/fapi/v1/exchangeInfo')
current={x['symbol']:x for x in prem};interval={x['symbol']:x['fundingIntervalHours'] for x in info}
status={x['symbol']:x['status'] for x in exchange['symbols']}
members=pd.read_csv('/tmp/pool2026_members.csv').native.tolist()
rows=[];history=[]
def fetch(symbol):
    item=current.get(symbol);r=dict(symbol=symbol,status=status.get(symbol,'absent'),available=item is not None and status.get(symbol)=='TRADING')
    if not r['available']:return r,[]
    h=get('/fapi/v1/fundingRate',dict(symbol=symbol,startTime=start,endTime=now,limit=1000))
    assert len(h)<1000,'history needs pagination'
    rates=[float(x['fundingRate']) for x in h]
    hours=interval.get(symbol,8)
    r.update(latest_rate=float(item['lastFundingRate']),interval_hours=hours,
             current_daily_equivalent=float(item['lastFundingRate'])*24/hours,
             observations_30d=len(h),mean_daily_net_30d=sum(rates)/30,
             mean_daily_absolute_30d=sum(abs(v) for v in rates)/30,
             min_settlement_30d=min(rates) if rates else None,max_settlement_30d=max(rates) if rates else None,
             first_observation=h[0]['fundingTime'] if h else None,next_funding_time=item['nextFundingTime'])
    return r,h
with ThreadPoolExecutor(max_workers=6) as ex:
    fs={ex.submit(fetch,s):s for s in members}
    for i,f in enumerate(as_completed(fs),1):
        try:r,h=f.result();rows.append(r);history.extend(h)
        except Exception as e:rows.append(dict(symbol=fs[f],available=False,status='fetch_error',error=repr(e)))
        if i%75==0:print('Fetched',i,flush=True)
df=pd.DataFrame(rows).sort_values('symbol');df.to_csv(OUT/'funding_by_coin.csv',index=False)
pd.DataFrame(history).to_parquet(OUT/'settlements_30d.parquet',index=False)
(OUT/'snapshot.json').write_text(json.dumps(dict(server_time=now,beijing_time=pd.Timestamp(now,unit='ms',tz='UTC').tz_convert('Asia/Shanghai').isoformat(),start_time=start,premiumIndex=prem,fundingInfo=info),indent=2))
a=df[df.available]
print('TIME',pd.Timestamp(now,unit='ms',tz='UTC').tz_convert('Asia/Shanghai'))
print('COVERAGE',len(df),len(a),df.status.value_counts().to_dict())
print('INTERVAL',a.interval_hours.value_counts().to_dict())
print('SIGNS',int((a.latest_rate>0).sum()),int((a.latest_rate<0).sum()),int((a.latest_rate==0).sum()))
print('QUANTILES',a[['latest_rate','current_daily_equivalent','mean_daily_net_30d','mean_daily_absolute_30d']].quantile([0,.1,.25,.5,.75,.9,1]).to_string())
cols=['symbol','latest_rate','interval_hours','current_daily_equivalent','mean_daily_net_30d','mean_daily_absolute_30d']
print('POSITIVE',a.nlargest(5,'current_daily_equivalent')[cols].to_string(index=False))
print('NEGATIVE',a.nsmallest(5,'current_daily_equivalent')[cols].to_string(index=False))
print('MAJORS',a[a.symbol.isin(['BTCUSDT','ETHUSDT','SOLUSDT','DOGEUSDT','XRPUSDT'])][cols].to_string(index=False))
