"""Persist 2021..t daily forward returns for each already-selected annual pool."""
from __future__ import annotations

import hashlib
import io
import json
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/returns/annual_pools_2021_to_t'
CACHE = ROOT / 'data/cache/binance_market_cap_4h'
LIST_BASE = 'https://s3.ap-northeast-1.amazonaws.com/data.binance.vision'
ARCHIVE_BASE = 'https://data.binance.vision/'
NS = {'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
BAR = 14_400_000
DAY = 86_400_000
LOCAL = threading.local()
AS_OF = pd.Timestamp.now(tz='Asia/Shanghai')


def session():
    if not hasattr(LOCAL,'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.headers['User-Agent'] = 'AlphaWeb3Research/1.0'
    return LOCAL.session


def get(url, params=None):
    for attempt in range(4):
        try:
            response = session().get(url, params=params, timeout=(15,45))
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            if e.response is not None and e.response.status_code < 500 and e.response.status_code != 429:
                raise
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def listing(native):
    destination = OUT / 'archive_lists' / f'{native}.json'
    if destination.exists():
        cached = json.loads(destination.read_text())
        if cached['retrieved_at'][:10] == AS_OF.strftime('%Y-%m-%d'):
            return cached
    result = []
    for frequency in ['monthly','daily']:
        prefix = f'data/futures/um/{frequency}/klines/{native}/4h/'
        if frequency == 'daily':
            prefix += f'{native}-4h-{AS_OF:%Y-%m}'
        params = {'list-type':'2','prefix':prefix}
        while True:
            response = get(LIST_BASE, params=params)
            doc = ET.fromstring(response.content)
            result.extend({'key':node.findtext('s:Key',namespaces=NS),
                           'size':int(node.findtext('s:Size',namespaces=NS)),
                           'last_modified':node.findtext('s:LastModified',namespaces=NS)}
                          for node in doc.findall('s:Contents',NS)
                          if node.findtext('s:Key',namespaces=NS).endswith('.zip'))
            if doc.findtext('s:IsTruncated',namespaces=NS) != 'true':
                break
            params['continuation-token'] = doc.findtext('s:NextContinuationToken',namespaces=NS)
            assert params['continuation-token']
    record = {'native':native,'retrieved_at':AS_OF.isoformat(),'source':LIST_BASE,'objects':result}
    destination.write_text(json.dumps(record,indent=2))
    return record


def download(item, expected):
    key = item['key']
    path = CACHE / key.rsplit('/',1)[-1]
    reused = path.exists()
    if not reused:
        body = get(ARCHIVE_BASE + key).content
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            assert z.testzip() is None
        temp = path.with_suffix('.part')
        temp.write_bytes(body)
        temp.replace(path)
    body = path.read_bytes()
    if len(body) != item['size']:
        raise ValueError(f'archive size mismatch {key}')
    sha = hashlib.sha256(body).hexdigest()
    prior = expected.get(key)
    if prior and prior != sha:
        raise ValueError(f'archive disagrees with previous research SHA256 {key}')
    return {'key':key,'bytes':len(body),'sha256':sha,'reused_cache':reused,
            'matches_previous_research':True if prior else None,'retrieved_or_verified_at':AS_OF.isoformat()}


def parse(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith('.csv')]
        assert len(names) == 1
        raw = pd.read_csv(z.open(names[0]),header=None,usecols=range(9),dtype=str)
    raw.columns = ['stamp','open','high','low','close','volume','close_time','quote_volume','trades']
    first = pd.to_numeric(raw.stamp,errors='coerce')
    # Only the documented header may be removed; malformed numerical data fails.
    header = first.isna() & raw.stamp.isin(['open_time','open time','timestamp'])
    raw = raw.loc[~header].apply(pd.to_numeric,errors='raise')
    raw.loc[raw.stamp > 1e14,'stamp'] //= 1000
    raw.loc[raw.close_time > 1e14,'close_time'] //= 1000
    assert (raw.stamp % BAR == 0).all()
    assert np.isfinite(raw.to_numpy()).all()
    assert (raw[['open','high','low','close']] > 0).all().all()
    assert (raw.high >= raw[['open','close','low']].max(axis=1)).all()
    assert (raw.low <= raw[['open','close','high']].min(axis=1)).all()
    assert (raw[['volume','quote_volume','trades']] >= 0).all().all()
    assert (raw.close_time == raw.stamp + BAR - 1).all()
    return raw


def build(native, end_year, keys):
    symbol = native[:-4] + '/USDT:USDT'
    last_day = min(pd.Timestamp(f'{end_year}-12-31',tz='Asia/Shanghai'), AS_OF.normalize()-pd.Timedelta(days=1))
    dates = pd.date_range(pd.Timestamp('2021-01-01',tz='Asia/Shanghai'),last_day)
    stamps = dates.tz_convert('UTC').as_unit('ms').asi8
    parts, repair_dates = [], set()
    for key in keys:
        part = parse(CACHE / key.rsplit('/',1)[-1])
        parts.append(part)
        if '/monthly/' in key and len(part):
            expected_stamps = np.arange(int(part.stamp.min()),int(part.stamp.max())+BAR,BAR)
            absent = np.setdiff1d(expected_stamps,part.stamp.to_numpy())
            repair_dates.update(pd.to_datetime(absent,unit='ms',utc=True).strftime('%Y-%m-%d'))
    repairs = []
    for day in sorted(repair_dates):
        key = f'data/futures/um/daily/klines/{native}/4h/{native}-4h-{day}.zip'
        path = CACHE / key.rsplit('/',1)[-1]
        try:
            if not path.exists():
                body = get(ARCHIVE_BASE+key).content
                with zipfile.ZipFile(io.BytesIO(body)) as z:
                    assert z.testzip() is None
                path.write_bytes(body)
            part = parse(path)
            existing = set(np.concatenate([p.stamp.to_numpy() for p in parts]))
            extra = part.loc[~part.stamp.isin(existing)]
            parts.append(extra)
            repairs.append({'key':key,'status':'available','inserted_bars':len(extra),
                            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        except requests.HTTPError as e:
            if e.response.status_code != 404:
                raise
            repairs.append({'key':key,'status':'daily_archive_not_found','inserted_bars':0})
    (OUT/'gap_repairs'/f'{native}.json').write_text(json.dumps(repairs,indent=2))
    if not parts:
        raise ValueError(f'no source archives for {native}')
    bars = pd.concat(parts).sort_values('stamp')
    duplicate = bars.loc[bars.duplicated('stamp',keep=False)]
    if len(duplicate):
        assert duplicate.groupby('stamp').nunique().max().max() == 1, native
    bars = bars.drop_duplicates('stamp').set_index('stamp')
    first, last = int(bars.index.min()), int(bars.index.max())
    windows = bars[['open','volume','quote_volume']].reindex((stamps[:,None]+np.arange(7)[None,:]*BAR).ravel()).to_numpy().reshape(len(stamps),7,3)
    count = np.isfinite(windows[:,:,0]).sum(axis=1)
    entry, exit_price = windows[:,0,0], windows[:,-1,0]
    endpoint = np.isfinite(entry) & np.isfinite(exit_price)
    full = count == 7
    liquid = (windows[:,:,1:] > 0).all(axis=(1,2))
    completed_exit_bar = stamps + DAY + BAR <= int(AS_OF.timestamp() * 1000)
    valid = full & liquid & completed_exit_bar
    raw_return = np.where(endpoint, exit_price / entry - 1, np.nan)
    status = np.where(valid, 'available', np.where(~completed_exit_bar,'exit_bar_not_closed',
                        np.where(full & ~liquid,'zero_volume_in_window',
                        np.where(stamps + DAY < first,'before_first_archived_bar',
                        np.where(stamps > last,'after_last_archived_bar','missing_bar_or_archive')))))
    frame = pd.DataFrame({'symbol':symbol,'native':native,'data_year':dates.year,
                          'decision_date_beijing':dates.strftime('%Y-%m-%d'),
                          'entry_time_utc':pd.to_datetime(stamps,unit='ms',utc=True),
                          'exit_time_utc':pd.to_datetime(stamps+DAY,unit='ms',utc=True),
                          'return_known_not_before_utc':pd.to_datetime(stamps+DAY+BAR,unit='ms',utc=True),
                          'entry_open':entry,'exit_open':exit_price,
                          'price_return_24h':raw_return, 'return_24h':np.where(valid,raw_return,np.nan),
                          'bars_present_7':count, 'has_complete_window':full,
                          'all_seven_bars_positive_volume':full & liquid,'status':status})
    assert not frame.duplicated(['symbol','entry_time_utc']).any()
    assert np.isfinite(frame.loc[valid,'return_24h']).all()
    assert (frame.loc[valid,'return_24h'] > -1).all()
    assert np.allclose(frame.loc[valid,'return_24h'],frame.loc[valid,'exit_open']/frame.loc[valid,'entry_open']-1)
    frame.to_parquet(OUT / 'by_symbol' / f'{native}.parquet',index=False)
    return frame


def main():
    for p in [OUT, OUT/'archive_lists', OUT/'by_symbol', OUT/'gap_repairs', CACHE]:
        p.mkdir(parents=True,exist_ok=True)
    scope = json.loads((ROOT/'data/fundamentals/cmc_2021_2025/scope.json').read_text())
    pools = {t:sorted(a['base']+'USDT' for a in scope if t-1 in a['research_years']) for t in range(2022,2027)}
    assert [len(pools[t]) for t in pools] == [79,78,143,209,297]
    end_years = {native:max(t for t in pools if native in pools[t]) for native in set.union(*map(set,pools.values()))}
    expected = {}
    for year in range(2021,2026):
        acquisition = json.loads((ROOT/f'reports/annual_factor_diagnostics_2021_2025_midnight/{year}/acquisition.json').read_text())
        assert sorted(r['native'] for r in acquisition['records']) == pools[year+1]
        for r in acquisition['records']:
            for s in r['sources']:
                expected[s['key']] = s['sha256']
    records = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        fs = {ex.submit(listing,n):n for n in sorted(end_years)}
        for i,f in enumerate(as_completed(fs),1):
            records[fs[f]] = f.result()
            if i%40 == 0 or i==len(fs):
                print(f'Archive index {i}/{len(fs)}',flush=True)
    tasks, keys_by_native = [], {}
    for native,end_year in end_years.items():
        keys_by_native[native] = []
        upper = min(f'{end_year}-12',AS_OF.strftime('%Y-%m'))
        for item in records[native]['objects']:
            key = item['key']
            suffix = key.rsplit('/',1)[-1][len(native)+4:-4]  # YYYY-MM or YYYY-MM-DD
            if '/monthly/' in key:
                include = '2020-12' <= suffix <= upper and suffix < AS_OF.strftime('%Y-%m')
            else:
                include = suffix[:7] <= upper and suffix < AS_OF.strftime('%Y-%m-%d')
            if include:
                tasks.append(item)
                keys_by_native[native].append(key)
    reused = sum((CACHE/i['key'].rsplit('/',1)[-1]).exists() for i in tasks)
    print(f'Archives needed {len(tasks)}, cached {reused}, new {len(tasks)-reused}',flush=True)
    sources, errors = [], []
    with ThreadPoolExecutor(max_workers=20) as ex:
        fs = {ex.submit(download,item,expected):item['key'] for item in tasks}
        for i,f in enumerate(as_completed(fs),1):
            try:
                sources.append(f.result())
            except Exception as e:
                errors.append({'key':fs[f],'error':repr(e)})
            if i%400 == 0 or i==len(fs):
                print(f'Download/verify {i}/{len(fs)}, errors {len(errors)}',flush=True)
    (OUT/'acquisition.json').write_text(json.dumps({'as_of':AS_OF.isoformat(),'sources':sources,'errors':errors},indent=2))
    if errors:
        raise RuntimeError(str(errors[:5]))
    frames = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        fs = {ex.submit(build,n,end_years[n],sorted(keys_by_native[n])):n for n in sorted(end_years)}
        for i,f in enumerate(as_completed(fs),1):
            frames.append(f.result())
            if i%30 == 0 or i==len(fs):
                print(f'Return series {i}/{len(fs)}',flush=True)
    union = pd.concat(frames,ignore_index=True).sort_values(['symbol','entry_time_utc'])
    union.to_parquet(OUT/'all_symbols_daily_returns.parquet',index=False)
    repairs = [r for p in sorted((OUT/'gap_repairs').glob('*.json')) for r in json.loads(p.read_text())]
    (OUT/'supplemental_acquisition.json').write_text(json.dumps(repairs,indent=2))
    summary, checks, coverage = [], [], []
    for t,natives in pools.items():
        destination = OUT/f'pool_{t}'
        destination.mkdir(exist_ok=True)
        symbols = [n[:-4]+'/USDT:USDT' for n in natives]
        frame = union.loc[union.native.isin(natives) & (union.data_year <= t)].copy()
        frame.insert(0,'pool_year',t)
        frame.to_parquet(destination/'daily_returns.parquet',index=False)
        frame.to_csv(destination/'daily_returns.csv.gz',index=False,compression='gzip')
        wide = frame.pivot(index='decision_date_beijing',columns='symbol',values='return_24h').reindex(columns=symbols)
        wide.to_csv(destination/'returns_wide.csv',float_format='%.12g')
        pd.DataFrame({'symbol':symbols,'native':natives}).to_csv(destination/'members.csv',index=False)
        grouped = frame.groupby(['symbol','data_year'],sort=True)
        for (symbol,year),p in grouped:
            good = p.loc[p.status=='available']
            coverage.append({'pool_year':t,'symbol':symbol,'data_year':year,'expected_days':len(p),
                             'available_days':len(good),'coverage':len(good)/len(p),
                             'first_available_date':None if good.empty else good.decision_date_beijing.min(),
                             'last_available_date':None if good.empty else good.decision_date_beijing.max()})
        counts = frame.groupby(['data_year','status']).size().unstack(fill_value=0).reset_index()
        counts.to_csv(destination/'status_by_year.csv',index=False)
        summary.append({'pool_year':t,'members':len(natives),'start_date':frame.decision_date_beijing.min(),
                        'end_date_grid':frame.decision_date_beijing.max(),'rows':len(frame),
                        'available_returns':int(frame.return_24h.notna().sum()),
                        'last_available_date':frame.loc[frame.return_24h.notna(),'decision_date_beijing'].max()})
        # Compare against the actual benchmark used by the earlier factor reports.
        year=t-1
        old = pd.read_parquet(ROOT/f'reports/annual_factor_diagnostics_2021_2025_midnight/{year}/decile_daily.parquet',columns=['decision_timestamp','benchmark_return'])
        old = old.drop_duplicates('decision_timestamp').set_index('decision_timestamp').benchmark_return
        matching = frame.loc[frame.data_year==year].copy()
        matching['decision_timestamp'] = matching.entry_time_utc.dt.as_unit('ms').astype('int64')
        new = matching.groupby('decision_timestamp').return_24h.agg(['mean','count'])
        common = old.index.intersection(new.index[new['count']==len(natives)])
        err = float((old.loc[common]-new.loc[common,'mean']).abs().max())
        assert len(common) > 300 and err < 1e-10, (t,len(common),err)
        checks.append({'pool_year':t,'original_data_year':year,'comparison_days':len(common),'maximum_absolute_benchmark_difference':err})
    pd.DataFrame(coverage).to_csv(OUT/'coverage_by_pool_symbol_year.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'pool_summary.csv',index=False)
    (OUT/'validation.json').write_text(json.dumps({'prior_factor_benchmark_checks':checks,
        'unique_symbol_date':not union.duplicated(['symbol','entry_time_utc']).any(),
        'valid_return_formula_checked':True,'archive_SHA256_matches_previous_research':sum(s['matches_previous_research'] is True for s in sources)},indent=2))
    manifest = {'created_at':AS_OF.isoformat(),'frequency':'daily; 00:00 Asia/Shanghai; next-24h open-to-open',
                'source':ARCHIVE_BASE,'market':'Binance USD-M USDT perpetual futures',
                'union_contracts':len(end_years),'pools':summary,'archives':len(sources),'archives_reused':reused,
                'monthly_gap_daily_archives_found':sum(r['status']=='available' for r in repairs),
                'monthly_gap_bars_recovered':sum(r['inserted_bars'] for r in repairs),
                'strict_returns':'seven consecutive 4h bars including exit, all positive base/quote volume, exit bar closed',
                'raw_price_returns':'endpoint price ratios retained separately, even when zero-volume bars or intermediate gaps invalidate strict return',
                'missing_policy':'explicit null; no price carry-forward; no substitution by spot/new token symbol; delisted contracts retained',
                'pool_selection':'pool t uses t-1 full-year screening; files intentionally contain 2021..t, not only t-1',
                'costs':'gross; excludes fees, slippage, funding; no compounding',
                'availability':'return_known_not_before_utc conservatively requires exit 4h bar closed for volume eligibility; not original archive publication time'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (OUT/'数据说明.md').write_text('''# 年度币池日收益数据

2022/2023/2024/2025/2026 币池分别含 79/78/143/209/297 个合约；所有币池并集为 341 个。
每个 pool_t 保存该币池所有成员从 2021 年到第 t 年的日收益，2026 年截至本次可获得的完整区间。
日期是开仓日（北京时间），不是收益兑现日。每个合约在各币池中的历史行情完全共用。

return_24h = 次日北京时间 00:00 的 4 小时 K 线开盘价 / 当日北京时间 00:00 的开盘价 - 1。
数值为小数，例如 0.01 表示 1%。未扣手续费、滑点和资金费，不复利。
与此前因子回测一致，严格收益要求包含开仓、持有过程及平仓共 7 根连续 4 小时 K 线，
且每根成交量和成交额均为正。return_known_not_before_utc 为第 7 根线收盘后，
因为完整正成交量条件只有这时才能核验；它不是归档首次公开时间，也不能作为当日因子输入。

price_return_24h 另保留只看两端价格的收益，即使中间缺线或零成交导致严格收益为空。
研究时默认使用 return_24h；不要混用两个字段。entry_open/exit_open 保留开平价格，
bars_present_7 为实际存在的线数。合约倍数不影响同一合约价格比值。
不拼接改名后的新合约，避免单位重置产生假收益；仍需结合迁移公告理解个别异常。

上市前、下架后、零成交或行情缺失都保留日期并标记 status，不填零、不填充价格。
before_first_archived_bar/after_last_archived_bar 仅表示所取历史归档的边界，不是已核实的上市/下架日期。
missing_bar_or_archive 包含跨边界窗口及尚未公开的最新日归档，不一律解释为停牌或退市。
2026 年最后日期以 pool_summary.csv 为准，个别合约可能更早停止交易。

all_symbols_daily_returns.parquet 是去重的并集主表；by_symbol 按合约保存独立序列。
各 pool_t 目录含 daily_returns.parquet、daily_returns.csv.gz、returns_wide.csv、members.csv。
宽表行是北京时间开仓日、列是币种，值为严格 return_24h；空白为缺失。
coverage_by_pool_symbol_year.csv 提供逐币逐年覆盖率，status_by_year.csv 提供缺失原因计数。
validation.json 对照此前因子报告的全币池基准，检查同一天同一币池收益一致。
acquisition.json 保存源 URL 对应对象键、SHA256、是否复用缓存；历史来源清单在 archive_lists。
月度包内部缺线时另查官方日度包，仅补入缺少的 K 线，不覆盖月包已有数值；
补取记录见 supplemental_acquisition.json 和 gap_repairs，日包不存在则保留缺失。

币池是按 t-1 年筛选的固定名单，不代表 2021 年已经能预知该名单，
若用该币池研究更早年份，应把结果视为事后固定样本分析。
''',encoding='utf-8')
    bundle = OUT.parent/'年度币池_2021至各币池年份_日收益.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and 'archive_lists' not in p.parts:
                z.write(p,p.relative_to(OUT.parent).as_posix())
    print(json.dumps({'summary':summary,'validation':checks,'bundle':str(bundle)},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
