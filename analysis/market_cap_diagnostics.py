"""Daily Beijing-midnight size-factor diagnostics with explicit availability alignment."""
from __future__ import annotations
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/market_cap_2021_2025'
CACHE = ROOT / 'data/cache/binance_market_cap_4h'
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
DAY = 86400000
OFFSET = 28800000


def archive(native, month):
    name = f'{native}-4h-{month}.zip'
    path = CACHE / name
    url = f'https://data.binance.vision/data/futures/um/monthly/klines/{native}/4h/{name}'
    if not path.exists():
        for attempt in range(4):
            try:
                content = urlopen(url, timeout=45).read()
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    assert z.testzip() is None
                path.write_bytes(content)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
    return {'name': name, 'url': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def labels(native, year):
    dest = CACHE / f'{native}-{year}-returns.parquet'
    if dest.exists():
        return pd.read_parquet(dest)
    months = [f'{year-1}-12', *[f'{year}-{m:02d}' for m in range(1,13)], f'{year+1}-01']
    parts = []
    for month in months:
        path = CACHE / f'{native}-4h-{month}.zip'
        with zipfile.ZipFile(path) as z:
            part = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0,1,5,7], dtype=str)
        part.columns = ['stamp', 'open', 'volume', 'quote_volume']
        part = part.apply(pd.to_numeric, errors='coerce').dropna()
        part.loc[part.stamp > 1e14, 'stamp'] //= 1000
        parts.append(part)
    bars = pd.concat(parts).drop_duplicates('stamp').sort_values('stamp').set_index('stamp')
    stamps = pd.date_range(f'{year}-01-01', f'{year}-12-31', tz='Asia/Shanghai').tz_convert('UTC').as_unit('ms').asi8
    windows = bars.reindex((stamps[:,None] + np.arange(7)[None,:] * 14400000).ravel()).to_numpy().reshape(len(stamps),7,3)
    good = np.isfinite(windows).all(axis=(1,2)) & (windows > 0).all(axis=(1,2))
    forward = np.where(good, windows[:,-1,0]/windows[:,0,0]-1, np.nan)
    frame = pd.DataFrame({'native':native, 'contract':native[:-4]+'/USDT:USDT',
                          'decision_timestamp':stamps, 'year':year, 'forward_return':forward})
    frame.to_parquet(dest, index=False)
    return frame


def stats(part, label):
    daily = part.drop_duplicates('decision_timestamp')
    def ratio(x):
        return float(x.mean()/x.std(ddof=1)) if x.std(ddof=1) > 0 else None
    return {'period': label, 'days': len(daily), 'IC': float(daily.ic.mean()), 'ICIR': ratio(daily.ic),
            'RankIC': float(daily.rank_ic.mean()), 'RankICIR': ratio(daily.rank_ic),
            'raw_cap_IC': float(daily.raw_ic.mean()),
            'G1_excess': float(part.loc[part.decile == 1, 'excess_return'].sum()),
            'G10_excess': float(part.loc[part.decile == 10, 'excess_return'].sum()),
            'small_minus_large': float(part.loc[part.decile == 1, 'excess_return'].sum() - part.loc[part.decile == 10, 'excess_return'].sum()),
            'mean_factor_members': float(daily.factor_members.mean())}


def main():
    pools = {}
    for year in range(2021,2026):
        acquisition = json.loads((ROOT / f'reports/annual_factor_diagnostics_2021_2025_midnight/{year}/acquisition.json').read_text())
        pools[year] = sorted(r['native'] for r in acquisition['records'])
        assert len(pools[year]) == {2021:79,2022:78,2023:143,2024:209,2025:297}[year]
    tasks = sorted({(native, month) for year, natives in pools.items() for native in natives
                    for month in [f'{year-1}-12', *[f'{year}-{m:02d}' for m in range(1,13)], f'{year+1}-01']})
    sources, errors = [], []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(archive, *task): task for task in tasks}
        for i, future in enumerate(as_completed(futures),1):
            try:
                sources.append(future.result())
            except Exception as error:
                errors.append({'task': futures[future], 'error': repr(error)})
            if i % 250 == 0 or i == len(tasks):
                print(f'Archives {i}/{len(tasks)}, errors {len(errors)}', flush=True)
    (OUT / 'sources.json').write_text(json.dumps({'sources': sources, 'errors': errors}))
    if errors:
        raise RuntimeError(str(errors[:5]))
    returns = []
    for year, natives in pools.items():
        with ThreadPoolExecutor(max_workers=8) as ex:
            returns.extend(ex.map(lambda n: labels(n,year), natives))
        print(f'Return labels complete {year}', flush=True)
    returns = pd.concat(returns, ignore_index=True)
    returns.to_parquet(OUT / 'return_labels.parquet', index=False)
    checks = []
    for year in pools:
        old = pd.read_parquet(ROOT / f'reports/annual_factor_diagnostics_2021_2025_midnight/{year}/decile_daily.parquet', columns=['decision_timestamp','benchmark_return']).drop_duplicates('decision_timestamp').set_index('decision_timestamp').benchmark_return
        new = returns.loc[returns.year == year].groupby('decision_timestamp').forward_return.agg(['mean','count'])
        common = old.index.intersection(new.index[new['count'] == len(pools[year])])
        error = float((old.loc[common] - new.loc[common,'mean']).abs().max())
        assert len(common) > 300 and error < 1e-10, (year,len(common),error)
        checks.append({'year':year,'overlap_days':len(common),'max_absolute_benchmark_difference':error})
    (OUT / 'benchmark_validation.json').write_text(json.dumps(checks,indent=2))
    funds = pd.read_parquet(ROOT / 'data/fundamentals/cmc_2021_2025/contract_daily.parquet')
    cap_lookup = funds.set_index(['contract', pd.to_datetime(funds.date).dt.strftime('%Y-%m-%d')])
    outputs, metrics, coverage, datasets = [], [], [], []
    for lag in [24, 0]:
        data = returns.copy()
        stamps = pd.to_datetime(data.decision_timestamp, unit='ms', utc=True)
        # An exact observation date, not a stale forward fill. EOD + lag <= decision.
        source_days = (stamps - pd.Timedelta(hours=lag)).dt.floor('D') - pd.Timedelta(days=1)
        keys = pd.MultiIndex.from_arrays([data.contract, source_days.dt.strftime('%Y-%m-%d')])
        selected = cap_lookup.reindex(keys).reset_index(drop=True)
        cap = selected.market_cap_usd
        cap = cap.where(selected.marketcap_identity_error.fillna(0) <= .01)
        data['market_cap_usd'] = cap.to_numpy()
        data['source_date_utc'] = source_days.dt.strftime('%Y-%m-%d')
        data['source_usable_not_before_utc'] = selected.usable_not_before_utc.to_numpy()
        data['lag_hours'] = lag
        use_times = pd.to_datetime(data.source_usable_not_before_utc, utc=True)
        assert ((use_times + pd.Timedelta(hours=lag) <= stamps) | use_times.isna()).all()
        assert not data.duplicated(['contract','decision_timestamp']).any()
        factor_rows = []
        for (year, stamp), frame in data.groupby(['year','decision_timestamp'], sort=True):
            complete_returns = frame.forward_return.notna().all()
            eligible = frame.loc[frame.market_cap_usd.gt(0) & frame.forward_return.notna()].sort_values('contract')
            expected_tokens = sum(n[:-4] not in {'BLUEBIRD','BTCDOM','DEFI','FOOTBALL'} for n in pools[year])
            included = complete_returns and len(eligible) >= max(20, int(np.ceil(expected_tokens * .90)))
            coverage.append({'lag_hours': lag, 'year': year, 'decision_timestamp': stamp,
                             'beijing_date': (pd.Timestamp(stamp, unit='ms', tz='UTC') + pd.Timedelta(hours=8)).strftime('%Y-%m-%d'),
                             'pool_members': len(frame), 'return_members': int(frame.forward_return.notna().sum()),
                             'factor_members': len(eligible), 'included': included})
            if not included:
                continue
            benchmark = frame.forward_return.mean()  # Includes index and missing-cap members.
            x = np.log(eligible.market_cap_usd)
            y = eligible.forward_return - benchmark
            ic, rank_ic = float(x.corr(y)), float(x.rank().corr(y.rank()))
            raw_ic = float(eligible.market_cap_usd.corr(y))
            order = np.argsort(x.to_numpy(), kind='stable')
            covered_return = float(eligible.forward_return.mean())
            for decile, indices in enumerate(np.array_split(order,10),1):
                r = float(eligible.iloc[indices].forward_return.mean())
                factor_rows.append({'lag_hours':lag, 'year':year, 'decision_timestamp':int(stamp),
                                    'decile':decile, 'members':len(indices), 'factor_members':len(eligible),
                                    'ic':ic, 'rank_ic':rank_ic, 'raw_ic':raw_ic,
                                    'group_return':r, 'benchmark_return':float(benchmark),
                                    'covered_benchmark_return':covered_return, 'excess_return':r-benchmark})
            recent = factor_rows[-10:]
            assert np.isclose(sum(r['group_return']*r['members'] for r in recent)/len(eligible), covered_return)
        daily = pd.DataFrame(factor_rows)
        assert len(daily) > 0
        daily['cumulative_excess_simple_return'] = daily.groupby('decile').excess_return.cumsum()
        outputs.append(daily)
        datasets.append(data)
        for year, part in daily.groupby('year'):
            metrics.append({'lag_hours':lag, **stats(part,str(year))})
        metrics.append({'lag_hours':lag, **stats(daily,'2021-2025')})
    daily = pd.concat(outputs)
    metrics = pd.DataFrame(metrics)
    daily.to_parquet(OUT / 'decile_daily.parquet', index=False)
    pd.concat(datasets).to_parquet(OUT / 'aligned_dataset.parquet', index=False)
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT / 'date_coverage.csv', index=False)
    render(daily, metrics)
    (OUT / 'manifest.json').write_text(json.dumps({
        'factor':'ln(circulating market cap USD)', 'main_additional_lag_hours':24, 'sensitivity_additional_lag_hours':0,
        'pool_sizes':{str(y):len(v) for y,v in pools.items()}, 'decision':'00:00 Asia/Shanghai',
        'returns':'24h open-to-open; require seven positive-volume exact-clock 4h bars including exit',
        'benchmark':'all fixed pool members equal weight, including index and missing-cap names; exclude day if any return missing',
        'factor_universe':'positive reported market cap, identity discrepancy <=1%; at least 90% token pool coverage each day',
        'missing':'exact source dates only; no fill; no 2020 fundamentals',
        'grouping':'G1 smallest cap, G10 largest, stable contract tie-break; no sign flipping',
        'ICIR':'daily mean / sample std; not annualized', 'return_aggregation':'daily excess summed; gross before trading/funding costs',
        'limitations':['retrospective pools use analyzed full year; survivorship/lookahead pool selection',
                       'provider historical revisions; 24h delay is an assumption, not original publication verification',
                       'partial factor coverage can shift all groups relative to full-pool benchmark'],
        'validations':['unique contract/date','factor availability + assumed delay <= decision',
                       'member-weighted decile returns reconcile to covered universe'],
    }, ensure_ascii=False, indent=2))
    print(metrics.to_string(index=False), flush=True)


def render(daily, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    main = daily.loc[daily.lag_hours == 24]
    fig, ax = plt.subplots(figsize=(13,6), layout='constrained')
    interactive = go.Figure()
    for d, part in main.groupby('decile'):
        part = part.sort_values('decision_timestamp')
        dates = pd.to_datetime(part.decision_timestamp + DAY + OFFSET, unit='ms')
        ax.plot(dates, part.cumulative_excess_simple_return, label=f'G{d}', linewidth=1.3)
        interactive.add_scatter(x=dates, y=part.cumulative_excess_simple_return, name=f'G{d}', mode='lines')
    ax.set_title('Log circulating market cap | 2021-2025 | G1 small / G10 large\nBeijing 00:00 daily | extra 24h lag | full-pool equal-weight benchmark | gross')
    ax.set_ylabel('Cumulative excess simple return')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    ax.legend(ncol=10, fontsize=8)
    fig.savefig(OUT / 'market_cap_2021_2025.png', dpi=150)
    plt.close(fig)
    interactive.update_layout(height=650, yaxis_tickformat='.0%', title='流通市值对数 · 2021—2025 整体十档超额单利收益', xaxis_title='收益兑现日期（北京时间）', yaxis_title='累计超额单利收益', legend_orientation='h')
    notes = '''<h1>市值因子 · 2021—2025</h1><p>因子 = ln(美元流通市值)，G1 最小市值，G10 最大市值，不调整方向。北京时间每天 00:00 调仓，持有 24 小时。
    主结果使用 UTC 日终市值，并额外延迟 24 小时，另列不额外延迟的敏感性结果。两种口径都不使用调仓后的日终市值。</p>
    <p>超额基准为当年固定全币池等权，包括缺市值的币和指数合约。分组仅纳入有有效市值的币；每日代币覆盖须≥90%，全币池任一成员缺收益则整日剔除。
    每档超额逐日相加、不复利、不扣手续费、滑点或资金费。IC 使用市值对数的 Pearson 相关，RankIC 使用 Spearman，IR 不年化。raw_cap_IC 为原始市值 Pearson 相关，分组不变。</p>
    <p>注意：币池使用所分析全年信息筛选，存在事后筛选偏差；供应商可能修订历史。额外延迟不能证明已消除历史发布时点偏差。
    缺市值不填补；流通量单位复核不影响本次市值美元单位，但不同供应商资产版本仍有映射风险。
    缺市值币不进入十档，却保留在全币池基准，所以十档超额均值未必为零。</p>'''
    tables = []
    for lag in [24,0]:
        table = metrics.loc[metrics.lag_hours == lag].drop(columns='lag_hours').copy()
        for col in ['G1_excess','G10_excess','small_minus_large']:
            table[col] = table[col].map(lambda v:f'{v:.2%}')
        tables.append(f'<h2>额外延迟 {lag} 小时</h2>' + table.to_html(index=False, float_format=lambda x:f'{x:.4f}'))
    page = '<!doctype html><meta charset="utf-8"><title>市值因子回看诊断</title><style>body{font:15px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#19283b}p{line-height:1.8}table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}</style>'
    page += notes + '<script>' + get_plotlyjs() + '</script>' + interactive.to_html(full_html=False,include_plotlyjs=False) + ''.join(tables)
    (OUT / '市值因子_2021-2025.html').write_text(page,encoding='utf-8')


if __name__ == '__main__':
    main()
