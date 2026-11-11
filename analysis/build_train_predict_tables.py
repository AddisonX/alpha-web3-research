"""Build ten-factor daily cross-sectional percentile train/predict tables. No training."""
from __future__ import annotations
import hashlib,io,json,zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RET=ROOT/'data/returns/annual_pools_2021_to_t'
CACHE=ROOT/'data/cache/binance_market_cap_4h'
OUT=ROOT/'data/model_tables/factors10_2021_2026'
FEATURE_ROOT=OUT/'raw_features_by_symbol'
BAR=14400000
DAY=86400000
FEATURES=['market_cap_mean_10d','market_cap_mean_20d','market_cap_mean_30d','market_cap_mean_40d',
          'mean_return_4','range_to_quote_volume_2','return_6','turnover_relative_volume_42','volume_change_12','volume_change_24']
PRICE_FEATURES=FEATURES[4:]


def sources():
    acquisition=json.loads((RET/'acquisition.json').read_text())
    assert not acquisition['errors']
    extra=json.loads((RET/'supplemental_acquisition.json').read_text())
    grouped={}
    for r in [*acquisition['sources'],*[r for r in extra if r['status']=='available']]:
        key=r['key'];native=key.split('/klines/')[1].split('/')[0]
        grouped.setdefault(native,{})[key]=r['sha256']
    return grouped


def symbol_features(native,keys):
    dest=FEATURE_ROOT/f'{native}.parquet'
    if dest.exists():return pd.read_parquet(dest)
    parts=[]
    # A daily supplement may only fill timestamps absent from monthly records.
    for key,sha in sorted(keys.items(),key=lambda kv:('/daily/' in kv[0],kv[0])):
        path=CACHE/key.rsplit('/',1)[-1]
        body=path.read_bytes();assert hashlib.sha256(body).hexdigest()==sha,key
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            names=[n for n in z.namelist() if n.endswith('.csv')];assert len(names)==1
            raw=z.read(names[0])
        skip=0 if raw.split(b',',1)[0].isdigit() else 1
        values=np.loadtxt(io.BytesIO(raw),delimiter=',',skiprows=skip,usecols=[0,1,2,3,4,5,7],ndmin=2)
        if values.size:
            values[:,0]=np.where(values[:,0]>1e14,values[:,0]/1000,values[:,0])
            assert np.isfinite(values).all()
            parts.append(values)
    raw=pd.DataFrame(np.concatenate(parts),columns=['bar_open_ms','open','high','low','close','volume','quote_volume_actual'])
    raw=raw.drop_duplicates('bar_open_ms',keep='first').sort_values('bar_open_ms')
    assert (raw.bar_open_ms%BAR==0).all()
    raw=raw.loc[(raw.volume>0)&(raw.quote_volume_actual>0)].copy()
    segments=raw.bar_open_ms.diff().ne(BAR).cumsum()
    output=[]
    for _,p in raw.groupby(segments,sort=False):
        p=p.copy()
        ret=p.close/p.close.shift(1)-1
        p['avg_range_4']=((p.high-p.low)/p.close).rolling(4,min_periods=2).mean()
        p['avg_range_3']=((p.high-p.low)/p.close).rolling(3,min_periods=2).mean()
        p['distance_to_low_6']=p.close/p.low.rolling(6,min_periods=2).min()-1
        p['distance_to_low_4']=p.close/p.low.rolling(4,min_periods=2).min()-1
        p['volatility_4']=ret.rolling(4,min_periods=2).std(ddof=1)
        p['mean_return_4']=ret.rolling(4,min_periods=2).mean()
        p['return_6']=p.close/p.close.shift(6)-1
        p['range_to_quote_volume_2']=((p.high-p.low)/p.close)/((p.close*p.volume).rolling(2,min_periods=2).mean()+1)
        p['volume_change_12']=p.volume/p.volume.shift(12)-1
        p['volume_change_24']=p.volume/p.volume.shift(24)-1
        p['turnover_relative_volume_42']=p.volume/p.volume.shift(1).rolling(42,min_periods=42).mean()
        # The original legacy-factor diagnostics only expose values after 25 closed bars.
        p.loc[p.index[:24],[n for n in PRICE_FEATURES if n!='turnover_relative_volume_42']]=np.nan
        decision=p.bar_open_ms+BAR
        ready=(decision+8*3600000)%DAY==0
        q=p.loc[ready,['bar_open_ms',*PRICE_FEATURES]].copy()
        q['decision_timestamp']=(q.bar_open_ms+BAR).astype('int64')
        q['factor_bar_close_utc']=pd.to_datetime(q.decision_timestamp,unit='ms',utc=True)
        q['factor_last_bar_open_utc']=pd.to_datetime(q.bar_open_ms,unit='ms',utc=True)
        q['symbol']=native[:-4]+'/USDT:USDT'
        output.append(q.drop(columns='bar_open_ms'))
    result=pd.concat(output,ignore_index=True).replace([np.inf,-np.inf],np.nan)
    result=result.loc[result.decision_timestamp>=int(pd.Timestamp('2021-01-01',tz='Asia/Shanghai').timestamp()*1000)]
    assert not result.duplicated(['symbol','decision_timestamp']).any()
    assert result.factor_bar_close_utc.le(pd.to_datetime(result.decision_timestamp,unit='ms',utc=True)).all()
    result.to_parquet(dest,index=False)
    return result


def cap_features():
    base=ROOT/'data/fundamentals/cmc_2021_2025/contract_daily.parquet'
    supplement=ROOT/'data/fundamentals/cmc_2026_supplement/contract_market_cap_2026.parquet'
    cols=['contract','date','cmc_id','market_cap_usd','marketcap_identity_error']
    f=pd.concat([pd.read_parquet(base,columns=cols),pd.read_parquet(supplement,columns=cols)],ignore_index=True)
    f['date']=pd.to_datetime(f.date)
    f.loc[(f.market_cap_usd<=0)|(f.marketcap_identity_error>.01),'market_cap_usd']=np.nan
    cap=f.pivot(index='date',columns='contract',values='market_cap_usd').sort_index()
    cap=cap.reindex(pd.date_range(cap.index.min(),cap.index.max()))
    ids=f.pivot(index='date',columns='contract',values='cmc_id').reindex_like(cap)
    result={}
    for w in [10,20,30,40]:
        stable=ids.rolling(w,min_periods=w).min().eq(ids.rolling(w,min_periods=w).max())
        result[f'market_cap_mean_{w}d']=cap.rolling(w,min_periods=w).mean().where(stable)
    return result


def make_table(t,split,raw_features,cap_panels):
    pool=t+1 if split=='train' else t
    data=pd.read_parquet(RET/f'pool_{pool}/daily_returns.parquet')
    data=data.loc[data.data_year.le(t)] if split=='train' else data.loc[data.data_year.eq(t)]
    data=data.copy().sort_values(['entry_time_utc','symbol']).reset_index(drop=True)
    data['decision_timestamp']=data.entry_time_utc.dt.as_unit('ms').astype('int64')
    data=data.merge(raw_features,on=['symbol','decision_timestamp'],how='left',validate='one_to_one')
    stamp=pd.to_datetime(data.entry_time_utc,utc=True)
    source_date=(stamp-pd.Timedelta(hours=24)).dt.floor('D').dt.tz_localize(None)-pd.Timedelta(days=1)
    data['market_cap_source_last_date_utc']=source_date
    data['market_cap_available_bound_utc']=(source_date+pd.Timedelta(days=1)).dt.tz_localize('UTC')+pd.Timedelta(hours=24)
    assert data.market_cap_available_bound_utc.le(stamp).all()
    for name,panel in cap_panels.items():
        ri=panel.index.get_indexer(source_date)
        ci=panel.columns.get_indexer(data.symbol)
        v=np.full(len(data),np.nan)
        good=(ri>=0)&(ci>=0)
        v[good]=panel.to_numpy()[ri[good],ci[good]]
        data[name]=v
    raw=data[['symbol','decision_timestamp',*FEATURES]].copy()
    # Percentile universe depends only on pool membership and each available factor.
    # Future-return validity must never determine a factor's rank denominator.
    daily=data.groupby('decision_timestamp',sort=False)
    rank_counts={}
    for name in FEATURES:
        data[name]=daily[name].rank(method='average',pct=True)
        rank_counts[name]=daily[name].count()
        assert data[name].dropna().between(0,1,inclusive='right').all()
    usable_price=data.price_return_24h.where(np.isfinite(data.price_return_24h))
    data['realized_return_24h']=usable_price
    daily=data.groupby('decision_timestamp',sort=False)
    data['benchmark_return_24h']=daily.realized_return_24h.transform('mean')
    data['benchmark_members']=daily.realized_return_24h.transform('count')
    data['pool_members']=daily.symbol.transform('size')
    data['benchmark_coverage']=data.benchmark_members/data.pool_members
    data['excess_return_24h']=data.realized_return_24h-data.benchmark_return_24h
    data['features_complete']=data[FEATURES].notna().all(axis=1)
    data['label_available']=data.realized_return_24h.notna()
    data['pool_available_at_utc']=pd.Timestamp(f'{pool}-01-01',tz='UTC')
    data['pool_known_at_decision']=data.entry_time_utc.ge(data.pool_available_at_utc)
    data['prediction_eligible_on_features']=data.features_complete & data.pool_known_at_decision
    data['label_price_time_utc']=data.exit_time_utc
    data.rename(columns={'return_24h':'strict_liquid_return_24h','return_known_not_before_utc':'strict_quality_known_not_before_utc',
                         'status':'return_data_status'},inplace=True)
    main_cols=['pool_year','data_year','decision_date_beijing','symbol','entry_time_utc','exit_time_utc',
               *FEATURES,'realized_return_24h','benchmark_return_24h','excess_return_24h',
               'benchmark_members','pool_members','benchmark_coverage','features_complete','label_available',
               'pool_available_at_utc','pool_known_at_decision','prediction_eligible_on_features',
               'factor_last_bar_open_utc','factor_bar_close_utc','market_cap_source_last_date_utc','market_cap_available_bound_utc',
               'label_price_time_utc','strict_quality_known_not_before_utc','entry_open','exit_open',
               'strict_liquid_return_24h','bars_present_7','has_complete_window','all_seven_bars_positive_volume','return_data_status']
    result=data[main_cols].copy()
    assert not result.duplicated(['symbol','entry_time_utc']).any()
    assert result.entry_time_utc.dt.tz_convert('Asia/Shanghai').dt.hour.eq(0).all()
    assert result.entry_time_utc.dt.tz_convert('Asia/Shanghai').dt.minute.eq(0).all()
    assert (result.exit_time_utc-result.entry_time_utc).eq(pd.Timedelta(hours=24)).all()
    assert np.allclose(result.loc[result.label_available,'realized_return_24h'],
                       (result.exit_open/result.entry_open-1).loc[result.label_available])
    sums=result.groupby('entry_time_utc').excess_return_24h.sum()
    assert sums.abs().max()<1e-10
    for name in FEATURES:
        independently=raw.groupby('decision_timestamp')[name].rank(method='average',pct=True)
        assert np.allclose(data[name],independently,equal_nan=True)
    name=f'{split}_table_{t}'
    destination=OUT/f'{split}_table_{"2021-2025" if split=="train" else "2022-2026"}'
    destination.mkdir(exist_ok=True)
    result.to_parquet(destination/f'{name}.parquet',index=False)
    result.to_csv(destination/f'{name}.csv.gz',index=False,compression={'method':'gzip','compresslevel':3})
    ranks=pd.DataFrame(rank_counts).add_suffix('_rank_members').reset_index()
    cover=result.groupby('entry_time_utc',sort=True).agg(decision_date_beijing=('decision_date_beijing','first'),
        pool_members=('pool_members','first'),benchmark_members=('benchmark_members','first'),
        benchmark_coverage=('benchmark_coverage','first'),complete_feature_members=('features_complete','sum')).reset_index()
    cover['decision_timestamp']=cover.entry_time_utc.dt.as_unit('ms').astype('int64')
    cover=cover.merge(ranks,on='decision_timestamp',validate='one_to_one')
    cover.to_csv(destination/f'{name}_coverage.csv',index=False)
    return {'table':name,'pool_year':pool,'members':result.symbol.nunique(),'rows':len(result),
            'first_date':result.decision_date_beijing.min(),'last_date':result.decision_date_beijing.max(),
            'complete_feature_rows':int(result.features_complete.sum()),'realized_return_rows':int(result.label_available.sum()),
            'complete_features_and_label_rows':int((result.features_complete&result.label_available).sum()),
            'last_date_with_realized_return':result.loc[result.label_available,'decision_date_beijing'].max(),
            'min_benchmark_coverage':float(result.benchmark_coverage.min()),
            'median_benchmark_coverage':float(result.benchmark_coverage.median()),
            'parquet':str((destination/f'{name}.parquet').relative_to(OUT))}


def main():
    FEATURE_ROOT.mkdir(parents=True,exist_ok=True)
    retained=pd.read_csv(ROOT/'reports/persistent_extreme_factors/retained_factors.csv').factor.tolist()
    assert set(retained)==set(FEATURES)
    keys=sources();frames=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        fs={ex.submit(symbol_features,n,k):n for n,k in sorted(keys.items())}
        for i,f in enumerate(as_completed(fs),1):
            frames.append(f.result())
            if i%40==0 or i==len(fs):print(f'Raw features {i}/{len(fs)}',flush=True)
    features=pd.concat(frames,ignore_index=True)
    assert features.symbol.nunique()==341
    assert not features.duplicated(['symbol','decision_timestamp']).any()
    features.to_parquet(OUT/'six_price_factors_raw.parquet',index=False)
    panels=cap_features()
    summaries=[]
    for split,years in [('train',range(2021,2026)),('predict',range(2022,2027))]:
        for year in years:
            r=make_table(year,split,features,panels);summaries.append(r)
            print(json.dumps(r,ensure_ascii=False),flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'table_summary.csv',index=False)
    (OUT/'feature_columns.json').write_text(json.dumps(FEATURES,indent=2))
    metadata={'tables':summaries,'features':FEATURES,'factor_values':'within-pool daily cross-sectional pandas rank(method=average,pct=True), interval (0,1]; no sign inversion; not fitted over entire time history',
              'train_definition':'table t uses pool t+1 for all dates 2021 through t',
              'predict_definition':'table t uses pool t for year t only; latest partial calendar year stops at cached return date grid',
              'benchmark':'user chose equal-weight mean of same-pool members with complete finite two-endpoint price returns on each day; includes zero-volume recorded prices with separate strict flags; coverage recorded',
              'labels':'realized_return_24h = next Beijing 00:00 open / current 00:00 open - 1; excess_return_24h subtracts contemporaneous benchmark; gross',
              'timing':'six market factors include last closed 4h candle at decision 00:00; signal ready at/after close, not a promise of zero-latency execution',
              'cap_timing':'UTC end-of-day caps plus 24h conservative delay; actual source-end age at Beijing midnight is 40h; complete 10/20/30/40 calendar-day arithmetic means',
              'legacy_formulas':'exact original formulas incl range_to_quote_volume_2 uses close*base_volume proxy and +1 denominator; zero-volume/gaps reset windows; 25 closed bars legacy warmup',
              'turnover_formula':'current base volume / mean of 42 prior 4h base volumes, excluding current bar; complete contiguous window',
              'missing':'no fill; retain all member-date rows, including prelisting, delisted and missing caps; percentile ranks independent of future label availability',
              'boundaries':'pool t available Jan1 00:00 UTC (=08:00 Beijing), so first Jan1 midnight is flagged false in pool_known_at_decision; training cohort historical rows are intentionally retrospective',
              'training_warning':'do not train on labels whose label_price_time_utc exceeds fit cutoff; strict volume-quality eligibility only known at strict_quality_known_not_before_utc; do not pass label/benchmark/quality flags to model as features',
              'model_input':'only ten names in feature_columns.json; prediction tables include realized labels for evaluation only; not a live forecast',
              'limitations':['factor shortlist used 2021-2025 results; not independent out-of-sample evaluation','CMC historical data may be revised; publication vintage not verified','gross zero-latency price labels exclude costs/funding/slippage'],
              'validation':['archive SHA256 verification','no symbol/date duplicates','factor sources <= decision','label interval exactly24h','recomputed raw price returns','percentile independently recomputed from raw values','daily excess sums to zero among benchmark members']}
    (OUT/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    (OUT/'README_数据说明.md').write_text('''# 十因子训练及预测表

train_table_t：固定第t+1年币池，保存2021年到t年的所有日记录。
predict_table_t：固定第t年币池，只保存t年的日记录。2026截至已缓存数据的最近日期。
不是模型训练或预测结果，本次没有训练XGBoost。

主表为parquet及相同内容的csv.gz；每行是一个池内币在北京时间零点的一条记录。
十个模型输入列列于feature_columns.json，全部已转为该池该日的百分位，范围(0,1]，
相同值取平均名次。每个因子按自身非空样本排名，不参考未来收益是否存在，不跨日期做排名。
所有因子保持原始方向，没有根据后验多空收益翻转符号。

六个行情因子包含零点刚收盘的4小时K线，因此完整信号只能在收盘确认后产生。
return_6、mean_return_4、volume_change_12/24等窗口单位仍为4小时K线根数，不是天数。
turnover_relative_volume_42：当前根成交量/之前42根平均成交量，分母不包含当前根。
range_to_quote_volume_2严格沿用原公式：当前振幅/（两根close*volume均值+1）。
正成交量条件只用于已经收盘的历史特征线；缺线或零成交后重新预热。
四个市值列是日终流通市值的10/20/30/40日算术平均，再按该池当日排名；
市值额外延迟24小时，零点时使用的最近市值日终实际早40小时，不使用未来市值。
窗口内缺值或供应商资产版本发生变化即为空；指数合约没有单币市值。

realized_return_24h：本日零点开盘价到次日零点开盘价的原始价格收益。
benchmark_return_24h：该池当天所有有完整两端价格收益的币等权平均（用户指定口径）。
excess_return_24h：该币真实价格收益减去上述基准。无收益的币不参与基准，不填零。
benchmark_members/pool_members/benchmark_coverage及独立coverage文件说明覆盖情况。
价格收益允许端点存在但成交量为零的报价，无法保证这些价格可交易；
strict_liquid_return_24h及七根线成交量、缺线标记单独保留，绝不据其删掉预测因子行。

所有收益、基准、未来成交量质量标记是监督标签或审计列，不得作为模型输入。
label_price_time_utc是平仓价格时点，不含网络/公布延迟；
strict_quality_known_not_before_utc需要等平仓那根4小时线收盘，才知道其完整成交量。
训练时必须按实际训练截止时点剔除尚未完成的标签。

币池t要求上一UTC年完整行情，最早于t年1月1日北京时间08:00确定。
predict_table_t仍保留1月1日零点记录，但pool_known_at_decision=false；
实际按此池交易应从1月2日零点开始。prediction_eligible_on_features只看当时特征及币池时点，
不依赖未来收益。train表是已知年度币池回溯历史的研究样本，历史pool_known_at_decision为false属于预期。

features_complete为十列全有效标记，未据此删行。上市前、下架后、未满窗口等仍保留空值。
市值始于2021，故2021年初市值特征需预热；2026最新一日可能有因子而无次日收益。
历史修订市值及事后选择的因子名单意味着这些表不自动构成严格样本外验证。
''',encoding='utf-8')
    bundle=OUT.parent/'十因子_train2021-2025_predict2022-2026.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and 'raw_features_by_symbol' not in p.parts and p.name!='six_price_factors_raw.parquet':
                z.write(p,p.relative_to(OUT.parent).as_posix())
    with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
    print('TABLES COMPLETE',flush=True)


if __name__=='__main__':main()
