"""Five fixed-parameter CUDA XGBoost fits, strictly train_table_(t-1) -> predict_table_t."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

PARAMS={'objective':'reg:squarederror','tree_method':'hist','device':'cuda:0',
        'max_depth':4,'eta':.05,'subsample':1.,'colsample_bytree':1.,
        'min_child_weight':1.,'lambda':1.,'alpha':0.,'max_bin':256,
        'seed':42,'nthread':4,'eval_metric':'rmse'}
ROUNDS=200


def correlation(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.ptp(x)>0 and np.ptp(y)>0 else None


def summarize(daily,year):
    days=daily.drop_duplicates('decision_timestamp')
    row={'prediction_year':year,'valid_days':len(days),
         'mean_predicted_members':float(days.predicted_members.mean()),
         'mean_benchmark_coverage':float(days.benchmark_coverage.mean())}
    for col,name in [('ic','IC'),('rank_ic','RankIC'),('pearson_ic','Pearson_IC')]:
        vals=days[col].dropna(); std=vals.std(ddof=1)
        row[name]=float(vals.mean());row[name+'IR']=float(vals.mean()/std) if std>0 else None
    for g in range(1,11):
        row[f'G{g}_excess_simple']=float(daily.loc[daily.decile==g,'excess_return'].sum())
    row['G10_minus_G1']=row['G10_excess_simple']-row['G1_excess_simple']
    return row


def run(root):
    base=root/'data/model_tables/factors22_2021_2026'
    out=Path('/tmp/xgb15_gpu_expanding_200_depth4_lr005_2022_2026');out.mkdir(parents=True,exist_ok=True)
    features=json.loads((base/'feature_columns.json').read_text())[:15]
    assert len(features)==15 and len(set(features))==15
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
    assert xgb.build_info().get('USE_CUDA') is True,'XGBoost was built without CUDA'
    print('GPU',gpu,'XGBOOST',xgb.__version__,flush=True)
    all_rows,cover,metrics,model_records,importance=[],[],[],[],[]
    for year in range(2022,2027):
        year_dir=out/str(year);year_dir.mkdir(exist_ok=True)
        train_path=base/f'train_table_2021-2025/train_table_{year-1}.parquet'
        predict_path=base/f'predict_table_2022-2026/predict_table_{year}.parquet'
        train=pd.read_parquet(train_path); test=pd.read_parquet(predict_path)
        test=test.loc[test.decision_date_beijing.le('2026-10-27')].copy()
        assert train.pool_year.eq(year).all() and test.pool_year.eq(year).all()
        assert train.data_year.lt(year).all() and test.data_year.eq(year).all()
        fit_asof=pd.Timestamp(f'{year}-01-01',tz='UTC')
        fit_mask=np.isfinite(train[features]).all(axis=1)&np.isfinite(train.excess_return_24h)&train.label_price_time_utc.le(fit_asof)
        fitting=train.loc[fit_mask].copy()
        assert len(fitting)>1000
        assert fitting.factor_bar_close_utc.le(fitting.entry_time_utc).all()
        assert fitting.market_cap_available_bound_utc.le(fitting.entry_time_utc).all()
        assert fitting.label_price_time_utc.max()<=fit_asof
        # Never select inference rows using a realized outcome or future quality flag.
        predict_mask=np.isfinite(test[features]).all(axis=1)&test.pool_known_at_decision
        forecast=test.loc[predict_mask].sort_values(['entry_time_utc','symbol']).copy()
        assert forecast.entry_time_utc.min()>fit_asof
        dtrain=xgb.DMatrix(fitting[features],label=fitting.excess_return_24h,feature_names=features)
        evals={};before=time.perf_counter()
        booster=xgb.train(PARAMS,dtrain,num_boost_round=ROUNDS,evals=[(dtrain,'train')],evals_result=evals,verbose_eval=False)
        fit_seconds=time.perf_counter()-before
        config=json.loads(booster.save_config())
        actual_device=config['learner']['generic_param']['device']
        assert actual_device.startswith('cuda'),f'CPU fallback: {actual_device}'
        assert 'grow_gpu_hist' in json.dumps(config),'GPU histogram updater not active'
        assert booster.num_boosted_rounds()==200
        assert int(config['learner']['gradient_booster']['tree_train_param']['max_depth'])==4
        booster.save_model(year_dir/'model.ubj')
        (year_dir/'booster_config.json').write_text(json.dumps(config,indent=2))
        (year_dir/'training_rmse.json').write_text(json.dumps(evals))
        before=time.perf_counter()
        forecast['predicted_excess_return']=booster.predict(xgb.DMatrix(forecast[features],feature_names=features))
        prediction_seconds=time.perf_counter()-before
        assert np.isfinite(forecast.predicted_excess_return).all()
        forecast['prediction_decile']=0
        yearly=[]
        for stamp,frame in forecast.groupby('entry_time_utc',sort=True):
            order=np.argsort(frame.predicted_excess_return.to_numpy(),kind='stable')
            groups=np.array_split(order,10)
            # Assign deciles to every forecast before inspecting outcomes.
            for g,indices in enumerate(groups,1):
                forecast.loc[frame.index[indices],'prediction_decile']=g
            enough=len(frame)>=20
            labels_complete=frame.realized_return_24h.notna().all() and frame.excess_return_24h.notna().all()
            included=bool(enough and labels_complete)
            day=stamp.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d')
            cover.append({'prediction_year':year,'date_beijing':day,'decision_timestamp':int(stamp.timestamp()*1000),
                          'predicted_members':len(frame),'label_members':int(frame.realized_return_24h.notna().sum()),
                          'benchmark_members':int(frame.benchmark_members.iloc[0]),'pool_members':int(frame.pool_members.iloc[0]),
                          'benchmark_coverage':float(frame.benchmark_coverage.iloc[0]),'included':included,
                          'exclusion_reason':'' if included else ('too_few_forecasts' if not enough else 'some_predicted_members_missing_realized_returns')})
            if not included:continue
            assert frame.benchmark_return_24h.nunique()==1
            pred=frame.predicted_excess_return.to_numpy();excess=frame.excess_return_24h.to_numpy()
            ic=correlation(pred,excess)
            rank_ic=correlation(pd.Series(pred).rank().to_numpy(),pd.Series(excess).rank().to_numpy())
            benchmark=float(frame.benchmark_return_24h.iloc[0]); returns=frame.realized_return_24h.to_numpy()
            recent=[]
            for g,indices in enumerate(groups,1):
                gr=float(returns[indices].mean())
                r={'prediction_year':year,'date_beijing':day,'decision_timestamp':int(stamp.timestamp()*1000),
                   'exit_timestamp':int(stamp.timestamp()*1000)+86400000,'decile':g,'members':len(indices),
                   'predicted_members':len(frame),'ic':rank_ic,'rank_ic':rank_ic,'pearson_ic':ic,
                   'group_return':gr,'benchmark_return':benchmark,'excess_return':gr-benchmark,
                   'benchmark_coverage':float(frame.benchmark_coverage.iloc[0])}
                recent.append(r)
            assert np.isclose(sum(r['group_return']*r['members'] for r in recent)/len(frame),returns.mean(),atol=1e-12)
            yearly.extend(recent)
        observed_dates={r['decision_timestamp'] for r in cover if r['prediction_year']==year}
        for stamp,frame in test.groupby('entry_time_utc',sort=True):
            stamp_ms=int(stamp.timestamp()*1000)
            if stamp_ms in observed_dates:continue
            cover.append({'prediction_year':year,'date_beijing':stamp.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d'),
                          'decision_timestamp':stamp_ms,'predicted_members':0,'label_members':0,
                          'benchmark_members':int(frame.benchmark_members.iloc[0]),'pool_members':int(frame.pool_members.iloc[0]),
                          'benchmark_coverage':float(frame.benchmark_coverage.iloc[0]),'included':False,
                          'exclusion_reason':'pool_not_yet_known' if not frame.pool_known_at_decision.any() else 'no_complete_feature_rows'})
        daily=pd.DataFrame(yearly).sort_values(['decile','decision_timestamp'])
        assert not daily.empty
        daily['cumulative_excess_simple_return']=daily.groupby('decile').excess_return.cumsum()
        daily.to_parquet(year_dir/'decile_daily.parquet',index=False)
        forecast.to_parquet(year_dir/'predictions_with_realized_labels.parquet',index=False)
        metrics.append(summarize(daily,year));all_rows.append(daily)
        gains=booster.get_score(importance_type='gain')
        for name in features:importance.append({'prediction_year':year,'feature':name,'gain':gains.get(name,0.)})
        record={'prediction_year':year,'training_table':train_path.name,'prediction_table':predict_path.name,
                'training_table_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),
                'prediction_table_sha256':hashlib.sha256(predict_path.read_bytes()).hexdigest(),
                'training_rows':len(fitting),'training_years':sorted(map(int,fitting.data_year.unique())),
                'training_label_latest_price_time_utc':fitting.label_price_time_utc.max().isoformat(),
                'fit_cutoff_utc':fit_asof.isoformat(),'first_forecast_utc':forecast.entry_time_utc.min().isoformat(),
                'forecasts':len(forecast),'forecast_days':forecast.entry_time_utc.nunique(),
                'evaluation_days':daily.decision_timestamp.nunique(),
                'gpu_device':actual_device,'gpu_updater':'grow_gpu_hist','trees':booster.num_boosted_rounds(),
                'max_depth':4,'fit_seconds':fit_seconds,'prediction_seconds':prediction_seconds,
                'first_evaluated_date':daily.date_beijing.min(),'last_evaluated_date':daily.date_beijing.max(),
                'unevaluated_forecast_dates':[r['date_beijing'] for r in cover if r['prediction_year']==year and not r['included']]}
        model_records.append(record)
        (year_dir/'run.json').write_text(json.dumps(record,indent=2))
        print(f'YEAR {year}: CUDA fit {fit_seconds:.3f}s, predict {prediction_seconds:.3f}s, train {len(fitting)}, eval {record["evaluation_days"]} days, IC {metrics[-1]["IC"]:+.5f}',flush=True)
    combined=pd.concat(all_rows,ignore_index=True)
    metric=pd.DataFrame(metrics);metric.to_csv(out/'annual_metrics.csv',index=False)
    combined.to_parquet(out/'all_years_decile_daily.parquet',index=False)
    pd.DataFrame(cover).to_csv(out/'date_coverage.csv',index=False)
    pd.DataFrame(importance).to_csv(out/'feature_importance_gain.csv',index=False)
    manifest={'training_mode':'expanding_2021_through_previous_year','gpu':gpu,'xgboost_version':xgb.__version__,'params':PARAMS,'num_boost_round':200,'early_stopping':False,
              'features':features,'target':'excess_return_24h','model_records':model_records,
              'sampling':'complete15 features and label for train; complete features and pool_known_at_decision for inference; equal weight per training observation',
              'benchmark':'provided table benchmark: equal-weight members of fixed annual pool with finite two-endpoint realized returns; not just forecast subset',
              'evaluation':'deciles assigned before observing outcomes; days with <20 forecasts or any forecast missing an outcome excluded only from scoring, retained in forecast outputs',
              'grouping':'ascending predicted excess; G1 lowest, G10 highest; fixed direction, no test-period sign flipping',
              'IC':'IC and RankIC both mean daily Spearman; Pearson_IC explicitly separate; ICIR mean/sample std, not annualized',
              'returns':'sum daily group return minus daily benchmark; year plots reset each year; gross, no fees/funding/slippage',
              'limitations':['Original15 previously screened on historical evaluation years; not an untouched out-of-sample validation',
                             'reported historical inputs may be revised','zero-latency entry-price labels do not model post-close compute/execution delays',
                             'missing outcome dates excluded from metrics; delisting/forced-settlement losses not independently reconstructed']}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    # Report comparison produced separately; no HTML generation.

if __name__=="__main__":
    run(Path("/tmp/xgb22run"))
