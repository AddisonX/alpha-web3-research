from pathlib import Path
import json,time
import numpy as np,pandas as pd,xgboost as xgb
BASE=Path('/workspace/xgb16run/data/model_tables/factors15_2021_2026');OUT=Path('/tmp/xgb15_rolling2m_120_lr005');OUT.mkdir(exist_ok=True)
features=json.loads((BASE/'feature_columns.json').read_text());assert len(features)==15
PARAMS=dict(objective='reg:squarederror',tree_method='hist',device='cuda:0',max_depth=4,eta=.05,subsample=1.,colsample_bytree=1.,min_child_weight=1.,reg_lambda=1.,reg_alpha=0.,max_bin=256,seed=42,nthread=4,eval_metric='rmse')
assert xgb.build_info()['USE_CUDA'];records=[];daily_all=[];annual=[];coverage=[]
def corr(x,y):return float(np.corrcoef(x,y)[0,1]) if np.ptp(x)>0 and np.ptp(y)>0 else np.nan
for year in range(2022,2027):
 train=pd.read_parquet(BASE/f'train_table_2021-2025/train_table_{year-1}.parquet');test=pd.read_parquet(BASE/f'predict_table_2022-2026/predict_table_{year}.parquet')
 test=test.loc[test.decision_date_beijing.le('2026-10-27')].copy();assert train.pool_year.eq(year).all() and test.pool_year.eq(year).all()
 hist=pd.concat([train,test],ignore_index=True).sort_values(['entry_time_utc','symbol']);assert not hist.duplicated(['entry_time_utc','symbol']).any()
 complete=np.isfinite(hist[features]).all(axis=1)&np.isfinite(hist.excess_return_24h)
 predictions=[];ydir=OUT/str(year);ydir.mkdir(exist_ok=True)
 for month in range(1,13 if year<2026 else 11):
  boundary=pd.Timestamp(year=year,month=month,day=1,tz='Asia/Shanghai');start=boundary-pd.DateOffset(months=2);end=boundary+pd.DateOffset(months=1)
  fit=hist.loc[complete&hist.entry_time_utc.ge(start)&hist.entry_time_utc.lt(boundary)&hist.label_price_time_utc.le(boundary)].copy()
  forecast=test.loc[test.entry_time_utc.ge(boundary)&test.entry_time_utc.lt(end)&np.isfinite(test[features]).all(axis=1)&test.pool_known_at_decision].copy()
  assert len(fit)>1000 and len(forecast)>0
  assert fit.factor_bar_close_utc.le(fit.entry_time_utc).all() and fit.market_cap_available_bound_utc.le(fit.entry_time_utc).all()
  assert fit.label_price_time_utc.max()<=boundary
  before=time.perf_counter();model=xgb.train(PARAMS,xgb.DMatrix(fit[features],label=fit.excess_return_24h,feature_names=features),num_boost_round=120)
  seconds=time.perf_counter()-before;config=json.loads(model.save_config());assert config['learner']['generic_param']['device'].startswith('cuda');assert 'grow_gpu_hist' in json.dumps(config) and model.num_boosted_rounds()==120
  model.save_model(ydir/f'model_{month:02d}.ubj');forecast['predicted_excess_return']=model.predict(xgb.DMatrix(forecast[features],feature_names=features));forecast['model_month']=f'{year}-{month:02d}';predictions.append(forecast)
  records.append(dict(prediction_month=f'{year}-{month:02d}',pool_year=year,training_start=str(start),training_end_exclusive=str(boundary),training_rows=len(fit),training_label_latest=str(fit.label_price_time_utc.max()),first_forecast=str(forecast.entry_time_utc.min()),forecast_rows=len(forecast),gpu=config['learner']['generic_param']['device'],fit_seconds=seconds))
 forecast=pd.concat(predictions).sort_values(['entry_time_utc','symbol']);forecast['prediction_decile']=0;rows=[]
 for stamp,frame in forecast.groupby('entry_time_utc',sort=True):
  order=np.argsort(frame.predicted_excess_return.to_numpy(),kind='stable');groups=np.array_split(order,10)
  for g,indices in enumerate(groups,1):forecast.loc[frame.index[indices],'prediction_decile']=g
  included=len(frame)>=20 and frame.realized_return_24h.notna().all() and frame.excess_return_24h.notna().all()
  day=stamp.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d');ts=int(stamp.timestamp()*1000)
  coverage.append(dict(prediction_year=year,date_beijing=day,included=included,predicted_members=len(frame),label_members=int(frame.realized_return_24h.notna().sum())))
  if not included:continue
  pred=frame.predicted_excess_return.to_numpy();excess=frame.excess_return_24h.to_numpy();ic=corr(pd.Series(pred).rank().to_numpy(),pd.Series(excess).rank().to_numpy());pic=corr(pred,excess)
  benchmark=float(frame.benchmark_return_24h.iloc[0]);returns=frame.realized_return_24h.to_numpy()
  for g,ix in enumerate(groups,1):
   gr=float(returns[ix].mean());rows.append(dict(prediction_year=year,date_beijing=day,decision_timestamp=ts,exit_timestamp=ts+86400000,decile=g,members=len(ix),predicted_members=len(frame),ic=ic,rank_ic=ic,pearson_ic=pic,group_return=gr,benchmark_return=benchmark,excess_return=gr-benchmark,benchmark_coverage=float(frame.benchmark_coverage.iloc[0])))
 daily=pd.DataFrame(rows).sort_values(['decile','decision_timestamp']);daily['cumulative_excess_simple_return']=daily.groupby('decile').excess_return.cumsum();daily.to_parquet(ydir/'decile_daily.parquet',index=False);forecast.to_parquet(ydir/'predictions_with_realized_labels.parquet',index=False);daily_all.append(daily)
 days=daily.drop_duplicates('decision_timestamp');rec=dict(prediction_year=year,valid_days=len(days),mean_predicted_members=days.predicted_members.mean())
 for col,name in [('rank_ic','IC'),('pearson_ic','Pearson_IC')]:
  v=days[col].dropna();rec[name]=v.mean();rec[name+'IR']=v.mean()/v.std(ddof=1)
 rec['RankIC']=rec['IC'];rec['RankICIR']=rec['ICIR']
 for g in range(1,11):rec[f'G{g}_excess_simple']=daily.loc[daily.decile==g,'excess_return'].sum()
 annual.append(rec);print('YEAR',year,'RankIC',rec['IC'],'days',len(days),flush=True)
pd.concat(daily_all,ignore_index=True).to_parquet(OUT/'all_years_decile_daily.parquet',index=False);pd.DataFrame(annual).to_csv(OUT/'annual_metrics.csv',index=False);pd.DataFrame(records).to_csv(OUT/'monthly_training_records.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'date_coverage.csv',index=False)
(OUT/'manifest.json').write_text(json.dumps(dict(features=features,params=PARAMS,trees=120,xgboost=xgb.__version__,training='Previous2 full Beijing calendar months; monthly independent fit; use prediction-year fixedpool historical percentiles and price excess labels; no funding factors',label_boundary='Training label price timestamp no later than month boundary; January predictions begin Jan2 because pool becomes known after Jan1 BJT midnight',evaluation='IC is Spearman daily cross-sectional, ICIR unannualized; daily deciles omit incomplete-label dates; portfolio backtest retains forecasts and separately flags estimated held outcomes',models=len(records),monthly_records=records),indent=2))
print('MONTHLY MODELS',len(records),'GPU fit seconds',sum(x['fit_seconds'] for x in records),flush=True)
