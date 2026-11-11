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
ROUNDS=120


def correlation(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.ptp(x)>0 and np.ptp(y)>0 else None


def summarize(daily,year):
    days=daily.drop_duplicates('decision_timestamp')
    row={'prediction_year':year,'valid_days':len(days),
         'mean_predicted_members':float(days.predicted_members.mean()),
         'mean_benchmark_coverage':float(days.benchmark_coverage.mean())}
    for col,name in [('ic','IC'),('rank_ic','RankIC')]:
        vals=days[col].dropna(); std=vals.std(ddof=1)
        row[name]=float(vals.mean());row[name+'IR']=float(vals.mean()/std) if std>0 else None
    for g in range(1,11):
        row[f'G{g}_excess_simple']=float(daily.loc[daily.decile==g,'excess_return'].sum())
    row['G10_minus_G1']=row['G10_excess_simple']-row['G1_excess_simple']
    return row


def run(root):
    base=root/'data/model_tables/factors15_2021_2026'
    out=Path('/tmp/xgb15_rolling3y_120_lr005');out.mkdir(parents=True,exist_ok=True)
    features=json.loads((base/'feature_columns.json').read_text())
    assert len(features)==15 and len(set(features))==15
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
    assert xgb.build_info().get('USE_CUDA') is True,'XGBoost was built without CUDA'
    print('GPU',gpu,'XGBOOST',xgb.__version__,flush=True)
    all_rows,cover,metrics,model_records,importance=[],[],[],[],[]
    for year in range(2022,2027):
        year_dir=out/str(year);year_dir.mkdir(exist_ok=True)
        train_path=Path('/tmp/extend2019/data/model_tables/factors15_2020_2026')/f'train_table_2021-2025/train_table_{year-1}.parquet'
        predict_path=base/f'predict_table_2022-2026/predict_table_{year}.parquet'
        train=pd.read_parquet(train_path); train=train.loc[train.data_year.ge(year-3)].copy(); test=pd.read_parquet(predict_path)
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
        assert booster.num_boosted_rounds()==120
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
                   'predicted_members':len(frame),'ic':ic,'rank_ic':rank_ic,
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
    manifest={'training_mode':'expanding_2021_through_previous_year','gpu':gpu,'xgboost_version':xgb.__version__,'params':PARAMS,'num_boost_round':120,'early_stopping':False,
              'features':features,'target':'excess_return_24h','model_records':model_records,
              'sampling':'complete fifteen features and label for train; complete features and pool_known_at_decision for inference; equal weight per training observation',
              'benchmark':'provided table benchmark: equal-weight members of fixed annual pool with finite two-endpoint realized returns; not just forecast subset',
              'evaluation':'deciles assigned before observing outcomes; days with <20 forecasts or any forecast missing an outcome excluded only from scoring, retained in forecast outputs',
              'grouping':'ascending predicted excess; G1 lowest, G10 highest; fixed direction, no test-period sign flipping',
              'IC':'mean daily cross-sectional Pearson; RankIC Spearman; ICIR mean/sample std, not annualized',
              'returns':'sum daily group return minus daily benchmark; year plots reset each year; gross, no fees/funding/slippage',
              'limitations':['ten factors selected using 2021-2025 performance; historical test years overlap feature selection',
                             'reported historical inputs may be revised','zero-latency entry-price labels do not model post-close compute/execution delays',
                             'missing outcome dates excluded from metrics; delisting/forced-settlement losses not independently reconstructed']}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    render(out,combined,metric,model_records)
    bundle=out.parent/'XGBoost_GPU_15因子_累计历史_120树_深度4_学习率005_2022-2026.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out.parent).as_posix())
    with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
    print(metric.to_string(index=False),flush=True)
    print('GPU WALKFORWARD COMPLETE',flush=True)


def render(out,daily,metrics,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    colors=['#30123b','#4145ab','#4675ed','#39a2fc','#1bcfd4','#24eca6','#61fc6c','#b2f235','#faba39','#df2e16']
    doc='<meta charset="utf-8"><title>XGBoost GPU年度预测</title><style>body{font:15px system-ui;margin:25px auto;max-width:1120px;padding:0 20px;color:#203047}p{line-height:1.8}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}.wide{overflow:auto}nav{position:sticky;top:0;background:white;padding:12px;z-index:100}</style><h1>XGBoost · GPU · 15因子 · 累计历史 · 120棵树 · 深度4</h1>'
    doc+='<p>固定15个百分位因子，目标为未来24小时超额收益，学习率0.05，seed42，不早停、不逐年调参。train_table_(t-1)使用全部2021至t-1年数据训练后预测predict_table_t，分别对应2022—2026。实际模型配置确认CUDA和grow_gpu_hist，逐年恰好120轮。</p>'
    doc+='<p>使用训练表中2021年至上一年全部有效历史；仅使用训练截止时刻已经完成的标签。元旦零点币池尚未确定，因此首个预测日为1月2日。预测样本由因子与币池可知时间决定，先预测分组、再评价。任一预测币缺真实收益则该日不评分，预测记录仍保留。</p>'
    doc+='<p>G1预测值最低、G10最高。基准沿用表中同一固定币池当天有完整价格收益的成员等权，不是预测子集。十档为累计超额单利毛收益，不扣手续费、资金费和滑点；年度曲线分别从零累计。因子名单已参考2021—2025表现，历史成绩不属于严格独立的样本外检验。</p>'
    show=metrics.copy()
    for c in show.columns:
        if c.endswith('_excess_simple') or c=='G10_minus_G1':show[c]=show[c].map(lambda x:f'{x:.2%}')
    doc+='<h2>年度汇总</h2><div class="wide">'+show.to_html(index=False,float_format=lambda x:f'{x:.5f}')+'</div>'
    doc+='<nav>'+''.join(f'<a style="margin-right:25px" href="#y{r["prediction_year"]}">{r["prediction_year"]}</a>' for r in records)+'</nav><script>'+get_plotlyjs()+'</script>'
    fig_all,axes=plt.subplots(5,1,figsize=(13,23),layout='constrained')
    for ax_preview,record in zip(axes,records):
        year=record['prediction_year'];part=daily.loc[daily.prediction_year==year]
        m=metrics.loc[metrics.prediction_year==year].iloc[0]
        dates=pd.date_range(f'{year}-01-01',part.date_beijing.max())
        stamps=dates.tz_localize('Asia/Shanghai').tz_convert('UTC').as_unit('ms').asi8
        exit_dates=dates+pd.Timedelta(days=1)
        fig,ax=plt.subplots(figsize=(13,6),layout='constrained');chart=go.Figure()
        for g,color in enumerate(colors,1):
            values=part.loc[part.decile==g].set_index('decision_timestamp').cumulative_excess_simple_return.reindex(stamps).to_numpy()
            ax.plot(exit_dates,values,label=f'G{g}',color=color,linewidth=1.3)
            ax_preview.plot(exit_dates,values,label=f'G{g}',color=color,linewidth=1.1)
            chart.add_scatter(x=exit_dates,y=values,name=f'G{g}',mode='lines',connectgaps=False,line={'color':color})
        title=f'XGBoost GPU 15 factors expanding eta0.05 | predict {year} | 120 trees, depth 4\nIC {m.IC:+.4f} | ICIR {m.ICIR:+.4f} | RankIC {m.RankIC:+.4f} | {int(m.valid_days)} evaluated days'
        ax.set_title(title);ax_preview.set_title(title,fontsize=11)
        for a in [ax,ax_preview]:
            a.yaxis.set_major_formatter(PercentFormatter(1));a.grid(alpha=.2);a.axhline(0,color='gray',linewidth=.6)
        ax.set_ylabel('Cumulative excess simple return vs provided full-pool benchmark')
        ax.set_xlabel('Exit date, Beijing time | G1 lowest score / G10 highest | gross')
        ax.legend(ncol=10,fontsize=8)
        fig.savefig(out/f'xgb_{year}_deciles.png',dpi=145);plt.close(fig)
        chart.update_layout(height=600,title=f'{year} · IC {m.IC:+.4f} · ICIR {m.ICIR:+.4f} · RankIC {m.RankIC:+.4f}',yaxis_title='累计超额单利收益',yaxis_tickformat='.0%',xaxis_title='收益兑现日期（北京时间）',legend_orientation='h')
        doc+=f'<section id="y{year}"><h2>{year} 年预测</h2><p>训练样本 {record["training_rows"]:,}；GPU训练 {record["fit_seconds"]:.3f} 秒；有效评分 {record["evaluation_days"]} 天，{record["first_evaluated_date"]}—{record["last_evaluated_date"]}。</p>'+chart.to_html(full_html=False,include_plotlyjs=False)+'</section>'
    axes[0].legend(ncol=10,fontsize=8)
    fig_all.savefig(out/'xgb_2022_2026_overview.png',dpi=125);plt.close(fig_all)
    doc+='<h2>模型与数据记录</h2><pre>'+json.dumps(records,ensure_ascii=False,indent=2)+'</pre>'
    (out/'XGBoost_GPU年度预测_2022-2026.html').write_text(doc,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args();run(args.root)

if __name__ == "__main__":
    output=Path('/tmp/xgb15_rolling3y_120_lr005')
    with zipfile.ZipFile('/tmp/xgb15_rolling3y_charts.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for p in output.iterdir():
            if p.is_file():archive.write(p,output.name+'/'+p.name)
