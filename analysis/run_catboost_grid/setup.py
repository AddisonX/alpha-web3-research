from pathlib import Path
import json,re
R=Path('.');O=R/'analysis/run_catboost_grid'
base=(R/'analysis/run_xgb42screen_200/train.py').read_text(encoding='utf-8')
features42=json.loads((R/'analysis/run_xgb42screen_200/feature_columns.json').read_text());features15=json.loads((R/'data/model_tables/factors57alpha_2021_2026/feature_columns.json').read_text())[:15]
for count,names in [(15,features15),(42,features42)]:
 for trees in [120,400]:
  tag=f'catboost{count}_{trees}';dest=O/tag;dest.mkdir(exist_ok=True)
  s=base.replace('import xgboost as xgb','import catboost as cb\nfrom catboost.utils import get_gpu_device_count')
  start=s.index('PARAMS=');end=s.index('\n\n\ndef correlation',start)
  s=s[:start]+f"PARAMS=dict(loss_function='RMSE',task_type='GPU',devices='0',depth=4,learning_rate=.04,iterations={trees},random_seed=42,thread_count=4,nan_mode='Min',boosting_type='Plain',bootstrap_type='No',l2_leaf_reg=3.,border_count=128,allow_writing_files=False,use_best_model=False,verbose=False)\nROUNDS={trees}"+s[end:]
  s=s.replace('xgb42screen_native_gpu_expanding_200_depth4_lr004',f'{tag}_gpu_expanding_depth4_lr004')
  s=re.sub(r'    features=.*\n    assert len\(features\).*',f'    features={names!r}\n    assert len(features)=={count} and len(set(features))=={count}',s)
  s=s.replace("assert xgb.build_info().get('USE_CUDA') is True,'XGBoost was built without CUDA'","assert get_gpu_device_count()>0,'No CUDA device available'")
  s=s.replace("'XGBOOST',xgb.__version__","'CATBOOST',cb.__version__")
  start=s.index('        dtrain=xgb.DMatrix');end=s.index('        prediction_seconds=',start)
  s=s[:start]+'''        dtrain=cb.Pool(fitting[features],label=fitting.excess_return_24h,feature_names=features)
        before=time.perf_counter();booster=cb.CatBoostRegressor(**PARAMS);booster.fit(dtrain)
        fit_seconds=time.perf_counter()-before;config=booster.get_all_params();actual_device=config['task_type']
        assert actual_device=='GPU' and booster.tree_count_==ROUNDS and config['depth']==4
        assert np.isclose(config['learning_rate'],.04)
        booster.save_model(str(year_dir/'model.cbm'))
        (year_dir/'booster_config.json').write_text(json.dumps(config,indent=2))
        (year_dir/'training_rmse.json').write_text(json.dumps(booster.get_evals_result()))
        before=time.perf_counter()
        forecast['predicted_excess_return']=booster.predict(cb.Pool(forecast[features],feature_names=features),task_type='GPU')
'''+s[end:]
  s=s.replace("gains=booster.get_score(importance_type='gain')","gains=dict(zip(features,booster.feature_importances_))").replace("'gain':gains.get(name,0.)","'prediction_values_change':gains.get(name,0.)")
  s=s.replace("'gpu_updater':'grow_gpu_hist'","'gpu_updater':'CatBoost GPU'").replace('booster.num_boosted_rounds()','booster.tree_count_')
  s=s.replace("'xgboost_version':xgb.__version__","'catboost_version':cb.__version__").replace("'num_boost_round':200",f"'num_boost_round':{trees}").replace('XGBoost','CatBoost').replace('Selected42 filtered',f'Selected{count} filtered')
  s=s.replace('feature_importance_gain.csv','feature_importance.csv')
  (dest/'train.py').write_text(s,encoding='utf-8')
  for file in ['holdings.py','net.py','buffer.py']:
   s=(R/'analysis/run_xgb42screen_200'/file).read_text(encoding='utf-8')
   s=s.replace('xgb42screen_native_gpu_expanding_200_depth4_lr004',f'{tag}_gpu_expanding_depth4_lr004').replace('xgb42screen_native_200_lr004',f'{tag}_lr004')
   s=s.replace('/tmp/xgb42screencode200/',f'/tmp/catboostgrid/{tag}/')
   (dest/file).write_text(s,encoding='utf-8')
print('Four CatBoost cases prepared')
