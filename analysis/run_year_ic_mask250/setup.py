from pathlib import Path
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
for count in [19,15]:
 dest=O/f'case{count}';dest.mkdir(exist_ok=True)
 for name in ['train.py','holdings.py','net.py','buffer.py']:
  s=(R/'analysis/run_xgb19alpha_300'/name).read_text(encoding='utf-8')
  s=s.replace('xgb19alpha_gpu_expanding_300_depth4_lr005',f'xgb{count}_yearmask_gpu_expanding_250_depth4_lr005')
  s=s.replace('xgb19alpha_300_lr005',f'xgb{count}_yearmask_250_lr005')
  s=s.replace('/tmp/xgb19alphacode300/',f'/tmp/yearmask250code/case{count}/')
  if name=='train.py':
   s=s.replace('import xgboost as xgb','import xgboost as xgb\nimport sys\nsys.path.insert(0,"/tmp/yearmask250code")\nfrom mask_years import apply_masks')
   s=s.replace('ROUNDS=300','ROUNDS=250').replace('==300','==250').replace("'num_boost_round':300","'num_boost_round':250")
   if count==15:
    s=s.replace("features=json.loads((base/'feature_columns.json').read_text())","features=json.loads((base/'feature_columns.json').read_text())[:15]").replace('==19','==15')
   s=s.replace('        dtrain=xgb.DMatrix(', '        apply_masks(train,fitting,features,fit_asof,year,year_dir)\n        dtrain=xgb.DMatrix(')
   s=s.replace('complete19 features and label for train',f'Original complete{count} features and label select training rows before yearly IC masking; masked cells are NaN, not dropped')
   s=s.replace("'features':features,'target'", "'features':features,'yearly_training_mask':'abs(mean daily RankIC)<=0.01 -> NaN, training-year only; see per-model audits','target'")
  (dest/name).write_text(s,encoding='utf-8')
print('Prepared19 and15 yearly mask experiments;250trees eta0.05;no plots')
