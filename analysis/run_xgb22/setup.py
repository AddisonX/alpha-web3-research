from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'analysis/run_xgb22'
MODEL='/tmp/xgb22_gpu_expanding_200_depth4_lr003_2022_2026'
s=(R/'analysis/buffer20/train15.py').read_text(encoding='utf-8')
s=s.replace("'eta':.05","'eta':.03").replace('ROUNDS=120','ROUNDS=200').replace('==120','==200').replace("'num_boost_round':120","'num_boost_round':200")
s=s.replace('factors15_2021_2026','factors22_2021_2026').replace('==15','==22')
s=s.replace('complete fifteen features','complete22 features')
s=s.replace('mean daily cross-sectional Pearson; RankIC Spearman; ICIR mean/sample std, not annualized','IC and RankIC both mean daily Spearman; Pearson_IC explicitly separate; ICIR mean/sample std, not annualized')
s=s.replace('ten factors selected using 2021-2025 performance; historical test years overlap feature selection','Original15 previously screened; new7 selected using2021-2026 diagnostics, overlapping these historical evaluation years; not an untouched out-of-sample validation')
s=s.replace('/tmp/xgb15restore_gpu_expanding_120_depth4_lr005_2022_2026',MODEL)
# Keep established evaluation code; omit unused old rendering and archive names.
s=s.split('def render(')[0]
s=s.split("    bundle=out.parent/")[0]
s+='\nif __name__=="__main__":\n    run(Path("/tmp/xgb22run"))\n'
(O/'train.py').write_text(s,encoding='utf-8')
for src,dest in [('analysis/buffer20/15_gpu_xgb_holdings_turnover.py','holdings.py'),('analysis/buffer20/15_gpu_net_compound_metrics.py','net.py'),('analysis/buffer30/15_gpu_hysteresis_backtest.py','buffer.py')]:
 s=(R/src).read_text(encoding='utf-8')
 s=s.replace('/tmp/xgb15restore_gpu_expanding_120_depth4_lr005_2022_2026',MODEL)
 s=s.replace('/workspace/xgb16run/data/model_tables/factors15_2021_2026','/tmp/xgb22run/data/model_tables/factors22_2021_2026')
 s=s.replace('/workspace/buffer20/15_gpu_xgb_holdings_turnover.py','/tmp/xgb22code/holdings.py').replace('/workspace/buffer20/15_gpu_net_compound_metrics.py','/tmp/xgb22code/net.py')
 s=s.replace('xgb15_120_lr005','xgb22_200_lr003')
 s=s.replace("sys.path.append('/workspace/alpha-web3-research/.venv/lib/python3.12/site-packages')",'')
 (O/dest).write_text(s,encoding='utf-8')
print('22factor scripts generated')
