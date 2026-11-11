import sys
sys.path.append('/workspace/alpha-web3-research/.venv/lib/python3.12/site-packages')
from pathlib import Path
import pandas as pd
import torch,zipfile
rows=[]
for window,kind,folder in [('2021','direct','xgb15_120_lr005_net_compound_fee0005'),('2021','buffer','xgb15_120_lr005_hysteresis_10_15'),('rolling3y','direct','xgb15_rolling3y_net'),('rolling3y','buffer','xgb15_rolling3y_buffer')]:
    root=Path('/tmp')/folder
    annual=pd.read_csv(root/'annual_net_metrics.csv')
    if kind=='direct':
        daily=pd.read_csv(root/'daily_net_equity.csv')
        f=torch.tensor(daily.fee_fraction_before_trade.to_numpy(),device='cuda')
        annual['mean_daily_oneway_turnover']=[(f[torch.tensor(daily.date_beijing.str.startswith(str(y)).to_numpy(),device='cuda')].mean()/.001).item() for y in annual.year]
    annual['training_start']=window;annual['strategy']=kind
    rows.append(annual)
out=Path('/tmp/xgb15_rolling3y_comparison');out.mkdir(exist_ok=True)
pd.concat(rows,ignore_index=True).to_csv(out/'comparison.csv',index=False)
print(pd.concat(rows,ignore_index=True)[['training_start','strategy','year','net_compound_return','max_drawdown','sharpe','calmar','mean_daily_oneway_turnover']].to_string(index=False))
with zipfile.ZipFile('/tmp/xgb15_rolling3y_results.zip','w',zipfile.ZIP_DEFLATED) as z:
    for folder in ['xgb15_rolling3y_comparison','xgb15_rolling3y_120_lr005','xgb15_rolling3y_net','xgb15_rolling3y_buffer']:
        for p in (Path('/tmp')/folder).iterdir():
            if p.is_file():z.write(p,folder+'/'+p.name)
