"""Exact additive 2025 NAV contribution bridge, with CUDA counterfactuals."""
from pathlib import Path
source=Path('/tmp/gpu_hysteresis_backtest.py').read_text()
exec(source.split('selected =')[0])
OUT=Path('/tmp/xgb15_hysteresis_2025_attribution');OUT.mkdir(exist_ok=True)
mask_np=years==2025
mask=torch.tensor(mask_np,device=device)
day_dates=dates.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d')
month_np=day_dates.to_numpy()[mask_np].astype('U7')
cases={}
def dense(frame,col):
    a=torch.zeros((D,N),device=device,dtype=torch.float64)
    stamps_q=pd.to_datetime(frame.date_beijing).dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC').dt.as_unit('ms').astype('int64')
    rr=torch.tensor(pd.Index(stamps).get_indexer(stamps_q),device=device)
    cc=torch.tensor(pd.Index(symbols).get_indexer(frame.symbol),device=device)
    assert (rr>=0).all() and (cc>=0).all()
    a[rr,cc]=torch.tensor(frame[col].to_numpy(),device=device,dtype=torch.float64)
    return a
for name,dirname in [('baseline','xgb15_120_lr005_net_compound_fee0005'),
                     ('buffer_equal','xgb15_120_lr005_hysteresis_10_15'),
                     ('buffer_fixed','xgb15_120_lr005_hysteresis_fixed_quantities')]:
    folder=Path('/tmp')/dirname
    daily=pd.read_csv(folder/'daily_net_equity.csv')
    assert daily.date_beijing.tolist()==day_dates.tolist()
    r=torch.tensor(daily.net_return.to_numpy(),device=device,dtype=torch.float64)
    fcol='fee_fraction_before_trade' if name=='baseline' else 'fee_fraction'
    f=torch.tensor(daily[fcol].to_numpy(),device=device,dtype=torch.float64)
    if name=='baseline':
        post_weights=weights[:,:,10]*(1-f[:,None])
    elif name=='buffer_equal':
        h=pd.read_parquet(folder/'daily_holdings.parquet')
        post_weights=dense(h,'weight')*(1-f[:,None])
    else:
        h=pd.read_parquet(folder/'daily_holdings.parquet')
        nom=dense(h,'signed_notional')
        eq=torch.tensor(daily.net_equity.to_numpy(),device=device,dtype=torch.float64)
        pre_eq=torch.cat([torch.ones_like(eq[:1]),eq[:-1]])
        post_weights=nom/pre_eq[:,None]
    assert torch.max(((post_weights*return_matrix).sum(1)-f-r).abs())<1e-10
    ry=r[mask];fy=f[mask];wy=post_weights[mask]
    nav=torch.cumprod(1+ry,0)
    prior_nav=torch.cat([torch.ones_like(nav[:1]),nav[:-1]])
    asset_contrib=wy*return_matrix[mask]*prior_nav[:,None]
    fee_contrib=-fy*prior_nav
    assert torch.abs(asset_contrib.sum()+fee_contrib.sum()-(nav[-1]-1))<1e-10
    long_contrib=torch.where(wy>0,asset_contrib,0).sum(1)
    short_contrib=torch.where(wy<0,asset_contrib,0).sum(1)
    gross=wy.abs().sum(1)/(1-fy)
    long_exp=torch.clamp(wy,min=0).sum(1)/(1-fy)
    short_exp=-torch.clamp(wy,max=0).sum(1)/(1-fy)
    cases[name]=dict(r=ry,asset=asset_contrib,fees=fee_contrib,nav=nav,long=long_contrib,short=short_contrib,
                     gross=gross,long_exposure=long_exp,short_exposure=short_exp)
summary=[];monthly=[]
for name,c in cases.items():
    active=c['gross']>0
    summary.append(dict(case=name,net_return=(c['nav'][-1]-1).item(),long_pnl=c['long'].sum().item(),
                        short_pnl=c['short'].sum().item(),fee_drag=c['fees'].sum().item(),
                        mean_gross=c['gross'][active].mean().item(),
                        mean_long_exposure=c['long_exposure'][active].mean().item(),
                        mean_short_exposure=c['short_exposure'][active].mean().item(),
                        final_long_exposure=c['long_exposure'][-1].item(),final_short_exposure=c['short_exposure'][-1].item()))
    for month in sorted(set(month_np)):
        mm=torch.tensor(month_np==month,device=device)
        monthly.append(dict(month=month,case=name,net_return=(torch.prod(1+c['r'][mm])-1).item(),
                            contribution_to_year=(c['long'][mm]+c['short'][mm]+c['fees'][mm]).sum().item(),
                            mean_long_exposure=c['long_exposure'][mm].mean().item(),
                            mean_short_exposure=c['short_exposure'][mm].mean().item(),
                            long_pnl=c['long'][mm].sum().item(),short_pnl=c['short'][mm].sum().item(),fee_drag=c['fees'][mm].sum().item()))
pd.DataFrame(summary).to_csv(OUT/'annual_bridge.csv',index=False)
pd.DataFrame(monthly).to_csv(OUT/'monthly_attribution.csv',index=False)
coin=pd.DataFrame(dict(symbol=symbols,baseline=cases['baseline']['asset'].sum(0).cpu().numpy(),
                       buffer_equal=cases['buffer_equal']['asset'].sum(0).cpu().numpy(),
                       buffer_fixed=cases['buffer_fixed']['asset'].sum(0).cpu().numpy(),
                       sizing_difference=(cases['buffer_fixed']['asset']-cases['buffer_equal']['asset']).sum(0).cpu().numpy()))
coin=coin.sort_values('sizing_difference')
coin.to_csv(OUT/'coin_contribution.csv',index=False)
print('ANNUAL',pd.DataFrame(summary).to_string(index=False),sep='\n',flush=True)
print('MONTHLY',pd.DataFrame(monthly).to_string(index=False),sep='\n',flush=True)
print('COIN WORST',coin.head(12).to_string(index=False),sep='\n',flush=True)
print('COIN BEST',coin.tail(5).to_string(index=False),sep='\n',flush=True)
(OUT/'method.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
    attribution='Exact yearly additive NAV contribution = start-of-day year-normalized equity * daily asset return contribution; fees likewise. Contributions sum to yearly compound return.',
    comparison='Buffer-equal counterfactual retains daily decile counts, daily equalweights and gross1x; buffer-fixed retains initial annual slots, quantities, and slot notional on replacement. Thus bridge includes weighting, exposure drift and small count-rule differences.',
    missing='Same underlying missing-return0 and missing-mark assumptions as source cases; no new training or signals.'),indent=2))
