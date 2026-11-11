"""Rank-buffer strategy: retain quantities, replace each exited slot only."""
from pathlib import Path

source = Path('/tmp/gpu_hysteresis_backtest.py').read_text()
exec(source.split('selected =')[0])
OUT = Path('/tmp/xgb15_120_lr005_hysteresis_fixed_quantities')
OUT.mkdir(exist_ok=True)
rate = .0005
slots = 64
slot_id = torch.arange(slots, device=device)[None, :].expand(2, slots)
side_sign = torch.tensor([1., -1.], device=device, dtype=torch.float64)[:, None]
idx = torch.full((2, slots), -1, device=device, dtype=torch.int64)
amount = torch.zeros((2, slots), device=device, dtype=torch.float64)
pending = torch.zeros_like(amount)
quota = torch.zeros(2, device=device, dtype=torch.int64)
equity = torch.ones((), device=device, dtype=torch.float64)
prior_active = torch.zeros((), device=device, dtype=torch.bool)
retained_error = torch.zeros((),device=device,dtype=torch.float64)
safe_ratio = torch.where(good_px & good_prior, px/prior_price, torch.ones_like(px))
holdings = torch.zeros((D, N), device=device, dtype=torch.float64)
out = {k: [] for k in ['net_return','net_equity','gross_return','fee_fraction','one_way_turnover',
                       'gross_exposure','net_exposure','long_count','short_count','fee_estimated','return_estimated']}
for d in range(D):
    day_quota = torch.stack([counts[d,9], counts[d,0]]).to(torch.int64)
    active = eligible[d].any()
    initial = active & ~prior_active
    quota = torch.where(initial, day_quota, quota)
    valid = idx.ge(0)
    gather_idx = idx.clamp(min=0)
    marked = amount * safe_ratio[d, gather_idx]
    missing_mark = (valid & ~(good_px[d,gather_idx] & good_prior[d,gather_idx])).any()
    old_book = torch.zeros(N, device=device, dtype=torch.float64)
    old_book.scatter_add_(0, gather_idx.flatten(), (marked*side_sign).flatten())
    new_idx = torch.full_like(idx, -1)
    new_amount = torch.zeros_like(amount)
    next_pending = torch.zeros_like(pending)
    for side, ranks, orders in [(0,rank_long,order_long),(1,rank_short,order_short)]:
        existing = idx[side]
        lookup = existing.clamp(min=0)
        stay = valid[side] & eligible[d,lookup] & (ranks[d,lookup] <= keep_limit[d]) & active
        occupied = torch.zeros(N, device=device, dtype=torch.int64)
        occupied.scatter_add_(0, lookup, stay.to(torch.int64))
        candidate = eligible[d] & (ranks[d] <= day_quota[side]) & occupied.eq(0)
        candidate_positions = torch.where(candidate[orders[d]], torch.arange(N,device=device), N)
        candidate_positions = torch.sort(candidate_positions).values
        candidate_symbols = orders[d,candidate_positions.clamp(max=N-1)]
        available = candidate.sum()
        slot_active = (slot_id[side] < quota[side]) & active
        need = ~stay & slot_active
        replacement_order = torch.cumsum(need.to(torch.int64),0)-1
        fill = need & (replacement_order < available)
        chosen = candidate_symbols[replacement_order.clamp(min=0,max=N-1)]
        # Each replacement inherits the exited position's marked notional.
        # Unfilled slots keep that budget in cash until an eligible entry exists.
        budget = torch.where(valid[side], marked[side], pending[side])
        initial_budget = .5*equity/(1+rate)/quota[side].clamp(min=1)
        budget = torch.where(initial & slot_active, initial_budget, budget)
        budget = torch.where(slot_active, budget, torch.zeros_like(budget))
        new_idx[side] = torch.where(stay,existing,torch.where(fill,chosen,-1))
        new_amount[side] = torch.where(stay,marked[side],torch.where(fill,budget,0))
        retained_error = torch.maximum(retained_error,torch.where(stay,(new_amount[side]-marked[side]).abs(),0).max())
        next_pending[side] = torch.where(need & ~fill,budget,0)
    new_book = torch.zeros(N,device=device,dtype=torch.float64)
    new_book.scatter_add_(0,new_idx.clamp(min=0).flatten(),(new_amount*side_sign).flatten())
    traded = (new_book-old_book).abs().sum()
    fee = rate*traded
    post_fee_equity = equity-fee
    profit = (new_book*return_matrix[d]).sum()
    next_equity = post_fee_equity+profit
    out['net_return'].append(next_equity/equity-1)
    out['net_equity'].append(next_equity)
    out['gross_return'].append(profit/post_fee_equity)
    out['fee_fraction'].append(fee/equity)
    out['one_way_turnover'].append(traded/(2*equity))
    out['gross_exposure'].append(new_book.abs().sum()/post_fee_equity)
    out['net_exposure'].append(new_book.sum()/post_fee_equity)
    out['long_count'].append((new_book>0).sum())
    out['short_count'].append((new_book<0).sum())
    out['fee_estimated'].append(missing_mark)
    out['return_estimated'].append((new_book.ne(0)&missing_outcomes[d]).any())
    holdings[d] = new_book
    # No trade is generated for retained quantities: marked amount carries forward.
    assert torch.isfinite(profit)
    equity = next_equity
    idx,amount,pending = new_idx,new_amount,next_pending
    prior_active = active
out = {k:torch.stack(v) for k,v in out.items()}
assert (out['net_equity']>0).all(), 'Portfolio insolvent; cannot report compound risk metrics'
assert retained_error == 0, 'Retained position quantities changed'
baseline_dir=Path('/tmp/xgb15_120_lr005_net_compound_fee0005')
baseline_annual=pd.read_csv(baseline_dir/'annual_net_metrics.csv').set_index('year')
baseline_daily=pd.read_csv(baseline_dir/'daily_net_equity.csv')
assert baseline_daily.date_beijing.tolist()==dates.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d').tolist()
baseline_turnover=torch.tensor(baseline_daily.fee_fraction_before_trade.to_numpy(),device=device)/(.0005*2)
annual=[]
for year in range(2022,2027):
    mask=torch.tensor(years==year,device=device)
    r=out['net_return'][mask]
    nav=torch.cat([torch.ones(1,device=device,dtype=torch.float64),torch.cumprod(1+r,0)])
    dd=(1-nav/torch.cummax(nav,0).values).max()
    ann=nav[-1]**(365/len(r))-1
    held=mask & (out['long_count']+out['short_count']>0)
    annual.append(dict(year=year,days=len(r),net_compound_return=(nav[-1]-1).item(),
                       annualized_return=ann.item(),max_drawdown=dd.item(),
                       sharpe=(r.mean()/r.std(correction=1)*365**.5).item(),calmar=(ann/dd).item(),
                       mean_daily_oneway_turnover=out['one_way_turnover'][mask].mean().item(),
                       mean_gross_exposure=out['gross_exposure'][held].mean().item(),
                       min_gross_exposure=out['gross_exposure'][held].min().item(),
                       max_gross_exposure=out['gross_exposure'][held].max().item(),
                       long_min=out['long_count'][held].min().item(),long_max=out['long_count'][held].max().item(),
                       short_min=out['short_count'][held].min().item(),short_max=out['short_count'][held].max().item(),
                       estimated_return_days=out['return_estimated'][mask].sum().item(),
                       estimated_fee_days=out['fee_estimated'][mask].sum().item(),
                       baseline_net_compound_return=baseline_annual.loc[year,'net_compound_return'],
                       baseline_mean_daily_oneway_turnover=baseline_turnover[mask].mean().item()))
pd.DataFrame(annual).to_csv(OUT/'annual_net_metrics.csv',index=False)
df=pd.DataFrame({k:v.cpu().numpy() for k,v in out.items()})
df.insert(0,'date_beijing',dates.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d'))
df.to_csv(OUT/'daily_net_equity.csv',index=False)
rr,cc=torch.where(holdings.ne(0))
pd.DataFrame(dict(date_beijing=dates.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d').to_numpy()[rr.cpu().numpy()],
                  symbol=np.array(symbols)[cc.cpu().numpy()],signed_notional=holdings[rr,cc].cpu().numpy()
                  )).to_parquet(OUT/'daily_holdings.parquet',index=False)
method=dict(model=str(MODEL),gpu=torch.cuda.get_device_name(),fee=.0005,
            selection='Keep existing long if current score rank<=ceil(N*.15), short mirror; fill from current original G10/G1 in score order, skipping retained names. Symbol-ascending ties.',
            slots='Initial annual position counts equal original G10/G1 on first forecast day; fixed slots thereafter, one-for-one replacement; insufficient entry candidates leave slots in cash.',
            sizing='Initial gross1x, long50% short50% of post-fee equity. Retained quantities unchanged; each new coin inherits its exited slot marked notional. No daily weight or gross reset.',
            cash='Unfilled replacement slots retain marked notional budget in cash; fees deducted from equity, not funded by selling retained positions.',
            annual_reset='No forecasts on Jan1 => close old annual pool; next first forecast opens new annual slots at initial1x gross. Terminal exit Oct28 included.',
            missing_data='Missing outcome assumed zero only for affected held asset; unavailable entry mark treated unchanged. Days flagged.',
            metrics='Daily net self-financing equity; Sharpe mean/std(sample)*sqrt365,riskfree0; Calmar annualized compound return/max drawdown.',
            exposure='Gross and net exposure drift; annual mean/min/max reported. No slippage, funding or borrowing cost.',
            horizon='2022-01-02 to 2026-10-28, last holding2026-10-27')
(OUT/'method.json').write_text(json.dumps(method,ensure_ascii=False,indent=2))
print(pd.DataFrame(annual).to_string(index=False),flush=True)
print('Total net return',equity.item()-1,flush=True)
