"""Post-fee, self-financing 1x-gross long/short equity, calculated on CUDA."""
from pathlib import Path

# Reuse precisely the predictions, membership and prices used for turnover.
source = Path('/tmp/xgb20code300/holdings.py').read_text()
exec(source.split('rows=[];legrows=[]')[0])
OUT = Path('/tmp/xgb20_300_lr005_net_compound_fee0005')
OUT.mkdir(exist_ok=True)
rate = 0.0005
returns = scores.pivot(index='decision_timestamp', columns='decile', values='group_return')
gross_series = .5 * (returns[10] - returns[1])
gross_np = gross_series.reindex(stamps).to_numpy(copy=True)
active_np = total_count.cpu().numpy() > 0
missing_return_dates = dates[active_np & ~np.isfinite(gross_np)]
return_matrix = torch.zeros((D, N), device=device, dtype=torch.float64)
missing_outcomes = torch.zeros((D, N), device=device, dtype=torch.bool)
for year in range(2022, 2027):
    q = pd.read_parquet(MODEL/str(year)/'predictions_with_realized_labels.parquet')
    q = q.loc[q.entry_time_utc.dt.as_unit('ms').astype('int64') < end]
    qr = torch.tensor(pd.Index(stamps).get_indexer(q.entry_time_utc.dt.as_unit('ms').astype('int64')), device=device)
    qc = torch.tensor(pd.Index(symbols).get_indexer(q.symbol), device=device)
    vals = torch.tensor(q.realized_return_24h.to_numpy(), device=device, dtype=torch.float64)
    return_matrix[qr, qc] = torch.nan_to_num(vals, nan=0.)
    missing_outcomes[qr, qc] = ~torch.isfinite(vals)
raw_g = (weights[:, :, 10] * return_matrix).sum(dim=1)
missing_held_outcome = (weights[:, :, 10].ne(0) & missing_outcomes).any(dim=1)
print('Missing outcome on actual held dates:', dates[missing_held_outcome.cpu().numpy()].tolist(), flush=True)
known = torch.tensor(np.isfinite(gross_np), device=device)
assert torch.max((raw_g[known] - torch.tensor(gross_np[np.isfinite(gross_np)], device=device)).abs()) < 1e-10
gross_np[active_np & ~np.isfinite(gross_np)] = raw_g.cpu().numpy()[active_np & ~np.isfinite(gross_np)]
gross_np[~active_np] = 0
g = torch.tensor(gross_np, device=device, dtype=torch.float64)
w = weights[:, :, 10]
old_w = old[:, :, 10]
prior_g = torch.cat([torch.zeros_like(g[:1]), g[:-1]])
assert (1 + prior_g > 0).all()
# Four transitions lack a mark. Assume unchanged price ONLY for those missing
# marks, retaining actual known marks and portfolio equity drift. Flag each.
missing_mark = old_w.ne(0) & ~(good_px & good_prior)
safe_ratio = torch.where(good_px & good_prior, px / prior_price, torch.ones_like(px))
before = old_w * safe_ratio / (1 + prior_g[:, None])
# Targets use post-fee equity: f = rate * sum(abs((1-f)*w - old_marked)).
f = torch.zeros_like(g)
for _ in range(20):
    f = rate * ((1 - f[:, None]) * w - before).abs().sum(dim=1)
assert torch.max((f - rate * ((1 - f[:, None]) * w - before).abs().sum(dim=1)).abs()) < 1e-12
net = (1 - f) * (1 + g) - 1
assert (1 + net > 0).all()
equity = torch.cumprod(1 + net, dim=0)
annual = []
for year in range(2022, 2027):
    mask = torch.tensor(years == year, device=device)
    r = net[mask]
    nav = torch.cat([torch.ones(1, device=device, dtype=torch.float64), torch.cumprod(1+r, 0)])
    dd = 1 - nav / torch.cummax(nav, 0).values
    ret = nav[-1] - 1
    ann_ret = nav[-1] ** (365 / len(r)) - 1
    sharpe = r.mean() / r.std(correction=1) * (365 ** .5)
    mdd = dd.max()
    annual.append(dict(year=year, days=len(r), net_compound_return=ret.item(),
                       annualized_net_return=ann_ret.item(), max_drawdown=mdd.item(),
                       sharpe=sharpe.item(), calmar=(ann_ret/mdd).item(),
                       fee_estimated_days=missing_mark[mask].any(dim=1).sum().item(),
                       return_estimated_days=missing_held_outcome[mask].sum().item()))
pd.DataFrame(annual).to_csv(OUT/'annual_net_metrics.csv', index=False)
df = pd.DataFrame(dict(date_beijing=dates.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d'),
                       gross_return=g.cpu().numpy(), fee_fraction_before_trade=f.cpu().numpy(),
                       net_return=net.cpu().numpy(), net_equity=equity.cpu().numpy(),
                       fee_estimated=missing_mark.any(dim=1).cpu().numpy(),
                       return_estimated=missing_held_outcome.cpu().numpy()))
df.to_csv(OUT/'daily_net_equity.csv', index=False)
method = dict(model=str(MODEL), gpu=torch.cuda.get_device_name(), fee_per_traded_notional=rate,
              gross_exposure=1.0, long_exposure=.5, short_exposure=.5,
              rebalance='daily Beijing 00:00, holdings scaled to current post-fee equity',
              fee_equation='f = 0.0005 * sum(abs((1-f)*target_weights - old_positions_marked/pretrade_equity))',
              net_return='(1-f)*(1+0.5*(G10_return-G1_return))-1',
              sharpe='mean daily net return / sample daily std * sqrt(365), zero risk-free',
              calmar='annualized compounded net return / maximum drawdown; 365-day year',
              annual_assignment='return assigned to Beijing entry date; cross-year equity carried, annual drawdown includes start equity',
              missing_marks='Unchanged price assumed only for missing position marks; missing held-asset outcome assumed zero return; all estimated days flagged in daily output',
              includes='initial entry, cross-year exit/reentry, terminal liquidation; no slippage or funding',
              last_scored_holding='2026-10-27', terminal_liquidation='2026-10-28')
(OUT/'method.json').write_text(json.dumps(method,ensure_ascii=False,indent=2))
print(pd.DataFrame(annual).to_string(index=False))
print('Full-period net compounded return:', equity[-1].item()-1)
