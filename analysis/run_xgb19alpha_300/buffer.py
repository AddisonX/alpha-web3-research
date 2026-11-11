"""Reuse saved XGBoost forecasts; CUDA rank hysteresis and post-fee equity."""
from pathlib import Path

source = Path('/tmp/xgb19alphacode300/net.py').read_text()
# This initializes scores, forecasts, price arrays, return arrays and CUDA.
exec(source.split('raw_g =')[0])
OUT = Path('/tmp/xgb19alpha_300_lr005_hysteresis_10_30')
OUT.mkdir(exist_ok=True)
score = torch.full((D, N), -torch.inf, device=device, dtype=torch.float64)
for year in range(2022, 2027):
    q = pd.read_parquet(MODEL/str(year)/'predictions_with_realized_labels.parquet')
    q = q.loc[q.entry_time_utc.dt.as_unit('ms').astype('int64') < end]
    qr = torch.tensor(pd.Index(stamps).get_indexer(q.entry_time_utc.dt.as_unit('ms').astype('int64')), device=device)
    qc = torch.tensor(pd.Index(symbols).get_indexer(q.symbol), device=device)
    score[qr, qc] = torch.tensor(q.predicted_excess_return.to_numpy(), device=device, dtype=torch.float64)
eligible = torch.isfinite(score)
order_long = torch.argsort(score, dim=1, descending=True, stable=True)
order_short = torch.argsort(torch.where(eligible, score, torch.inf), dim=1, stable=True)
rank_long = torch.empty_like(order_long)
rank_short = torch.empty_like(order_short)
ordinal = torch.arange(1, N+1, device=device).expand(D, N)
rank_long.scatter_(1, order_long, ordinal)
rank_short.scatter_(1, order_short, ordinal)
keep_limit = torch.ceil(eligible.sum(dim=1) * .30).to(torch.int64)
selected = torch.zeros((D, N, 2), device=device, dtype=torch.bool)
prev = torch.zeros((N, 2), device=device, dtype=torch.bool)
for d in range(D):
    for side, ranks, orders, k in [(0, rank_long, order_long, counts[d, 9].to(torch.int64)),
                                    (1, rank_short, order_short, counts[d, 0].to(torch.int64))]:
        # Keep the best retained names if the target decile size shrinks.
        stay = prev[:, side] & eligible[d] & (ranks[d] <= keep_limit[d])
        stay_cumsum = torch.cumsum(stay[orders[d]].to(torch.int64), 0)
        stay_rank = torch.empty_like(stay_cumsum).scatter_(0, orders[d], stay_cumsum)
        stay = stay & (stay_rank <= k)
        # Only entry-decile members can fill vacancies; retained names are skipped.
        candidates = eligible[d] & (ranks[d] <= k) & ~stay
        ordered_cumsum = torch.cumsum(candidates[orders[d]].to(torch.int64), 0)
        entry_rank = torch.empty_like(ordered_cumsum).scatter_(0, orders[d], ordered_cumsum)
        add = candidates & (entry_rank <= (k - stay.sum()))
        selected[d, :, side] = stay | add
    prev = selected[d]
assert not (selected[:, :, 0] & selected[:, :, 1]).any()
assert torch.equal(selected[:, :, 0].sum(1), counts[:, 9].to(torch.int64))
assert torch.equal(selected[:, :, 1].sum(1), counts[:, 0].to(torch.int64))
w = (.5 * selected[:, :, 0].to(torch.float64) / selected[:, :, 0].sum(1).clamp(min=1)[:, None]
     - .5 * selected[:, :, 1].to(torch.float64) / selected[:, :, 1].sum(1).clamp(min=1)[:, None])
g = (w * return_matrix).sum(1)
missing_held = (w.ne(0) & missing_outcomes).any(1)
old_w = torch.cat([torch.zeros_like(w[:1]), w[:-1]])
prior_g = torch.cat([torch.zeros_like(g[:1]), g[:-1]])
safe_ratio = torch.where(good_px & good_prior, px/prior_price, torch.ones_like(px))
before = old_w * safe_ratio / (1 + prior_g[:, None])
missing_mark = (old_w.ne(0) & ~(good_px & good_prior)).any(1)
f = torch.zeros_like(g)
for _ in range(20):
    f = .0005 * ((1 - f[:, None])*w - before).abs().sum(1)
trade = f/.0005
net = (1-f)*(1+g)-1
assert (1+net > 0).all()
nav = torch.cumprod(1+net, 0)
annual = []
for year in range(2022, 2027):
    mask = torch.tensor(years == year, device=device)
    r = net[mask]
    yr_nav = torch.cat([torch.ones(1, device=device, dtype=torch.float64), torch.cumprod(1+r, 0)])
    dd = (1 - yr_nav/torch.cummax(yr_nav, 0).values).max()
    annualized = yr_nav[-1]**(365/len(r))-1
    annual.append(dict(year=year, days=len(r), net_compound_return=(yr_nav[-1]-1).item(),
                       annualized_return=annualized.item(), max_drawdown=dd.item(),
                       sharpe=(r.mean()/r.std(correction=1)*365**.5).item(), calmar=(annualized/dd).item(),
                       mean_daily_oneway_turnover=(trade[mask].mean()/2).item(),
                       estimated_return_days=missing_held[mask].sum().item(),
                       estimated_fee_days=missing_mark[mask].sum().item()))
pd.DataFrame(annual).to_csv(OUT/'annual_net_metrics.csv', index=False)
pd.DataFrame(dict(date_beijing=dates.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d'),
                  gross_return=g.cpu().numpy(), fee_fraction=f.cpu().numpy(),
                  one_way_turnover=(trade/2).cpu().numpy(), net_return=net.cpu().numpy(),
                  net_equity=nav.cpu().numpy(), long_count=selected[:,:,0].sum(1).cpu().numpy(),
                  short_count=selected[:,:,1].sum(1).cpu().numpy(),
                  return_estimated=missing_held.cpu().numpy(), fee_estimated=missing_mark.cpu().numpy()
                  )).to_csv(OUT/'daily_net_equity.csv', index=False)
rr, cc, ss = torch.where(selected)
pd.DataFrame(dict(date_beijing=dates.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d').to_numpy()[rr.cpu().numpy()],
                  symbol=np.array(symbols)[cc.cpu().numpy()],
                  side=np.where(ss.cpu().numpy()==0, 'long', 'short'),
                  weight=w[rr,cc].cpu().numpy())).to_parquet(OUT/'daily_holdings.parquet',index=False)
method = dict(model=str(MODEL), gpu=torch.cuda.get_device_name(), fee=.0005,
              selection='Keep long rank <= ceil(N*30%), short reverse rank <= ceil(N*30%); fill from top/bottom entry decile in score order; tie break symbol ascending',
              count='Same daily long/short count as original G10/G1, including integer rounding; shrink retains best ranked existing names',
              weighting='Daily equal weight within each side; long50%, short50% of post-fee equity; retained names also rebalance quantities',
              missing_data='Missing asset returns assumed0; missing price marks assumed unchanged; daily flags disclose estimates',
              costs='Both buy/sell .05%, including initial/final/cross-year; no funding or slippage',
              metrics='Daily net compounding; Sharpe sqrt365, sample std, risk-free0; Calmar annualized net compound / max drawdown',
              horizon='2022-01-02 to 2026-10-28, last holding2026-10-27')
(OUT/'method.json').write_text(json.dumps(method,ensure_ascii=False,indent=2))
print(pd.DataFrame(annual).to_string(index=False), flush=True)
print('Total net return:',nav[-1].item()-1,flush=True)
