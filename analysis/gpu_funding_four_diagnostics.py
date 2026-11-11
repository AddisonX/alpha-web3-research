import sys
sys.path.append('/workspace/alpha-web3-research/.venv/lib/python3.12/site-packages')
from pathlib import Path
import torch,pandas as pd,numpy as np,json
ROOT=Path('/tmp/funding_four_factors')
names=json.loads((ROOT/'feature_columns.json').read_text())
rows=[];metrics=[];coverage=[];assert torch.cuda.is_available()
def average_rank(a,mask):
    order=torch.argsort(torch.where(mask,a,torch.inf),dim=1,stable=True)
    vals=a.gather(1,order);valid=mask.gather(1,order)
    n=a.shape[1];pos=torch.arange(n,device='cuda')[None,:].expand_as(a)
    change=torch.ones_like(valid);change[:,1:]=vals[:,1:]!=vals[:,:-1]
    starts=torch.cummax(torch.where(change,pos,0),1).values
    endchange=torch.ones_like(valid);endchange[:,:-1]=vals[:,:-1]!=vals[:,1:]
    ends=torch.flip(torch.cummin(torch.flip(torch.where(endchange,pos,n-1),[1]),1).values,[1])
    ranked=(starts+ends).to(torch.float64)/2+1
    out=torch.empty_like(a).scatter_(1,order,ranked)
    return torch.where(mask,out,0)
def corr(a,b,mask):
    n=mask.sum(1).clamp(min=1)
    aa=torch.where(mask,a,0);bb=torch.where(mask,b,0)
    ac=torch.where(mask,aa-aa.sum(1)[:,None]/n[:,None],0)
    bc=torch.where(mask,bb-bb.sum(1)[:,None]/n[:,None],0)
    return (ac*bc).sum(1)/torch.sqrt(ac.square().sum(1)*bc.square().sum(1))
for year in range(2022,2027):
    df=pd.read_parquet(ROOT/f'funding_table_{year}.parquet').sort_values(['entry_time_utc','symbol'])
    n=df.symbol.nunique();d=df.entry_time_utc.nunique();assert len(df)==n*d
    def tensor(col):return torch.tensor(df[col].to_numpy().reshape(d,n),device='cuda',dtype=torch.float64)
    y=tensor('realized_return_24h');cost=tensor('funding_long_cost_24h');px=tensor('entry_open')
    known=tensor('pool_known_at_decision').bool();dates=df.drop_duplicates('entry_time_utc').decision_date_beijing.to_numpy()
    fullpool=torch.isfinite(y)&torch.isfinite(cost)
    bprice=torch.where(fullpool,y,0).sum(1)/fullpool.sum(1).clamp(min=1)
    bfund=torch.where(fullpool,y-cost,0).sum(1)/fullpool.sum(1).clamp(min=1)
    for name in names:
        x=tensor(name);mask=torch.isfinite(x)&known&torch.isfinite(px)&px.gt(0)
        signal_mask=mask.clone()
        signal_members=signal_mask.sum(1)
        signal_ranks=average_rank(x,signal_mask)
        df[name+'_pct']=torch.where(signal_mask,signal_ranks/signal_members.clamp(min=1)[:,None],torch.nan).cpu().numpy().reshape(-1)
        mask=signal_mask&fullpool
        members=mask.sum(1)
        valid=members>=10
        coverage.append(pd.DataFrame(dict(year=year,factor=name,date_beijing=dates,signal_members=signal_members.cpu().numpy(),evaluated_members=members.cpu().numpy(),pool_members=n,scored=valid.cpu().numpy())))
        ic=corr(x,y,mask);fic=corr(x,y-cost,mask)
        ranks=average_rank(x,mask)
        pct=torch.where(mask,ranks/members.clamp(min=1)[:,None],torch.nan)
        # Exported signal percentiles above never use future return availability.
        ric=corr(ranks,average_rank(y,mask),mask)
        fric=corr(ranks,average_rank(y-cost,mask),mask)
        order=torch.argsort(torch.where(mask,x,torch.inf),dim=1,stable=True)
        sy=torch.nan_to_num(y,nan=0).gather(1,order);sf=torch.nan_to_num(y-cost,nan=0).gather(1,order)
        py=torch.cat([torch.zeros((d,1),device='cuda'),sy.cumsum(1)],1)
        pf=torch.cat([torch.zeros((d,1),device='cuda'),sf.cumsum(1)],1)
        q=members//10;rem=members%10
        ok=valid.cpu().numpy()
        rec=dict(year=year,factor=name,eligible_days=int((signal_members>=10).sum()),scored_days=int(valid.sum()),mean_members=members[valid].double().mean().item())
        rec['first_scored_date']=dates[ok][0] if ok.any() else None
        rec['last_scored_date']=dates[ok][-1] if ok.any() else None
        for label,values in [('IC',ic),('funding_IC',fic),('RankIC',ric),('funding_RankIC',fric)]:
            good=values[valid&torch.isfinite(values)]
            rec[label]=good.mean().item() if len(good) else None
            rec[label+'IR']=(good.mean()/good.std(correction=1)).item() if len(good)>1 else None
            rec[label+'_days']=len(good)
        for g in range(10):
            start=g*q+torch.minimum(rem,torch.full_like(rem,g));sz=q+(rem>g);end=start+sz
            price=(py.gather(1,end[:,None]).squeeze(1)-py.gather(1,start[:,None]).squeeze(1))/sz.clamp(min=1)
            funded=(pf.gather(1,end[:,None]).squeeze(1)-pf.gather(1,start[:,None]).squeeze(1))/sz.clamp(min=1)
            pe=price-bprice;fe=funded-bfund
            rows.append(pd.DataFrame(dict(year=year,factor=name,decile=g+1,date_beijing=dates[ok],price_excess=pe.cpu().numpy()[ok],funding_adjusted_excess=fe.cpu().numpy()[ok],raw_price_return=price.cpu().numpy()[ok],raw_funding_adjusted_return=funded.cpu().numpy()[ok])))
            rec[f'G{g+1}_price_excess']=pe[valid].sum().item();rec[f'G{g+1}_funding_excess']=fe[valid].sum().item()
        metrics.append(rec)
    df.to_parquet(ROOT/f'funding_table_{year}_with_percentiles.parquet',index=False)
    print('GPU year complete',year,flush=True)
pd.concat(coverage,ignore_index=True).to_csv(ROOT/'daily_evaluation_coverage.csv',index=False)
pd.concat(rows,ignore_index=True).to_parquet(ROOT/'decile_daily.parquet',index=False)
a=pd.DataFrame(metrics);a.to_csv(ROOT/'annual_metrics.csv',index=False)
print(a[['factor','year','scored_days','IC','funding_IC','RankIC','funding_RankIC']].to_string(index=False),flush=True)
(ROOT/'method.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),features=names,
    signals='Latest settled rate perhour; sum of actual settled rates in preceding3/7days; preceding1day sum/24 minus7day sum/168. Settlement at decision00:00 excluded.',
    funding_cashflow='Long pays positive rate, short receives. Settlements in(t,t+24h] assigned to holdings entered at t; same-time settlement before rebalance. Rate multiplied by1h mark-price bar open / entry trade price.',
    approximation='Settlement mark approximated by nominal settlement-hour mark-kline open, not individual-account execution or exact matching-engine settlement snapshot.',
    benchmark='On each date, equalweight annual-pool members with both valid price return and funding cashflow; same benchmark population for two panels.',
    grouping='Daily available-case diagnostic: finite factor, current entry price, known pool, forward price return and funding cashflow. Stable-symbol ties;10 equalcount groups. Exported signal percentiles use signal-only eligibility, never future labels.',
    evaluation='Evaluate available coins each day; drop only individual missing observations, never an entire date due to one missing coin. At least10 valid coins for10 nonempty bins. Both panels use the same available coins. Daily coverage exported; no missing costs imputed. Available-case diagnostics are not executable backtests.',
    plots='Cumulative simple excess returns for price-only vs price-minus-funding; diagnostic curves before trading fees, not net executable long-short backtests.',
    limits='Historical final settled rates; actual publication lag unavailable. Strictly excludes simultaneous settlement. Funding interval transitions with incomplete lookback coverage excluded.'),indent=2))
