from pathlib import Path
import pandas as pd,json
r=Path('/tmp/alpha191_all_pct');c=pd.read_csv(r/'coverage_and_ties.csv');a=pd.read_csv(r/'all_factor_screen.csv');a['selected_ic_aligned']=a.pass_IC & ((a.long_positive_years>=4)|(a.short_negative_years>=4));a['selected']=a.any_fixed_endpoint_selected;a.to_csv(r/'all_factor_screen.csv',index=False);s=a.loc[a.selected].copy()
s['qualifying_legs']=s.apply(lambda x:';'.join(f'{g}_{side}:{int(x[col])}/5' for g in ['G1','G10'] for side,col in [('long',g+'_positive_years'),('short',g+'_negative_years')] if x[col]>=4),axis=1)
s['strong_side']=s.qualifying_legs.apply(lambda x:'both' if 'long' in x and 'short' in x else ('long' if 'long' in x else 'short'))
s['min_annual_IC_days']=s.factor.map(c.groupby('factor').IC_days.min());s['min_annual_decile_days']=s.factor.map(c.groupby('factor').decile_days.min());s['mean_unique_values']=s.factor.map(c.groupby('factor').mean_unique_values.mean())
s.to_csv(r/'selected_factors.csv',index=False)
print(s[['factor','RankIC','qualifying_legs','strong_side','min_annual_decile_days','mean_unique_values']].to_string(index=False))
print('COUNTS',len(s),'SIDES',s.strong_side.value_counts().to_dict(),'IC',int(a.pass_IC.sum()),'ANY',int(a.any_fixed_endpoint_selected.sum()))
rows=[]
for _,x in s.iterrows():
 for y in range(2021,2026):rows.append(dict(factor=x.factor,year=y,RankIC=x[f'IC_{y}'],G1_excess=x[f'G1_{y}'],G10_excess=x[f'G10_{y}'],long_group=x.long_group,short_group=x.short_group,long_excess=x[f'G{int(x.long_group)}_{y}'],short_underlying_excess=x[f'G{int(x.short_group)}_{y}']))
pd.DataFrame(rows).to_csv(r/'selected_annual_details.csv',index=False)

m=json.loads((r/'method.json').read_text());m['selection']='abs(overall RankIC)>0.03; independently test fixed G1-long,G1-short,G10-long,G10-short; at least one leg correct sign in>=4years; no annual direction flipping, no requirement to agree with IC sign';(r/'method.json').write_text(json.dumps(m,indent=2))
