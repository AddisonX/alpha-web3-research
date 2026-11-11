import torch,numpy as np,pandas as pd
import formulas_gpu as f
torch.manual_seed(42);torch.set_default_dtype(torch.float64)
T,N=420,24
c=torch.exp(torch.randn(T,N,device='cuda')*.02).cumprod(0)*10
o=c*torch.exp(torch.randn(T,N,device='cuda')*.01);h=torch.maximum(o,c)*1.01;l=torch.minimum(o,c)*.99;v=torch.exp(torch.randn(T,N,device='cuda'))*100;a=(h+l+c)/3*v
calc=f.Formulas(o,h,l,c,v,a)
for n,fn in sorted(calc.functions.items()):
 x=fn();assert x.shape==(T,N),(n,x.shape);assert torch.isfinite(x[-20:]).any(),n
print('179 formulas execute; correct shapes; recent finite values',flush=True)
pct_rank,pct_tsrank=f.rank,f.tsrank
f.rank,f.tsrank=f.b.rank,f.b.tsrank
old=f.b.factors(o,h,l,c,v,a)
for j,name in enumerate(f.b.NAMES):
 n=int(name[-3:]);new=calc.functions[n]();ref=old[:,:,j];mask=torch.isfinite(ref)
 assert torch.allclose(new[mask],ref[mask],atol=1e-9,rtol=1e-9),(n,float((new[mask]-ref[mask]).abs().max()))
print('All16 match old implementation under identical rank convention',flush=True)
f.rank,f.tsrank=pct_rank,pct_tsrank
x=torch.tensor([[1.,1.,3.],[2.,3.,2.],[4.,2.,2.],[3.,5.,4.]],device='cuda')
assert np.allclose(f.rank(x).cpu(),pd.DataFrame(x.cpu().numpy()).rank(axis=1,pct=True).values)
assert np.allclose(f.std(x,3)[2:].cpu(),pd.DataFrame(x.cpu().numpy()).rolling(3).std().values[2:])
assert torch.allclose(f.slope(torch.arange(20,device='cuda').double()[:,None]*3+2,6)[5:],torch.full((15,1),3.,device='cuda'))
assert torch.allclose(f.decay(torch.arange(5,device='cuda').double()[:,None],3)[2:,0],torch.tensor([8/6,14/6,20/6],device='cuda'))
print('Rank ties, sample STD, regression slope and decay reference checks passed',flush=True)
