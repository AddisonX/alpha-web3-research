"""Small deterministic checks of CUDA operators and causality, not a CPU backtest."""
import importlib.util
import torch
spec=importlib.util.spec_from_file_location('a','/workspace/alpha191_16_gpu.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
x=torch.tensor([[2.,1.,2.,float('nan')]],device='cuda')
assert torch.allclose(a.rank(x)[0,:3],torch.tensor([2.5,1.,2.5],device='cuda'))
g=torch.Generator(device='cuda').manual_seed(42)
c=100+torch.randn((300,4),generator=g,device='cuda').cumsum(0)
o=c*(1+.002*torch.randn((300,4),generator=g,device='cuda'))
h=torch.maximum(o,c)+1;l=torch.minimum(o,c)-1
v=100+20*torch.rand((300,4),generator=g,device='cuda');am=c*v
f=a.factors(o,h,l,c,v,am);p=a.factors(o[:250],h[:250],l[:250],c[:250],v[:250],am[:250])
assert torch.allclose(f[:250],p,equal_nan=True), 'Future data affected factor values'
assert torch.allclose(f[150:,:,14],-v[150:]/a.roll(v,20)[150:])
assert (torch.isfinite(f[150:]).double().mean((0,1))>.8).all()
# A constant rank sequence has undefined correlation, correctly left missing.
for j in [0,2]:
 z=f[150:,:,j];z=z[torch.isfinite(z)];assert (z.abs()<=1.000001).all()
for j in [4,7,10,15]:
 z=f[150:,:,j];z=z[torch.isfinite(z)];assert ((z>=0)&(z<=100.000001)).all()
z=f[150:,:,9];z=z[torch.isfinite(z)];assert (z.abs()<=100.000001).all()
print('PASS CUDA ranks, causal prefix, relative-volume identity and factor bounds',flush=True)
