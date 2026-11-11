"""Official JQ Alpha191 OHLCV formula adaptations; 4h, percentile RANK and TSRANK.
Missing external inputs and materially ambiguous expressions are explicitly excluded.
Power expressions use monotonic logs solely to preserve ranks without overflow.
"""
import importlib.util
import torch
spec=importlib.util.spec_from_file_location('base','/workspace/alpha191_16_gpu.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
lag,roll,div,rank,tsrank,corr,age=b.lag,b.roll,b.div,b.rank,b.tsrank,b.corr,b.age
def rank(x,m=None):
 if m is None:m=torch.isfinite(x)
 return b.rank(x,m)/m.sum(1).clamp(min=1)[:,None]
def tsrank(x,n):return b.tsrank(x,n)/n

EXCLUDED={30:'Requires MKT/SMB/HML equity factors',75:'Requires benchmark index OHLC; official expression also truncated',149:'Requires benchmark index returns',181:'Requires benchmark index prices; denominator period omitted',182:'Requires benchmark index OHLC',54:'STD lookback omitted',127:'Outer MEAN lookback omitted',143:'SELF initial value/history origin unspecified',146:'Final SMA smoothing parameter omitted',165:'SUMAC and outer extrema windows/parentheses unspecified',166:'Malformed denominator and powers; cannot determine unique expression',183:'SUMAC and outer extrema windows/parentheses unspecified'}
NOTES={5:'YSRANK interpreted TSRANK',7:'MAX/MIN with integer interpreted rolling extrema',9:'Malformed division repaired to midpoint delta * range / volume',10:'MAX window5 applied to conditional squared series',17:'Export monotonic log of positive power: delta(close,5)*log(rank)',22:'SMEAN interpreted SMA(12,1)',28:'Literal published second denominator MAX(HIGH,9)-TSMAX(LOW,9), not silently changed to MIN(LOW,9)',31:'LOSE repaired CLOSE and missing opening parenthesis',36:'Misplaced periods interpreted RANK(SUM(CORR(RANK(volume),RANK(vwap),6),2))',52:'L interpreted LOW',69:'DTM/DBM use standard open-based directional movement',83:'COVIANCE interpreted sample covariance',99:'COVIANCE interpreted sample covariance',108:'Export negative log magnitude, monotonic in original negative power',115:'Export log of original positive power',116:'SEQUENCE(20), regression slope with intercept',121:'Export negative log magnitude, monotonic in original negative power',131:'DELAT repaired DELTA; export log of original power',159:'HGIH repaired HIGH; retain literal published SUM(low) numerator and third weight6*24',172:'HD=high-delay(high), LD=delay(low)-low, TR standard',188:'Corrupted separator interpreted subtraction',190:'Square deviation from geometric return threshold; parentheses normalized'}

@torch.jit.script
def sma(x:torch.Tensor,n:int,m:int):
 out=torch.empty_like(x);state=torch.full_like(x[0],float('nan'))
 for i in range(x.size(0)):
  state=torch.where(torch.isfinite(state),m/n*x[i]+(1-m/n)*state,x[i]);out[i]=state
 return out

def mean(x,n):return roll(x,n)
def summ(x,n):return roll(x,n,'sum')
def lo(x,n):return roll(x,n,'min')
def hi(x,n):return roll(x,n,'max')
def std(x,n):return roll(x,n,'std')
def delta(x,n=1):return x-lag(x,n)
def minimum(a,c):return torch.minimum(a,c) if isinstance(c,torch.Tensor) else a.clamp(max=c)
def maximum(a,c):return torch.maximum(a,c) if isinstance(c,torch.Tensor) else a.clamp(min=c)
def choose(cond,a,c,*required):
 z=torch.where(cond,a,c)
 for r in required:z=torch.where(torch.isfinite(r),z,torch.nan)
 return z
def decay(x,n):
 w=torch.arange(1,n+1,device=x.device,dtype=x.dtype);v=(x.unfold(0,n,1)*w).sum(-1)/w.sum()
 return torch.cat([torch.full_like(x[:n-1],torch.nan),v])
def wma(x,n):
 w=.9**torch.arange(n-1,-1,-1,device=x.device,dtype=x.dtype);v=(x.unfold(0,n,1)*w).sum(-1)/w.sum()
 return torch.cat([torch.full_like(x[:n-1],torch.nan),v])
def covariance(x,y,n):
 a=x.unfold(0,n,1);c=y.unfold(0,n,1);v=((a-a.mean(-1,keepdim=True))*(c-c.mean(-1,keepdim=True))).sum(-1)/(n-1)
 return torch.cat([torch.full_like(x[:n-1],torch.nan),v])
def slope(x,n):
 w=torch.arange(n,device=x.device,dtype=x.dtype);w=w-w.mean();v=(x.unfold(0,n,1)*w).sum(-1)/w.square().sum()
 return torch.cat([torch.full_like(x[:n-1],torch.nan),v])

class Formulas:
 def __init__(self,o,h,l,c,v,a):
  self.args=(o,h,l,c,v,a)
  pc=lag(c);po=lag(o);ph=lag(h);pl=lag(l);dc=c-pc;r=div(c,pc)-1;vw=div(a,v);mid=(h+l)/2;typ=(h+l+c)/3
  rg=div(2*c-h-l,h-l);up=choose(dc>0,dc,0.,dc);dn=choose(dc<0,-dc,0.,dc)
  udv=choose(dc>0,v,0.,dc,v);ddv=choose(dc<=0,v,0.,dc,v);signedv=choose(dc>0,v,choose(dc<0,-v,0.,dc),dc,v)
  move=c-choose(dc>0,minimum(l,pc),maximum(h,pc),dc);move=choose(dc==0,0.,move,dc)
  aa=(h-pc).abs();bb=(l-pc).abs();cc=(h-pl).abs();dd=(pc-po).abs()
  si=16*div(c-pc+(c-o)/2+pc-po,choose((aa>bb)&(aa>cc),aa+bb/2+dd/4,choose((bb>cc)&(bb>aa),bb+aa/2+dd/4,cc+dd/4,aa,bb,cc,dd),aa,bb,cc,dd))*maximum(aa,bb)
  mm=maximum((h-ph).abs(),(l-pl).abs());sumdn=summ(choose(h+l<ph+pl,mm,0.,h,l,ph,pl),12);sumup=summ(choose(h+l>ph+pl,mm,0.,h,l,ph,pl),12)
  k=100*div(c-lo(l,9),hi(h,9)-lo(l,9));w=100*div(hi(h,6)-c,hi(h,6)-lo(l,6))
  dtm=summ(choose(o>po,maximum(h-o,o-po),0.,o,po,h),20);dbm=summ(choose(o<po,maximum(o-l,o-po),0.,o,po,l),20)
  hd=h-ph;ld=pl-l;tr=maximum(h-l,maximum((h-pc).abs(),(l-pc).abs()))
  plus=summ(choose((hd>0)&(hd>ld),hd,0.,hd,ld),14);minus=summ(choose((ld>0)&(ld>hd),ld,0.,hd,ld),14)
  adx=mean(100*div((plus-minus).abs(),plus+minus),6)
  rsi=lambda n:100*div(sma(up,n,1),sma(dc.abs(),n,1))
  bias=lambda n:100*div(c-mean(c,n),mean(c,n))
  rec=sma(c,13,2)-sma(c,27,2);vrec=sma(v,13,2)-sma(v,27,2)
  f={
   1:lambda:-corr(rank(delta(v.log())),rank(div(c-o,o)),6),
   2:lambda:-delta(rg),3:lambda:summ(move,6),
   4:lambda:choose(mean(c,8)+std(c,8)<mean(c,2),-1.,choose(mean(c,2)<mean(c,8)-std(c,8),1.,choose(div(v,mean(v,20))>=1,1.,-1.,mean(v,20))),mean(c,8),std(c,8),mean(c,2)),
   5:lambda:-hi(corr(tsrank(v,5),tsrank(h,5),5),3),6:lambda:-rank(delta(.85*o+.15*h,4).sign()),
   7:lambda:(rank(hi(vw-c,3))+rank(lo(vw-c,3)))*rank(delta(v,3)),8:lambda:rank(-delta(.2*mid+.8*vw,4)),
   9:lambda:sma(div(delta(mid)*(h-l),v),7,2),10:lambda:rank(hi(choose(r<0,std(r,20),c,r).square(),5)),
   11:lambda:summ(rg*v,6),12:lambda:-rank(o-mean(vw,10))*rank((c-vw).abs()),13:lambda:(h*l).sqrt()-vw,
   14:lambda:delta(c,5),15:lambda:div(o,pc)-1,16:lambda:-hi(rank(corr(rank(v),rank(vw),5)),5),
   17:lambda:rank(vw-hi(vw,15)).log()*delta(c,5),18:lambda:div(c,lag(c,5)),19:lambda:div(delta(c,5),maximum(c,lag(c,5))),
   20:lambda:100*(div(c,lag(c,6))-1),21:lambda:slope(mean(c,6),6),22:lambda:sma(delta(div(c-mean(c,6),mean(c,6)),3),12,1),
   23:lambda:100*div(sma(choose(dc>0,std(c,20),0.,dc,std(c,20)),20,1),sma(choose(dc>0,std(c,20),0.,dc,std(c,20)),20,1)+sma(choose(dc<=0,std(c,20),0.,dc,std(c,20)),20,1)),
   24:lambda:sma(delta(c,5),5,1),25:lambda:-rank(delta(c,7)*(1-rank(decay(div(v,mean(v,20)),9))))*(1+rank(summ(r,250))),
   26:lambda:mean(c,7)-c+corr(vw,lag(c,5),230),27:lambda:wma(100*(div(c,lag(c,3))-1+div(c,lag(c,6))-1),12),
   28:lambda:3*sma(k,3,1)-2*sma(sma(100*div(c-lo(l,9),hi(h,9)-hi(l,9)),3,1),3,1),
   29:lambda:(div(c,lag(c,6))-1)*v,31:lambda:bias(12),32:lambda:-summ(rank(corr(rank(h),rank(v),3)),3),
   33:lambda:(lag(lo(l,5),5)-lo(l,5))*rank((summ(r,240)-summ(r,20))/220)*tsrank(v,5),34:lambda:div(mean(c,12),c),
   35:lambda:-minimum(rank(decay(delta(o),15)),rank(decay(corr(v,o,17),7))),36:lambda:rank(summ(corr(rank(v),rank(vw),6),2)),
   37:lambda:-rank(delta(summ(o,5)*summ(r,5),10)),38:lambda:choose(mean(h,20)<h,-delta(h,2),0.,mean(h,20),h),
   39:lambda:rank(decay(corr(.3*vw+.7*o,summ(mean(v,180),37),14),12))-rank(decay(delta(c,2),8)),
   40:lambda:100*div(summ(udv,26),summ(ddv,26)),41:lambda:-rank(hi(delta(vw,3),5)),42:lambda:-rank(std(h,10))*corr(h,v,10),
   43:lambda:summ(signedv,6),44:lambda:tsrank(decay(corr(l,mean(v,10),7),6),4)+tsrank(decay(delta(vw,3),10),15),
   45:lambda:rank(delta(.6*c+.4*o))*rank(corr(vw,mean(v,150),15)),46:lambda:div(mean(c,3)+mean(c,6)+mean(c,12)+mean(c,24),4*c),
   47:lambda:sma(w,9,1),48:lambda:-div(rank(dc.sign()+lag(dc).sign()+lag(dc,2).sign())*summ(v,5),summ(v,20)),
   49:lambda:div(sumdn,sumdn+sumup),50:lambda:div(sumup-sumdn,sumup+sumdn),51:lambda:div(sumup,sumup+sumdn),
   52:lambda:100*div(summ(maximum(h-lag(typ),0),26),summ(maximum(lag(typ)-l,0),26)),
   53:lambda:100*mean(choose(dc>0,1.,0.,dc),12),55:lambda:summ(si,20),
   56:lambda:choose(rank(o-lo(o,12))<rank(rank(corr(summ(mid,19),summ(mean(v,40),19),13))**5),1.,0.,o-lo(o,12),corr(summ(mid,19),summ(mean(v,40),19),13)),
   57:lambda:sma(k,3,1),58:lambda:100*mean(choose(dc>0,1.,0.,dc),20),59:lambda:summ(move,20),60:lambda:summ(rg*v,20),
   61:lambda:-maximum(rank(decay(delta(vw),12)),rank(decay(rank(corr(l,mean(v,80),8)),17))),62:lambda:-corr(h,rank(v),5),63:lambda:rsi(6),
   64:lambda:-maximum(rank(decay(corr(rank(vw),rank(v),4),4)),rank(decay(hi(corr(rank(c),rank(mean(v,60)),4),13),14))),
   65:lambda:div(mean(c,6),c),66:lambda:bias(6),67:lambda:rsi(24),68:lambda:sma(div(delta(mid)*(h-l),v),15,2),
   69:lambda:choose(dtm==dbm,0.,div(dtm-dbm,maximum(dtm,dbm)),dtm,dbm),70:lambda:std(a,6),71:lambda:bias(24),72:lambda:sma(w,15,1),
   73:lambda:rank(decay(corr(vw,mean(v,30),4),3))-tsrank(decay(decay(corr(c,v,10),16),4),5),
   74:lambda:rank(corr(summ(.35*l+.65*vw,20),summ(mean(v,40),20),7))+rank(corr(rank(vw),rank(v),6)),
   76:lambda:div(std(div(r.abs(),v),20),mean(div(r.abs(),v),20)),77:lambda:minimum(rank(decay(mid-vw,20)),rank(decay(corr(mid,mean(v,40),3),6))),
   78:lambda:div(typ-mean(typ,12),.015*mean((c-mean(typ,12)).abs(),12)),79:lambda:rsi(12),80:lambda:100*(div(v,lag(v,5))-1),81:lambda:sma(v,21,2),82:lambda:sma(w,20,1),
   83:lambda:-rank(covariance(rank(h),rank(v),5)),84:lambda:summ(signedv,20),85:lambda:tsrank(div(v,mean(v,20)),20)*tsrank(-delta(c,7),8),
   86:lambda:choose((lag(c,20)-2*lag(c,10)+c)/10>.25,-1.,choose((lag(c,20)-2*lag(c,10)+c)/10<0,1.,-dc),lag(c,20),lag(c,10),c),
   87:lambda:-rank(decay(delta(vw,4),7))-tsrank(decay(div(l-vw,o-mid),11),7),88:lambda:100*(div(c,lag(c,20))-1),89:lambda:2*(rec-sma(rec,10,2)),
   90:lambda:-rank(corr(rank(vw),rank(v),5)),91:lambda:-rank(c-hi(c,5))*rank(corr(mean(v,40),l,5)),
   92:lambda:-maximum(rank(decay(delta(.35*c+.65*vw,2),3)),tsrank(decay(corr(mean(v,180),c,13).abs(),5),15)),
   93:lambda:dbm,94:lambda:summ(signedv,30),95:lambda:std(a,20),96:lambda:sma(sma(k,3,1),3,1),97:lambda:std(v,10),
   98:lambda:choose(div(delta(mean(c,100),100),lag(c,100))<=.05,-(c-lo(c,100)),-delta(c,3),delta(mean(c,100),100),lag(c,100)),
   99:lambda:-rank(covariance(rank(c),rank(v),5)),100:lambda:std(v,20),
   101:lambda:choose(rank(corr(c,summ(mean(v,30),37),15))<rank(corr(rank(.1*h+.9*vw),rank(v),11)),-1.,0.,corr(c,summ(mean(v,30),37),15),corr(rank(.1*h+.9*vw),rank(v),11)),
   102:lambda:100*div(sma(maximum(delta(v),0),6,1),sma(delta(v).abs(),6,1)),103:lambda:5*(20-age(l,20,False)),
   104:lambda:-delta(corr(h,v,5),5)*rank(std(c,20)),105:lambda:-corr(rank(o),rank(v),10),106:lambda:delta(c,20),107:lambda:-rank(o-ph)*rank(o-pc)*rank(o-pl),
   108:lambda:-rank(h-lo(h,2)).log()*rank(corr(vw,mean(v,120),6)),109:lambda:div(sma(h-l,10,2),sma(sma(h-l,10,2),10,2)),
   110:lambda:100*div(summ(maximum(h-pc,0),20),summ(maximum(pc-l,0),20)),111:lambda:sma(v*rg,11,2)-sma(v*rg,4,2),
   112:lambda:100*div(summ(dc,12),summ(dc.abs(),12)),113:lambda:-rank(mean(lag(c,5),20))*corr(c,v,2)*rank(corr(summ(c,5),summ(c,20),2)),
   114:lambda:div(rank(lag(div(h-l,mean(c,5)),2))*rank(rank(v)),div(div(h-l,mean(c,5)),vw-c)),
   115:lambda:rank(corr(.9*h+.1*c,mean(v,30),10)).log()*rank(corr(tsrank(mid,4),tsrank(v,10),7)),
   116:lambda:slope(c,20),117:lambda:tsrank(v,32)*(1-tsrank(c+h-l,16))*(1-tsrank(r,32)),118:lambda:100*div(summ(h-o,20),summ(o-l,20)),
   119:lambda:rank(decay(corr(vw,summ(mean(v,5),26),5),7))-rank(decay(tsrank(lo(corr(rank(o),rank(mean(v,15)),21),9),7),8)),
   120:lambda:div(rank(vw-c),rank(vw+c)),121:lambda:-rank(vw-lo(vw,12)).log()*tsrank(corr(tsrank(vw,20),tsrank(mean(v,60),2),18),3),
   122:lambda:div(delta(sma(sma(sma(c.log(),13,2),13,2),13,2)),lag(sma(sma(sma(c.log(),13,2),13,2),13,2))),
   123:lambda:choose(rank(corr(summ(mid,20),summ(mean(v,60),20),9))<rank(corr(l,v,6)),-1.,0.,corr(summ(mid,20),summ(mean(v,60),20),9),corr(l,v,6)),
   124:lambda:div(c-vw,decay(rank(hi(c,30)),2)),125:lambda:div(rank(decay(corr(vw,mean(v,80),17),20)),rank(decay(delta(.5*c+.5*vw,3),16))),
   126:lambda:typ,128:lambda:self.moneyflow(typ,v),129:lambda:summ(dn,12),
   130:lambda:div(rank(decay(corr(mid,mean(v,40),9),10)),rank(decay(corr(rank(vw),rank(v),7),3))),
   131:lambda:rank(delta(vw)).log()*tsrank(corr(c,mean(v,50),18),18),132:lambda:mean(a,20),133:lambda:5*(age(l,20,False)-age(h,20,True)),
   134:lambda:(div(c,lag(c,12))-1)*v,135:lambda:sma(lag(div(c,lag(c,20))),20,1),136:lambda:-rank(delta(r,3))*corr(o,v,10),137:lambda:si,
   138:lambda:tsrank(decay(tsrank(corr(tsrank(l,8),tsrank(mean(v,60),17),5),19),16),7)-rank(decay(delta(.7*l+.3*vw,3),20)),
   139:lambda:-corr(o,v,10),140:lambda:minimum(rank(decay(rank(o)+rank(l)-rank(h)-rank(c),8)),tsrank(decay(corr(tsrank(c,8),tsrank(mean(v,60),20),8),7),3)),
   141:lambda:-rank(corr(rank(h),rank(mean(v,15)),9)),142:lambda:-rank(tsrank(c,10))*rank(delta(dc))*rank(tsrank(div(v,mean(v,20)),5)),
   144:lambda:div(summ(choose(r<0,div(r.abs(),a),0.,r),20),summ(choose(r<0,1.,0.,r),20)),145:lambda:100*div(mean(v,9)-mean(v,26),mean(v,12)),
   147:lambda:slope(mean(c,12),12),148:lambda:choose(rank(corr(o,summ(mean(v,60),9),6))<rank(o-lo(o,14)),-1.,0.,corr(o,summ(mean(v,60),9),6),o-lo(o,14)),
   150:lambda:typ*v,151:lambda:sma(delta(c,20),20,1),152:lambda:sma(mean(lag(sma(lag(div(c,lag(c,9))),9,1)),12)-mean(lag(sma(lag(div(c,lag(c,9))),9,1)),26),9,1),
   153:lambda:(mean(c,3)+mean(c,6)+mean(c,12)+mean(c,24))/4,
   154:lambda:choose(vw-lo(vw,16)<corr(vw,mean(v,180),18),1.,0.,vw-lo(vw,16),corr(vw,mean(v,180),18)),155:lambda:vrec-sma(vrec,10,2),
   156:lambda:-maximum(rank(decay(delta(vw,5),3)),rank(decay(-div(delta(.15*o+.85*l,2),.15*o+.85*l),3))),
   157:lambda:lo(rank(rank(lo(rank(rank(-rank(delta(c-1,5)))),2).log())),5)+tsrank(lag(-r,6),5),
   158:lambda:div(h-l,c),159:lambda:100/(6*12+6*24+12*24)*(div(c-summ(minimum(l,pc),6),summ(maximum(h,pc)-minimum(l,pc),6))*12*24+div(c-summ(minimum(l,pc),12),summ(maximum(h,pc)-minimum(l,pc),12))*6*24+div(c-summ(minimum(l,pc),24),summ(maximum(h,pc)-minimum(l,pc),24))*6*24),
   160:lambda:sma(choose(dc<=0,std(c,20),0.,dc,std(c,20)),20,1),161:lambda:mean(tr,12),162:lambda:div(rsi(12)-lo(rsi(12),12),hi(rsi(12),12)-lo(rsi(12),12)),
   163:lambda:rank(-r*mean(v,20)*vw*(h-c)),164:lambda:sma(100*div(choose(dc>0,div(torch.ones_like(dc),dc),1.,dc)-lo(choose(dc>0,div(torch.ones_like(dc),dc),1.,dc),12),h-l),13,2),
   167:lambda:summ(up,12),168:lambda:-div(v,mean(v,20)),169:lambda:sma(mean(lag(sma(dc,9,1)),12)-mean(lag(sma(dc,9,1)),26),10,1),
   170:lambda:div(rank(1/c)*v,mean(v,20))*div(h*rank(h-c),mean(h,5))-rank(delta(vw,5)),171:lambda:-div(l-c,c-h)*(div(o,c)**5),
   172:lambda:adx,173:lambda:3*sma(c,13,2)-2*sma(sma(c,13,2),13,2)+sma(sma(sma(c.log(),13,2),13,2),13,2),
   174:lambda:sma(choose(dc>0,std(c,20),0.,dc,std(c,20)),20,1),175:lambda:mean(tr,6),176:lambda:corr(rank(div(c-lo(l,12),hi(h,12)-lo(l,12))),rank(v),6),177:lambda:5*(20-age(h,20,True)),
   178:lambda:r*v,179:lambda:rank(corr(vw,v,4))*rank(corr(rank(l),rank(mean(v,50)),12)),
   180:lambda:choose(mean(v,20)<v,-tsrank(delta(c,7).abs(),60)*delta(c,7).sign(),-v,mean(v,20),v,delta(c,7)),
   184:lambda:rank(corr(lag(o-c),c,200))+rank(o-c),185:lambda:rank(-(1-div(o,c)).square()),186:lambda:(adx+lag(adx,6))/2,187:lambda:dtm,
   188:lambda:100*div(h-l-sma(h-l,11,2),sma(h-l,11,2)),189:lambda:mean((c-mean(c,6)).abs(),6),190:lambda:self.logratio(r,div(c,lag(c,19))**.05-1),191:lambda:corr(mean(v,20),l,5)+mid-c
  }
  self.functions=f
  assert set(f)|set(EXCLUDED)==set(range(1,192)) and not set(f)&set(EXCLUDED)
 @staticmethod
 def moneyflow(typ,v):
  d=delta(typ);p=summ(choose(d>0,typ*v,0.,d,v),14);n=summ(choose(d<0,typ*v,0.,d,v),14)
  return 100*div(p,p+n)
 @staticmethod
 def logratio(r,t):
  d=(r-t).square();up=summ(choose(r>t,1.,0.,r,t),20);dn=summ(choose(r<t,1.,0.,r,t),20)
  a=summ(choose(r<t,d,0.,r,t),20);c=summ(choose(r>t,d,0.,r,t),20);q=div((up-1)*a,dn*c)
  return torch.where(q>0,q.log(),torch.nan)
