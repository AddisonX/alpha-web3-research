"""Make the liquidity audit readable and package its auditable outputs."""
import json
import zipfile
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/small_cap_liquidity_100k_cny'
summary=pd.read_csv(OUT/'summary.csv')
overall=summary.loc[summary.period=='2021-2025'].copy()
compact=pd.DataFrame({'平均市值窗口':overall.factor.str.extract(r'(\d+)d')[0]+'天',
                      '持仓币数范围':overall.members_min.astype(str)+'—'+overall.members_max.astype(str),
                      '4小时成交额中位数（万USDT）':overall.median_4h_quote_usdt/10000,
                      '4小时成交额最弱1%分位（万USDT）':overall.p01_4h_quote_usdt/10000,
                      '净订单/4小时成交额P99':overall.actual_order_4h_participation_p99.map(lambda x:f'{x:.3%}'),
                      '净订单/4小时成交额最大值':overall.actual_order_4h_participation_max.map(lambda x:f'{x:.3%}')})
orders=pd.read_csv(OUT/'targeted_minute_order_checks.csv')
positions=pd.read_csv(OUT/'targeted_minute_full_position_checks.csv')
examples=[]
for symbol,day in [('TOMO/USDT:USDT','2023-01-01'),('SYS/USDT:USDT','2025-12-01'),('HIPPO/USDT:USDT','2025-03-19')]:
    r=orders.loc[orders.symbol.eq(symbol)&orders.date_beijing.eq(day)].sort_values('trade_usdt',ascending=False).iloc[0]
    examples.append({'日期（北京时间）':day,'合约':symbol,'场景':'换入' if r.side=='buy' else '换出',
                     '订单（USDT）':r.trade_usdt,'零点后1分钟成交额':r.quote_1min_usdt,
                     '零点后5分钟成交额':r.quote_5min_usdt,'订单/5分钟成交额':f'{r.participation_actual_5min:.1%}'})
r=positions.loc[positions.symbol.eq('1000WHY/USDT:USDT')&positions.date_beijing.eq('2025-11-16')].iloc[0]
examples.append({'日期（北京时间）':r.date_beijing,'合约':r.symbol,'场景':'假设整仓建/平，非实际净调仓',
                 '订单（USDT）':r.target_usdt,'零点后1分钟成交额':r.quote_1min_usdt,
                 '零点后5分钟成交额':r.quote_5min_usdt,'订单/5分钟成交额':f'{r.full_position_participation_actual_5min:.1%}'})
examples=pd.DataFrame(examples)
examples.to_csv(OUT/'关键薄弱时段.csv',index=False)
cost=pd.DataFrame({'平均市值窗口':overall.factor.str.extract(r'(\d+)d')[0]+'天',
                   '累计超额毛收益':overall.gross_g1_excess_simple.map(lambda x:f'{x:.2%}'),
                   '单边综合成本10bp':overall.g1_excess_minus_10bp_one_way_cost.map(lambda x:f'{x:.2%}'),
                   '单边综合成本50bp':overall.g1_excess_minus_50bp_one_way_cost.map(lambda x:f'{x:.2%}')})
old=(OUT/'小市值G1流动性检查.html').read_text(encoding='utf-8')
minute_note=json.loads((OUT/'minute_method.json').read_text())
new='<meta charset="utf-8"><title>小市值G1交易容量与滑点风险</title><style>body{font:15px system-ui;margin:30px auto;max-width:1500px;padding:0 20px;color:#203047}p{line-height:1.8}td,th{padding:9px;border-bottom:1px solid #ccc;text-align:right}table{border-collapse:collapse;width:100%}.table{overflow:auto}a{color:#155bb2}</style><h1>10万元 · 1倍名义仓位 · 小市值G1执行风险检查</h1>'
new+='<p><strong>结论：多数日常小额再平衡的4小时成交参与率较低，但存在明显短时流动性尾部风险，不能把零点开盘价视为每个币都能一次成交的价格。</strong>2025 年 G1 的成交额明显低于更早年份。应进一步把执行时长、盘口深度约束、限价未成交和资金费纳入可交易回测。</p>'
new+='<p>全资金等权投入 G1，约8—30个币，每币约3333—12500元；固定7元人民币/USDT仅作换算，不是历史实际汇率。金额按固定名义本金，不模拟复利。日常订单按目标金额减去上期持仓按当前价格计值计算，含换币、价格漂移后的再平衡和最终平仓。</p>'
new+='<h2>四个窗口整体容量</h2><div class="table">'+compact.to_html(index=False,float_format=lambda x:f'{x:,.2f}')+'</div>'
new+='<p>这里的4小时是北京时间00:00—04:00已实现成交额，属于事后统计，不是盘口挂单深度，也不是零点事先知道的量。P99表示99%的实际净订单参与率不超过该值；不会据此承诺市价单滑点。</p>'
new+='<h2>分钟数据确认的薄弱时段</h2><div class="table">'+examples.to_html(index=False,float_format=lambda x:f'{x:,.2f}')+'</div>'
new+=f'<p>只对最弱尾部有目的地补查了 {minute_note["available_symbol_dates"]} 个合约日期，全部取得官方1分钟归档；它们不代表全样本频率。1分钟累计240根的成交额已与原4小时成交额逐笔核对一致。参与率分母是未加入本策略订单的历史成交额，超过100%并不表示订单物理上无法成交，而是相对原有成交流很大。</p>'
new+='<p>成交额只能做容量筛查。要估计市价单的滑点bp，需要当时的买卖价差和各档盘口深度；本次没有这些历史数据。尤其不能把“订单占5分钟成交额26%”读成“滑点26%”。<a href="https://www.binance.com/en/academy/glossary/slippage">Binance 对滑点与深度的说明</a></p>'
new+='<h2>成本敏感性，不是实际滑点预测</h2><div class="table">'+cost.to_html(index=False)+'</div><p>假设固定单边综合成本，按实际模拟订单金额计费，基准仍是全币池毛收益。它不能刻画薄盘口的非线性冲击、拆单期间价格变化或限价未成交，不能据此把毛收益认定为可实现收益。</p>'
new+='<details><summary>展开原始逐币、订单和指标明细</summary>'+old+'</details>'
(OUT/'小市值G1流动性检查.html').write_text(new,encoding='utf-8')
(OUT/'数据说明.md').write_text('''# 小市值G1执行风险

本目录检查10/20/30/40天平均市值因子的G1，2021—2025年，含2026-01-01最终平仓。
本金10万元人民币、1倍总名义敞口，全资金等权投向G1，固定7元人民币/USDT进行金额换算。
均为历史回看，币池存在事后筛选偏差。持仓与之前因子报告的G1成员数及毛收益已核对一致。

summary.csv：逐窗口、逐年和整体成交额、参与率与固定成本敏感性。
g1_holdings_liquidity.parquet：逐笔目标仓位与完整建仓容量；它不是每天真实下单额。
rebalance_orders.parquet：目标金额减旧持仓当前价格计值后的净订单，含换入换出及最终平仓。
coin_liquidity.csv：逐币流动性尾部；同一日期会因四个因子重复出现，holding_dates已去重。
targeted_minute_order_checks.csv：极弱订单样本的真实1/5/15分钟成交参与率。
targeted_minute_full_position_checks.csv：同一批时段的完整目标仓位容量，非实际净调仓额。
targeted_minute_volume.csv：30个定向抽查合约日期的原始1/5/15/240分钟成交额及源URL/SHA256。
minute_method.json：分钟抽样与校验说明；method.json：完整假设和限制。

分钟数据是刻意抽取的极弱样本，不可据此推断全样本薄弱时段比例。
4小时、1/5/15分钟的本期成交额均为事后数据，不可当作调仓时已知过滤条件。
prev4h_quote、prev24h_quote、prev7d_*仅使用决策前收盘的K线，可以用于后续设计历史过滤规则。
带uniform_proxy的短时指标是4小时成交额均匀折算，不是已观测到的分钟成交额。
1%/5%只是筛查阈值，并不保证任何给定滑点。
所有参与率都基于不含本策略订单的历史市场成交额，不是买卖盘口深度。
没有历史订单簿，不给出伪精确的滑点bp。确切执行还需盘口、排队、拆单时序和手续费/资金费。
固定10/50bp成本敏感性只是情景假设，对比的基准仍为毛收益，不代表真实可实现超额。
''',encoding='utf-8')
bundle=OUT.parent/'小市值G1_10万元1倍_流动性检查.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            z.write(p,'small_cap_liquidity_100k_cny/'+p.name)
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None
print('report and bundle complete',round(bundle.stat().st_size/1024**2,2),'MB')
print(examples.to_string(index=False))
