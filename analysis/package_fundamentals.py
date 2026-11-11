"""Validate and package the acquired public fundamentals without raw caches."""
import json
import zipfile
from pathlib import Path
import polars as pl

out = Path('/workspace/alpha-web3-research/data/fundamentals/cmc_2021_2025')
df = pl.read_parquet(out / 'contract_daily.parquet')
assert df.height == 341 * 1826
assert df.select(pl.struct('contract', 'date').n_unique()).item() == df.height
assert df.filter(pl.col('has_market_cap_and_supply') & (pl.col('market_cap_usd').is_null() | pl.col('circulating_supply_tokens').is_null())).is_empty()
assert df.filter(pl.col('usable_not_before_utc').is_not_null() & (pl.col('usable_not_before_utc') <= pl.col('period_end_utc'))).is_empty()
missing = df.filter(pl.col('in_research_pool') & ~pl.col('has_market_cap_and_supply') & (pl.col('status') != 'not_applicable_index_contract'))
missing.write_csv(out / 'research_missing_days.csv')
review = df.filter(pl.col('supply_unit_review_required')).select('contract').unique().sort('contract').to_series().to_list()
validation = {'grid_rows': df.height, 'unique_contract_dates': True, 'positive_fields_consistent': True,
              'availability_after_period_end': True, 'research_missing_token_days': missing.height,
              'unit_review_contracts': review,
              'identity_error_over_1pct_rows': df.filter(pl.col('marketcap_identity_error') > .01).height}
(out / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2))
(out / '数据说明.md').write_text('''# 2021—2025 年历史市值与流通量（日频）

范围为五组年度币池的 341 个合约标识，337 个代币合约、4 个指数合约。
`contract_daily.parquet` 是主表，`contract_daily.csv.gz` 是相同数据的压缩 CSV。
完整日期网格包含上市前空值；不能用全日期网格覆盖率衡量年度研究所需覆盖率。
按 `in_research_pool=true` 取各年度实际研究范围，`research_pool_year` 为对应选币年份。
例如 2022 币池对应 2021 数据，2026 币池对应 2025 数据。

来源为 CoinMarketCap 公开历史页面及其公开数据响应，无付费 API。
示例 https://coinmarketcap.com/historical/20210101/
采集 URL、抓取时间随行保存，原始响应缓存另存于 Runpod 的 raw 目录。
这是目前查询到的历史序列，供应商可能修订历史，不能保证为当时发布版本。

主要字段：market_cap_usd 为流通市值（美元）；circulating_supply_tokens 为选中供应商资产版本的流通量；
price_usd_per_token 为该版本价格；contract_multiplier 为合约倍数；
contract_equivalent_supply 为流通量除以合约倍数，可与倍数合约成交数量保持量纲一致。
流通量直接来自供应商 circulatingSupply 字段，未用市值除以价格估算，未用总量替代。
1000SATS 的两种供应商报价单位已归一为 SATS 原生单位。

重要限制：SUN、BNX、PHB、EOS、LIT、KNC、DYDX 存在多个供应商资产版本。
`cmc_id`、provider_symbol/name、asset_version_mapping 保留实际版本，市值没有相加。
供应商可能把替代代币单位回溯到实际迁移之前，当前流通量不保证是当日交易合约的单位。
这些合约的 `supply_unit_review_required=true`，在逐日核对迁移边界、换算比例之前，
不应直接用于成交数量/流通量换手因子。主表的有值覆盖率仅表示供应商报告了数值，并非全部可直接回测。
BNX 官方换算为 1:100，SUN 为 1:1000，PHB 为 100:1，不能直接拼接不同版本。
供应商各资产版本的原始数值另存 provider_assets_daily.parquet，便于进一步审计。

日期为 UTC 日终。usable_not_before_utc 是下一天 UTC 00:00，仅为最早使用时间下界，
不是经核实的首次公开发布时间。北京时间每天 00:00 调仓必须按该时间做向后 as-of 对齐，
不能把同名日期的 UTC 日终数据用于当天北京时间 00:00。稳健回测还应增加发布延迟。

空值不插值、不前后填充、不用今天的流通量回填。供应商零值在原始供应商表保留，主表转为空。
1000RATS 在研究所需 2024—2025 年缺失 731 天；KMNO 在 2025-01-01—05-05 缺失 125 天；
NEAR 在 2021-01-20 缺失 1 天。缺失明细见 research_missing_days.csv。
BLUEBIRD、BTCDOM、DEFI、FOOTBALL 是指数合约，无单一代币流通量，标记为不适用。

coverage.csv 为逐币覆盖率，yearly_coverage.csv 为逐年实际研究范围覆盖率，
asset_mapping.json/csv 为映射，summary.json 为统计，validation.json 为验证结果。
Runpod 原始缓存不包含在下载包中，避免重复打包大量响应。
''', encoding='utf-8')
archive = out.parent / 'cmc_2021_2025_public.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
    for path in sorted(out.iterdir()):
        if path.is_file():
            z.write(path, arcname='cmc_2021_2025/' + path.name)
print(json.dumps(validation, ensure_ascii=False, indent=2))
print(archive)
