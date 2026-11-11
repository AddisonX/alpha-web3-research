from datetime import datetime, timezone

import numpy as np
import polars as pl

from alpha_web3_research.config import Settings
from alpha_web3_research.factors.builder import FactorBuilder, GROUP_COLUMNS
from alpha_web3_research.factors.turnover import TURNOVER_FACTOR_NAMES, build_turnover_factors


def bars() -> pl.DataFrame:
    index = np.arange(240)
    close = np.exp(0.001 * index + 0.01 * np.sin(index / 5))
    return pl.DataFrame({
        "exchange": ["binanceusdm"] * len(index), "market_type": ["swap"] * len(index),
        "symbol": ["A/USDT:USDT"] * len(index), "timeframe": ["4h"] * len(index),
        "timestamp": index * 14_400_000, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": 100 + index * 0.3 + 15 * np.sin(index / 7),
        "fetched_at": [datetime.now(timezone.utc).isoformat()] * len(index), "source": ["test"] * len(index),
    })


def test_turnover_formulas_are_scale_invariant_and_use_disjoint_baselines():
    data = bars()
    original = build_turnover_factors(data, GROUP_COLUMNS)
    scaled = build_turnover_factors(data.with_columns(pl.col("volume") * 1000, pl.col("close") * 10), GROUP_COLUMNS)
    for name in TURNOVER_FACTOR_NAMES:
        assert np.allclose(original[name].to_numpy(), scaled[name].to_numpy(), equal_nan=True, atol=1e-8)
    volume = data["volume"].to_numpy()
    last = original.row(-1, named=True)
    assert np.isclose(last["turnover_relative_volume_42"], volume[-1] / volume[-43:-1].mean())
    assert np.isclose(last["turnover_relative_daily_42"], volume[-6:].sum() / (6 * volume[-48:-6].mean()))
    assert np.isclose(last["turnover_volume_cv_42"], volume[-42:].std(ddof=1) / volume[-42:].mean())
    assert np.isclose(last["turnover_volume_maxshare_42"], volume[-42:].max() / volume[-42:].sum())


def test_turnover_has_no_lookahead_and_resets_after_a_gap():
    data = bars()
    prefix = build_turnover_factors(data.head(220), GROUP_COLUMNS)
    complete = build_turnover_factors(data, GROUP_COLUMNS).head(220)
    assert prefix.equals(complete)
    gap = data.filter(pl.col("timestamp") != 200 * 14_400_000)
    result = build_turnover_factors(gap, GROUP_COLUMNS)
    first_after = result.filter(pl.col("timestamp") == 201 * 14_400_000).row(0, named=True)
    assert all(first_after[name] is None for name in TURNOVER_FACTOR_NAMES)
    assert result["turnover_relative_volume_42"][-1] is None
    assert result["turnover_relative_volume_6"][-1] is not None


def test_zero_volume_is_unavailable_and_symbols_are_independent():
    data = bars().with_columns(pl.lit(0.0).alias("volume"))
    combined = pl.concat([bars(), data.with_columns(pl.lit("B/USDT:USDT").alias("symbol"))])
    result = build_turnover_factors(combined, GROUP_COLUMNS)
    zero = result.filter(pl.col("symbol") == "B/USDT:USDT")
    assert all(zero[name].null_count() == zero.height for name in TURNOVER_FACTOR_NAMES)
    assert build_turnover_factors(data.head(0), GROUP_COLUMNS).is_empty()


def test_normal_factor_builder_exposes_turnover_without_changing_old_returns(tmp_path):
    data = bars()
    result = FactorBuilder(Settings(project_root=tmp_path))._build_factor_long_frame(data)
    turnover = result.filter(pl.col("factor_name").str.starts_with("turnover_"))
    assert turnover["factor_name"].n_unique() == 16
    assert turnover["factor_family"].unique().to_list() == ["turnover_proxy"]
    old = result.filter(pl.col("factor_name") == "return_1").sort("timestamp")
    expected = data["close"].to_numpy()[1:] / data["close"].to_numpy()[:-1] - 1
    assert np.allclose(old["factor_value"].to_numpy()[1:], expected)
