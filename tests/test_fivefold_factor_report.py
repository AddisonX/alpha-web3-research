import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.stats import pearsonr, spearmanr


SPEC = importlib.util.spec_from_file_location(
    "fivefold_report", Path(__file__).resolve().parents[1] / "scripts/build_fivefold_factor_report.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def frame():
    return pl.DataFrame({
        "fold": [1] * 30, "split": ["train"] * 30, "timestamp": [0] * 30,
        "symbol": [f"S{i:02}" for i in range(30)],
        "factor": np.arange(30, dtype=float), "forward_return": np.arange(30, dtype=float) ** 3 / 100000,
    }).with_columns(pl.col("forward_return").mean().alias("universe_return"))


def test_true_spearman_not_pearson():
    f = frame()
    row = report.cross_sections(f, "factor").row(0, named=True)
    assert row["ic"] == pytest.approx(pearsonr(f["factor"], f["forward_return"]).statistic)
    assert row["rank_ic"] == pytest.approx(spearmanr(f["factor"], f["forward_return"]).statistic)
    assert row["rank_ic"] > row["ic"]


def test_constant_factor_undefined_correlation():
    row = report.cross_sections(frame().with_columns(pl.lit(1.0).alias("factor")), "factor").row(0, named=True)
    assert row["ic"] is None and row["rank_ic"] is None


def test_tails_ceil_ties_and_deciles():
    values = np.arange(89, dtype=float)
    symbols = np.array([f"S{i:03}" for i in range(89)])
    low, high = report.tail_indices(values, symbols)
    assert len(low) == len(high) == 9
    assert set(low).isdisjoint(high)
    assert set(high) == set(range(80, 89))
    low, high = report.tail_indices(np.zeros(89), symbols)
    assert list(high) == list(range(9))
    assert list(low) == list(range(9, 18))
    d = report.daily_sections(frame(), "factor")[0]
    assert d["raw_spread"] == pytest.approx(d["top_return"] - d["bottom_return"])
    assert sum(d[f"g{i}_count"] for i in range(1, 11)) == 30
    assert d["exit_timestamp"] - d["decision_timestamp"] == report.DAY


def test_ir_uses_sample_std_and_skips_missing():
    assert report.ratio(np.array([1, 2, 3, np.nan])) == pytest.approx(2.0)
    assert report.ratio(np.array([1, 1])) is None


def test_direction_uses_training_only():
    cs = pl.DataFrame({"fold": [1] * 4, "split": ["train", "train", "test", "test"], "factor_name": ["factor"] * 4, "timestamp": [0, 1, 2, 3], "eligible_count": [30] * 4, "ic": [-.1, -.3, .2, .4], "rank_ic": [-.1, -.3, .2, .4]})
    daily = pl.DataFrame(report.daily_sections(frame(), "factor"))
    daily = pl.concat([daily, daily.with_columns(pl.lit("test").alias("split"))])
    importance = pl.DataFrame({"fold": [1], "feature_name": ["factor"], "importance_gain_pct": [.2]})
    summary, result = report.summarize(cs, daily, importance)
    assert result["train_direction"].to_list() == [-1, -1]
    assert np.allclose(result["directed_spread"], -result["raw_spread"])
    assert summary.filter(pl.col("split") == "test")["rank_ic_direction_win_rate"].item() == 0
