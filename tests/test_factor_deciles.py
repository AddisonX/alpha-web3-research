import importlib
from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("build_factor_2026_curves"), importlib.import_module("build_factor_2026_deciles")


def samples():
    rows = []
    for day in range(3):
        for asset in range(23):
            value = asset / 100 + day / 1000
            rows.append({"symbol": f"S{asset:02}", "decision_timestamp": day * 86_400_000,
                         "label_end_timestamp": (day + 1) * 86_400_000,
                         "factor": asset // 2, "forward_return": value,
                         "universe_forward_return": .11 + day / 1000,
                         "excess_forward_return": value - .11 - day / 1000,
                         "label_universe_count": 23})
    return pl.DataFrame(rows)


def test_deciles_disjoint_complete_and_spread(modules):
    curves, deciles = modules
    frame = samples()
    metrics, _, _ = curves.calculate(frame, ["factor"])
    daily, spread, summary = deciles.compute(frame, ["factor"], metrics)
    assert daily.height == 30 and spread.height == 3
    for _, group in daily.partition_by("decision_timestamp", as_dict=True).items():
        members = sum(group["symbols"].to_list(), [])
        assert len(members) == len(set(members)) == 23
        assert group["member_count"].max() - group["member_count"].min() <= 1
        ordered = group.sort("decile")
        assert ordered["return"].is_sorted()
    assert np.allclose(spread["spread"], spread["g10_return"] - spread["g1_return"])
    assert summary["cumulative_spread"].item() == pytest.approx(spread["spread"].sum())


def test_missing_future_drops_whole_date_not_asset(modules):
    curves, _ = modules
    frame = samples().with_columns(pl.when((pl.col("symbol") == "S00") & (pl.col("decision_timestamp") == 0))
                                  .then(None).otherwise(pl.col("forward_return")).alias("forward_return"))
    summary, daily, missing = curves.calculate(frame, ["factor"])
    assert summary["daily_period_count"].item() == 2
    assert missing.height == 1
    assert daily["decision_timestamp"].min() == 86_400_000


def test_constant_cross_section_ic_is_missing(modules):
    curves, _ = modules
    frame = samples().with_columns(pl.when(pl.col("decision_timestamp") == 0).then(1)
                                  .otherwise(pl.col("factor")).alias("factor"))
    summary, daily, _ = curves.calculate(frame, ["factor"])
    assert summary["rank_ic_days"].item() == 2
    assert daily.filter(pl.col("decision_timestamp") == 0)["rank_ic"].item() is None
