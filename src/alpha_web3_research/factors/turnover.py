"""Scale-invariant turnover proxies using historical traded volume and prices."""
from __future__ import annotations

import ccxt
import polars as pl

TURNOVER_FACTOR_NAMES = (
    "turnover_directional_6", "turnover_directional_42", "turnover_directional_180",
    "turnover_price_corr_42", "turnover_price_corr_180",
    "turnover_relative_daily_42", "turnover_relative_daily_180",
    "turnover_relative_volume_6", "turnover_relative_volume_42", "turnover_relative_volume_180",
    "turnover_return_premium_42", "turnover_return_premium_180",
    "turnover_volume_cv_42", "turnover_volume_cv_180",
    "turnover_volume_maxshare_42", "turnover_volume_maxshare_180",
)


def build_turnover_factors(data: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
    """Return 16 wide factors, with complete windows reset after timestamp gaps.

    Windows count bars. On 4h data, 6, 42 and 180 bars represent 1, 7 and
    30 days. These measures do not use circulating supply or open interest.
    """
    identity = [*group_columns, "timestamp"]
    if data.is_empty():
        return data.select(identity).with_columns([
            pl.lit(None, dtype=pl.Float64).alias(name) for name in TURNOVER_FACTOR_NAMES
        ])
    frame = data.select(*identity, "close", "volume").sort(identity)
    steps = {value: ccxt.Exchange.parse_timeframe(value) * 1000
             for value in frame["timeframe"].unique().to_list()}
    frame = frame.filter(
        pl.col("close").is_finite() & (pl.col("close") > 0)
        & pl.col("volume").is_finite() & (pl.col("volume") >= 0)
    ).with_columns(
        (pl.col("timestamp").diff().over(group_columns)
         != pl.col("timeframe").replace_strict(steps)).fill_null(True).cast(pl.Int64).alias("_gap")
    ).with_columns(
        pl.col("_gap").cum_sum().over(group_columns).alias("_segment")
    )
    groups = [*group_columns, "_segment"]
    frame = frame.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over(groups) - 1.0).alias("_return")
    )
    volume, returns = pl.col("volume"), pl.col("_return")
    expressions = []
    for window in (6, 42, 180):
        prior_mean = volume.shift(1).rolling_mean(window, min_samples=window).over(groups)
        expressions.append(
            pl.when(prior_mean > 0).then(volume / prior_mean)
            .alias(f"turnover_relative_volume_{window}")
        )
        total = volume.rolling_sum(window, min_samples=window).over(groups)
        signed = (volume * returns.sign()).rolling_sum(window, min_samples=window).over(groups)
        expressions.append(
            pl.when(total > 0).then(signed / total).alias(f"turnover_directional_{window}")
        )
    for window in (42, 180):
        mean = volume.rolling_mean(window, min_samples=window).over(groups)
        total = volume.rolling_sum(window, min_samples=window).over(groups)
        prior_mean = volume.shift(6).rolling_mean(window, min_samples=window).over(groups)
        daily_total = volume.rolling_sum(6, min_samples=6).over(groups)
        std = volume.rolling_std(window, min_samples=window, ddof=1).over(groups)
        maximum = volume.rolling_max(window, min_samples=window).over(groups)
        weighted_return = (volume * returns).rolling_sum(window, min_samples=window).over(groups)
        average_return = returns.rolling_mean(window, min_samples=window).over(groups)
        expressions.extend([
            pl.when(prior_mean > 0).then(daily_total / (6.0 * prior_mean))
            .alias(f"turnover_relative_daily_{window}"),
            pl.when(mean > 0).then(std / mean).alias(f"turnover_volume_cv_{window}"),
            pl.when(total > 0).then(maximum / total).alias(f"turnover_volume_maxshare_{window}"),
            pl.rolling_corr(volume, returns, window_size=window, min_samples=window)
            .over(groups).alias(f"turnover_price_corr_{window}"),
            pl.when(total > 0).then(weighted_return / total - average_return)
            .alias(f"turnover_return_premium_{window}"),
        ])
    wide = frame.select(*identity, *expressions)
    names = [name for name in wide.columns if name not in identity]
    return wide.with_columns([
        pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
        for name in names
    ])
