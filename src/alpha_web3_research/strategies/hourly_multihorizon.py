from __future__ import annotations

import numpy as np
import polars as pl


class HourlyMultiHorizonAbsoluteScoreLongOnlyStrategy:
    strategy_key = "hourly_multihorizon_absolute_score_long_only@v1"

    def target_weights(
        self,
        scores: np.ndarray,
        risks: np.ndarray,
        current_weights: np.ndarray,
        entry_threshold: float,
        exit_threshold: float,
        gross_exposure: float,
        max_asset_weight: float,
    ) -> np.ndarray:
        active = current_weights > 1e-8
        eligible = np.isfinite(scores) & np.isfinite(risks) & (risks > 0)
        eligible &= np.where(active, scores >= exit_threshold, scores >= entry_threshold)
        conviction = np.where(eligible, np.maximum(scores - exit_threshold, 0.0), 0.0)
        raw_weights = np.divide(conviction, risks, out=np.zeros_like(conviction), where=risks > 0)
        return capped_normalized_weights(raw_weights, gross_exposure, max_asset_weight)


def build_composite_score(
    frame: pl.DataFrame,
    horizons: tuple[int, ...],
    horizon_weights: tuple[float, ...],
    round_trip_cost_rate: float,
    disagreement_penalty: float,
) -> pl.DataFrame:
    weights = np.asarray(horizon_weights, dtype=float)
    if len(weights) != len(horizons) or weights.sum() <= 0:
        raise ValueError("horizon weights must match horizons and have positive sum")
    weights = weights / weights.sum()
    z_columns: list[str] = []
    output = frame
    for horizon in horizons:
        name = f"z_h{horizon}"
        z_columns.append(name)
        output = output.with_columns(
            (
                (pl.col(f"predicted_return_h{horizon}") - round_trip_cost_rate)
                / pl.col(f"predicted_volatility_h{horizon}")
            ).alias(name)
        )
    weighted_score = sum(pl.col(name) * float(weight) for name, weight in zip(z_columns, weights, strict=True))
    disagreement = pl.concat_list(z_columns).list.std(ddof=0)
    return output.with_columns(
        weighted_score.alias("weighted_horizon_score"),
        disagreement.alias("horizon_disagreement"),
    ).with_columns(
        (pl.col("weighted_horizon_score") - disagreement_penalty * pl.col("horizon_disagreement")).alias("score")
    )


def capped_normalized_weights(raw: np.ndarray, gross_exposure: float, max_asset_weight: float) -> np.ndarray:
    weights = np.zeros_like(raw, dtype=float)
    remaining = np.flatnonzero(np.isfinite(raw) & (raw > 0))
    remaining_gross = gross_exposure
    while remaining.size and remaining_gross > 1e-12:
        values = raw[remaining]
        allocation = remaining_gross * values / values.sum()
        capped = allocation >= max_asset_weight
        if not capped.any():
            weights[remaining] = allocation
            break
        capped_indices = remaining[capped]
        weights[capped_indices] = max_asset_weight
        remaining_gross -= max_asset_weight * capped_indices.size
        remaining = remaining[~capped]
    return weights
