"""Original percentile XGBoost policy, with explicit signed daily allocations."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from src.strategies.base import DataRequirements, StrategyContext, StrategyDecision, StrategyMetadata

STRATEGY_ID = "xgb15-120-long-short-v1"
FEATURES = (
    "market_cap_mean_10d", "market_cap_mean_20d", "market_cap_mean_30d", "market_cap_mean_40d",
    "mean_return_4", "range_to_quote_volume_2", "return_6", "turnover_relative_volume_42",
    "volume_change_12", "volume_change_24", "avg_range_4", "distance_to_low_6", "avg_range_3",
    "distance_to_low_4", "volatility_4",
)


def percentile_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Rank each available factor before dropping incomplete prediction rows."""
    if raw.symbol.duplicated().any() or raw.symbol.isna().any():
        raise ValueError("duplicate or missing symbols")
    out = raw[["symbol", *FEATURES]].copy()
    out[list(FEATURES)] = out[list(FEATURES)].replace([np.inf, -np.inf], np.nan).rank(method="average", pct=True)
    return out


def buffered_targets(pairs: list[str], scores: np.ndarray, previous: pl.DataFrame | None) -> pl.DataFrame:
    if len(pairs) < 20 or len(set(pairs)) != len(pairs) or not np.isfinite(scores).all():
        raise ValueError("invalid scoring universe")
    frame = pd.DataFrame({"pair": pairs, "model_score": scores})
    upper = frame.sort_values(["model_score", "pair"], ascending=[False, True]).pair.tolist()
    lower = frame.sort_values(["model_score", "pair"], ascending=[True, True]).pair.tolist()
    old = {} if previous is None else dict(zip(previous["pair"].to_list(), previous["target_position"].to_list()))
    chosen = {}
    for sign, order, count in ((1, upper, len(pairs)//10), (-1, lower, math.ceil(len(pairs)/10))):
        keep = [p for p in order[:math.ceil(len(pairs)*.3)] if old.get(p) == sign][:count]
        keep += [p for p in order[:count] if p not in keep][:count-len(keep)]
        chosen.update({p: sign * .5 / count for p in keep})
    frame["target_weight"] = frame.pair.map(chosen).fillna(0.)
    frame["target_position"] = np.sign(frame.target_weight).astype("int8")
    frame["score_position"] = frame.pair.map({p: i+1 for i, p in enumerate(upper)})
    frame["model_score_rank_pct"] = frame.model_score.rank(method="average", pct=True)
    return pl.from_pandas(frame.sort_values("pair"))


class XGB15LongShortStrategy:
    strategy_id = STRATEGY_ID
    allow_short = True
    rebalance_hour_utc = 12  # Decision bar opens at 12 UTC, closes at BJT 00:00.
    requirements = DataRequirements(FEATURES, lookback_bars=64, min_closed_bars=43)
    metadata = StrategyMetadata("xgb15-expanding-2021-120-depth4-lr005-buffer30", 1, 6, "excess_return_24h")

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("features") != list(FEATURES) or self.manifest.get("preprocessing") != "percentile_average":
            raise ValueError("model feature/preprocessing contract mismatch")
        self.models = {}

    def _model(self, year: int) -> xgb.Booster:
        if year not in self.models:
            record = self.manifest["models"][str(year)]
            path = (self.model_dir / record["file"]).resolve()
            if not path.is_relative_to(self.model_dir.resolve()) or path.suffix != ".ubj":
                raise ValueError("invalid model path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("model SHA-256 mismatch")
            model = xgb.Booster()
            model.load_model(path)
            if model.num_boosted_rounds() != 120 or model.feature_names != list(FEATURES):
                raise ValueError("model tree count or feature order mismatch")
            model.set_param({"device": "cpu", "nthread": 4})  # Inference only, no training.
            self.models[year] = model
        return self.models[year]

    def decide(self, context: StrategyContext) -> StrategyDecision:
        frame = context.features.sort("symbol")
        if frame["symbol"].null_count() or frame["symbol"].n_unique() != frame.height:
            raise ValueError("duplicate or missing symbols")
        values = frame.select(FEATURES).to_numpy()
        if not np.isfinite(values).all() or (values <= 0).any() or (values > 1).any():
            raise ValueError("inputs must be complete percentile features in (0, 1]")
        signal_time = pd.Timestamp(context.decision_timestamp + 14_400_000, unit="ms", tz="UTC")
        if signal_time.hour != 16 or signal_time.minute or signal_time.second:
            raise ValueError("expected Beijing midnight signal")
        year = signal_time.tz_convert("Asia/Shanghai").year
        record = self.manifest["models"][str(year)]
        if signal_time <= pd.Timestamp(record["fit_cutoff_utc"]):
            raise ValueError("annual model or pool not yet available")
        if not set(frame["symbol"].to_list()).issubset(record["pairs"]):
            raise ValueError("symbols outside annual model pool")
        scores = self._model(year).predict(xgb.DMatrix(values, feature_names=list(FEATURES)))
        decision = StrategyDecision(context.decision_timestamp,
            buffered_targets(frame["symbol"].to_list(), scores, context.previous_targets), allow_short=True)
        decision.validate(context)
        return decision
