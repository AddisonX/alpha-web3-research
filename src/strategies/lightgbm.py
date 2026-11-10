from __future__ import annotations
import json
import math
from pathlib import Path
import lightgbm as lgb
import numpy as np
import polars as pl
from src.strategies.base import DEFAULT_STRATEGY_ID, DataRequirements, StrategyMetadata, StrategyContext, StrategyDecision

class LightGBMLongOnlyStrategy:
    strategy_id = DEFAULT_STRATEGY_ID

    def __init__(self, production_root: Path) -> None:
        self.model = ProductionEnsemble(production_root)
        self.requirements = DataRequirements(tuple(self.model.features), self.model.timeframe)
        self.metadata = StrategyMetadata(
            source_experiment=self.model.source_experiment,
            model_count=len(self.model.boosters),
            horizon_bars=self.model.horizon_bars,
            label_column=self.model.label_column,
        )

    def decide(self, context: StrategyContext) -> StrategyDecision:
        ranked = rank_long_only_scores(
            context.features["symbol"].to_list(),
            self.model.predict(context.features),
            self.model.top_quantile_fraction,
        )
        return StrategyDecision(context.decision_timestamp, ranked)


class ProductionEnsemble:
    def __init__(self, production_root: Path) -> None:
        latest_path = production_root / "latest.json"
        if not latest_path.exists():
            raise FileNotFoundError(f"production model pointer not found: {latest_path}")
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        self.bundle_dir = production_root / str(latest["bundle"])
        self.manifest_path = self.bundle_dir / "manifest.json"
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        source_config = manifest["source_config"]
        self.features = [str(name) for name in source_config["features"]]
        self.timeframe = str(source_config["timeframe"])
        self.horizon_bars = int(source_config["horizon_bars"])
        self.label_column = str(source_config["label_column"])
        self.top_quantile_fraction = float(source_config["top_quantile_fraction"])
        self.source_experiment = str(manifest["source_experiment"])
        self.boosters = [
            lgb.Booster(model_file=str(self.bundle_dir / str(item["file"])))
            for item in manifest["models"]
        ]
        if not self.boosters:
            raise ValueError(f"production bundle contains no models: {self.manifest_path}")
        for booster in self.boosters:
            if booster.feature_name() != self.features:
                raise ValueError("production model feature order does not match the manifest")

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        matrix = features.select(self.features).to_pandas()
        predictions = [booster.predict(matrix) for booster in self.boosters]
        return np.mean(np.vstack(predictions), axis=0)



def rank_long_only_scores(
    pairs: list[str],
    scores: np.ndarray,
    top_quantile_fraction: float,
) -> pl.DataFrame:
    if len(pairs) != len(scores):
        raise ValueError("pair and score counts do not match")
    selected_count = max(1, math.ceil(len(pairs) * top_quantile_fraction))
    return (
        pl.DataFrame({"pair": pairs, "model_score": scores})
        .sort(["model_score", "pair"], descending=[True, False])
        .with_row_index("score_position", offset=1)
        .with_columns(
            pl.lit(len(pairs)).alias("cross_section_count"),
            pl.lit(selected_count).alias("selected_count"),
            (
                1.0
                - (pl.col("score_position") - 1)
                / max(1, len(pairs) - 1)
            ).alias("model_score_rank_pct"),
            (pl.col("score_position") <= selected_count).cast(pl.Int8).alias("target_position"),
        )
    )

