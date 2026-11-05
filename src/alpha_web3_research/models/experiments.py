from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from alpha_web3_research.config import Settings
from alpha_web3_research.data.storage import ParquetOHLCVStore
from alpha_web3_research.models.training import LightGBMCrossSectionalTrainer, fmt


@dataclass(frozen=True)
class ModelExperimentGridResult:
    experiment_id: str
    summary_path: Path
    report_path: Path
    run_count: int


class ModelExperimentGridRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_root = settings.resolved_data_path()
        self.models_root = self.data_root / "models"
        self.experiments_root = self.models_root / "experiments"
        self.reports_root = settings.resolved_reports_path() / "models"
        self.decorrelated_path = self.data_root / "evaluation" / "decorrelated_factors.parquet"

    def run(
        self,
        labels: list[str],
        timeframes: list[str] | None = None,
        top_quantile_fraction: float = 0.10,
        train_fraction: float = 0.70,
        valid_fraction: float = 0.15,
        min_cross_section_count: int = 20,
        seed: int = 42,
    ) -> ModelExperimentGridResult:
        experiment_id = f"lgbm_grid_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        combinations = self._available_combinations(timeframes=timeframes)
        trainer = LightGBMCrossSectionalTrainer(settings=self.settings)
        rows: list[dict[str, object]] = []

        for combination in combinations.iter_rows(named=True):
            timeframe = str(combination["timeframe"])
            horizon_bars = int(combination["horizon_bars"])
            for label_column in labels:
                result = trainer.train(
                    timeframe=timeframe,
                    horizon_bars=horizon_bars,
                    label_column=label_column,
                    train_fraction=train_fraction,
                    valid_fraction=valid_fraction,
                    min_cross_section_count=min_cross_section_count,
                    top_quantile_fraction=top_quantile_fraction,
                    seed=seed,
                )
                metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "run_id": result.run_id,
                        "timeframe": timeframe,
                        "horizon_bars": horizon_bars,
                        "label_column": label_column,
                        "top_quantile_fraction": top_quantile_fraction,
                        "row_count": result.row_count,
                        "feature_count": result.feature_count,
                        "train_rows": result.split_counts.get("train", 0),
                        "valid_rows": result.split_counts.get("valid", 0),
                        "test_rows": result.split_counts.get("test", 0),
                        **split_metrics(metrics, "valid"),
                        **split_metrics(metrics, "test"),
                        "model_path": str(result.model_path),
                        "predictions_path": str(result.predictions_path),
                        "metrics_path": str(result.metrics_path),
                        "report_path": str(result.report_path),
                    }
                )

        summary = pl.DataFrame(rows)
        experiment_dir = self.experiments_root / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)
        summary_path = experiment_dir / "summary.parquet"
        report_path = self.reports_root / f"{experiment_id}.md"
        latest_summary_path = self.experiments_root / "latest_summary.parquet"
        latest_report_path = self.reports_root / "latest_lgbm_grid_report.md"
        summary.write_parquet(summary_path)
        summary.write_parquet(latest_summary_path)
        self._write_report(summary=summary, report_path=report_path, experiment_id=experiment_id, labels=labels)
        latest_report_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
        ParquetOHLCVStore(settings=self.settings).create_duckdb_database()
        return ModelExperimentGridResult(
            experiment_id=experiment_id,
            summary_path=summary_path,
            report_path=report_path,
            run_count=summary.height,
        )

    def _available_combinations(self, timeframes: list[str] | None) -> pl.DataFrame:
        if not self.decorrelated_path.exists():
            raise FileNotFoundError(f"decorrelated factors not found: {self.decorrelated_path}")
        frame = (
            pl.read_parquet(self.decorrelated_path)
            .group_by(["timeframe", "horizon_bars"])
            .agg(pl.len().alias("feature_count"))
            .sort(["timeframe", "horizon_bars"])
        )
        if timeframes:
            frame = frame.filter(pl.col("timeframe").is_in(timeframes))
        if frame.is_empty():
            raise ValueError("no timeframe/horizon combinations found for experiment grid")
        return frame

    def _write_report(self, summary: pl.DataFrame, report_path: Path, experiment_id: str, labels: list[str]) -> None:
        lines = [
            "# LightGBM Experiment Grid Report",
            "",
            f"- Experiment ID: `{experiment_id}`",
            f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
            f"- Labels: `{', '.join(labels)}`",
            f"- Runs: `{summary.height}`",
            f"- Spread quantile: top/bottom `{summary.get_column('top_quantile_fraction')[0]:.0%}`",
            "",
            "## Best Runs By Test Inverse Spread",
            "",
            "| rank | timeframe | horizon | label | test RankIC | test spread | inverse spread | valid spread | run |",
            "|---:|---|---:|---|---:|---:|---:|---:|---|",
        ]
        ranked = summary.sort(["test_inverse_spread", "test_rank_ic"], descending=[True, True])
        for index, row in enumerate(ranked.head(20).iter_rows(named=True), start=1):
            lines.append(
                f"| {index} | {row['timeframe']} | {row['horizon_bars']} | {row['label_column']} | "
                f"{fmt(row['test_rank_ic'])} | {fmt(row['test_spread'])} | {fmt(row['test_inverse_spread'])} | "
                f"{fmt(row['valid_spread'])} | `{row['run_id']}` |"
            )

        lines.extend(
            [
                "",
                "## Best Runs By Test Spread",
                "",
                "| rank | timeframe | horizon | label | test RankIC | test spread | inverse spread | valid spread | run |",
                "|---:|---|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        ranked_direct = summary.sort(["test_spread", "test_rank_ic"], descending=[True, True])
        for index, row in enumerate(ranked_direct.head(20).iter_rows(named=True), start=1):
            lines.append(
                f"| {index} | {row['timeframe']} | {row['horizon_bars']} | {row['label_column']} | "
                f"{fmt(row['test_rank_ic'])} | {fmt(row['test_spread'])} | {fmt(row['test_inverse_spread'])} | "
                f"{fmt(row['valid_spread'])} | `{row['run_id']}` |"
            )

        lines.extend(
            [
                "",
                "## All Runs",
                "",
                "| timeframe | horizon | label | rows | features | valid RankIC | valid spread | test RankIC | test spread | inverse spread |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary.sort(["timeframe", "horizon_bars", "label_column"]).iter_rows(named=True):
            lines.append(
                f"| {row['timeframe']} | {row['horizon_bars']} | {row['label_column']} | {row['row_count']} | "
                f"{row['feature_count']} | {fmt(row['valid_rank_ic'])} | {fmt(row['valid_spread'])} | "
                f"{fmt(row['test_rank_ic'])} | {fmt(row['test_spread'])} | {fmt(row['test_inverse_spread'])} |"
            )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_metrics(metrics: dict[str, object], split: str) -> dict[str, object]:
    split_metrics_data = metrics["splits"][split]  # type: ignore[index]
    return {
        f"{split}_rank_ic": split_metrics_data.get("mean_target_rank_ic"),  # type: ignore[union-attr]
        f"{split}_return_ic": split_metrics_data.get("mean_forward_return_ic"),  # type: ignore[union-attr]
        f"{split}_spread": split_metrics_data.get("mean_top_bottom_spread"),  # type: ignore[union-attr]
        f"{split}_inverse_spread": split_metrics_data.get("mean_inverse_top_bottom_spread"),  # type: ignore[union-attr]
        f"{split}_rmse": split_metrics_data.get("rmse"),  # type: ignore[union-attr]
        f"{split}_mae": split_metrics_data.get("mae"),  # type: ignore[union-attr]
    }
