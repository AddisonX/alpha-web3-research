from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from alpha_web3_research.config import Settings
from alpha_web3_research.data.ohlcv import OHLCV_COLUMNS
from alpha_web3_research.data.contracts import BarQuery
from alpha_web3_research.data.time import format_timestamp_ms


class ParquetOHLCVStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.resolved_data_path()
        self.ohlcv_root = self.root / "ohlcv"
        self.catalog_root = self.root / "catalog"
        self.reports_root = settings.resolved_reports_path()

    def upsert_ohlcv(self, frame: pl.DataFrame) -> int:
        if frame.is_empty():
            return 0

        written = 0
        for keys, group in frame.group_by(["exchange", "market_type", "symbol", "timeframe"], maintain_order=True):
            exchange, market_type, symbol, timeframe = keys
            path = self.ohlcv_path(exchange, market_type, symbol, timeframe)
            path.parent.mkdir(parents=True, exist_ok=True)
            combined = group
            if path.exists():
                existing = pl.read_parquet(path)
                combined = pl.concat([existing, group], how="diagonal_relaxed")
            combined = (
                combined.select(OHLCV_COLUMNS)
                .unique(subset=["exchange", "market_type", "symbol", "timeframe", "timestamp"], keep="last")
                .sort("timestamp")
            )
            combined.write_parquet(path)
            written += group.height
        return written

    def ohlcv_path(self, exchange: str, market_type: str, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        return (
            self.ohlcv_root
            / f"exchange={exchange}"
            / f"market_type={market_type}"
            / f"timeframe={timeframe}"
            / f"symbol={safe_symbol}"
            / "data.parquet"
        )

    def parquet_files(self) -> list[Path]:
        return sorted(self.ohlcv_root.glob("exchange=*/market_type=*/timeframe=*/symbol=*/data.parquet"))

    def read_ohlcv(self) -> pl.DataFrame:
        files = self.parquet_files()
        if not files:
            return pl.DataFrame(schema={column: pl.String for column in OHLCV_COLUMNS})
        return pl.concat([pl.read_parquet(file) for file in files], how="diagonal_relaxed")

    def read_bars(self, query: BarQuery) -> pl.DataFrame:
        """Read only matching legacy partitions, then enforce row-level identity.

        No user input is interpolated into a glob. Filename matching alone is
        insufficient because legacy symbol sanitization is not injective.
        """
        files = []
        for path in self.parquet_files():
            if (path.parent.parent.name != f"timeframe={query.timeframe}"
                    or path.parent.parent.parent.name != f"market_type={query.market_type}"
                    or path.parent.parent.parent.parent.name != f"exchange={query.venue}"):
                continue
            if query.symbols is not None:
                names = {self.ohlcv_path(query.venue, query.market_type, symbol, query.timeframe).parent.name
                         for symbol in query.symbols}
                if path.parent.name not in names:
                    continue
            files.append(path)
        if not files:
            return self.read_ohlcv_empty()
        predicate = ((pl.col("exchange") == query.venue)
                     & (pl.col("market_type") == query.market_type)
                     & (pl.col("timeframe") == query.timeframe))
        if query.symbols is not None:
            predicate &= pl.col("symbol").is_in(query.symbols)
        if query.start_ms is not None:
            predicate &= pl.col("timestamp") >= query.start_ms
        if query.end_ms is not None:
            predicate &= pl.col("timestamp") < query.end_ms
        return (pl.concat([pl.scan_parquet(path, hive_partitioning=False) for path in files],
                          how="diagonal_relaxed")
                .filter(predicate).collect().sort(["symbol", "timestamp"]))

    @staticmethod
    def read_ohlcv_empty() -> pl.DataFrame:
        return pl.DataFrame(schema={
            **{column: pl.String for column in OHLCV_COLUMNS},
            "timestamp": pl.Int64,
            **{column: pl.Float64 for column in ("open", "high", "low", "close", "volume")},
        })

    def build_catalog(self) -> pl.DataFrame:
        data = self.read_ohlcv()
        if data.is_empty():
            return pl.DataFrame(
                schema={
                    "exchange": pl.String,
                    "market_type": pl.String,
                    "symbol": pl.String,
                    "timeframe": pl.String,
                    "rows": pl.Int64,
                    "start": pl.String,
                    "end": pl.String,
                    "path": pl.String,
                }
            )

        catalog = (
            data.group_by(["exchange", "market_type", "symbol", "timeframe"])
            .agg(
                pl.len().alias("rows"),
                pl.min("timestamp").alias("start_ms"),
                pl.max("timestamp").alias("end_ms"),
            )
            .sort(["exchange", "market_type", "timeframe", "symbol"])
        )
        return catalog.with_columns(
            pl.struct(["exchange", "market_type", "symbol", "timeframe"])
            .map_elements(lambda row: str(self.ohlcv_path(row["exchange"], row["market_type"], row["symbol"], row["timeframe"])), return_dtype=pl.String)
            .alias("path"),
            pl.col("start_ms").map_elements(format_timestamp_ms, return_dtype=pl.String).alias("start"),
            pl.col("end_ms").map_elements(format_timestamp_ms, return_dtype=pl.String).alias("end"),
        ).select(["exchange", "market_type", "symbol", "timeframe", "rows", "start", "end", "path"])

    def write_catalog(self, frame: pl.DataFrame) -> Path:
        self.catalog_root.mkdir(parents=True, exist_ok=True)
        path = self.catalog_root / "ohlcv_catalog.parquet"
        frame.write_parquet(path)
        return path

    def read_catalog(self) -> pl.DataFrame:
        path = self.catalog_root / "ohlcv_catalog.parquet"
        if not path.exists():
            return self.build_catalog()
        return pl.read_parquet(path)

    def create_duckdb_database(self) -> Path:
        self.catalog_root.mkdir(parents=True, exist_ok=True)
        database_path = self.catalog_root / "research.duckdb"
        parquet_glob = str(self.ohlcv_root / "exchange=*" / "market_type=*" / "timeframe=*" / "symbol=*" / "data.parquet")
        with duckdb.connect(str(database_path)) as connection:
            connection.execute(
                f"CREATE OR REPLACE VIEW ohlcv AS "
                f"SELECT * FROM read_parquet('{_duckdb_string(parquet_glob)}', union_by_name=true, hive_partitioning=false)"
            )
            latest_universe = self.root / "universe" / "latest_universe.parquet"
            if latest_universe.exists():
                connection.execute(
                    f"CREATE OR REPLACE VIEW universe_latest AS "
                    f"SELECT * FROM read_parquet('{_duckdb_string(str(latest_universe))}', union_by_name=true)"
                )
            latest_labels = self.root / "labels" / "latest_labels.parquet"
            if latest_labels.exists():
                connection.execute(
                    f"CREATE OR REPLACE VIEW labels_latest AS "
                    f"SELECT * FROM read_parquet('{_duckdb_string(str(latest_labels))}', union_by_name=true)"
                )
            latest_factors = self.root / "factors" / "latest_factors.parquet"
            if latest_factors.exists():
                connection.execute(
                    f"CREATE OR REPLACE VIEW factors_latest AS "
                    f"SELECT * FROM read_parquet('{_duckdb_string(str(latest_factors))}', union_by_name=true)"
                )
            evaluation_root = self.root / "evaluation"
            evaluation_views = {
                "single_factor_summary": evaluation_root / "single_factor_summary.parquet",
                "single_factor_timeseries": evaluation_root / "single_factor_timeseries.parquet",
                "single_factor_group_returns": evaluation_root / "single_factor_group_returns.parquet",
                "decorrelation_decisions": evaluation_root / "decorrelation_decisions.parquet",
                "decorrelated_factors": evaluation_root / "decorrelated_factors.parquet",
            }
            for view_name, path in evaluation_views.items():
                if path.exists():
                    connection.execute(
                        f"CREATE OR REPLACE VIEW {view_name} AS "
                        f"SELECT * FROM read_parquet('{_duckdb_string(str(path))}', union_by_name=true)"
                    )
            models_root = self.root / "models"
            model_views = {
                "model_predictions_latest": models_root / "latest_predictions.parquet",
                "model_feature_importance_latest": models_root / "latest_feature_importance.parquet",
                "model_experiment_summary_latest": models_root / "experiments" / "latest_summary.parquet",
                "model_walk_forward_folds_latest": models_root / "walk_forward" / "latest_folds.parquet",
                "model_walk_forward_predictions_latest": models_root / "walk_forward" / "latest_predictions.parquet",
                "model_walk_forward_feature_importance_latest": models_root / "walk_forward" / "latest_feature_importance.parquet",
                "model_walk_forward_grid_summary_latest": models_root / "walk_forward" / "grids" / "latest_summary.parquet",
            }
            for view_name, path in model_views.items():
                if path.exists():
                    connection.execute(
                        f"CREATE OR REPLACE VIEW {view_name} AS "
                        f"SELECT * FROM read_parquet('{_duckdb_string(str(path))}', union_by_name=true)"
                    )
            strategy_root = self.root / "strategies"
            strategy_views = {
                "strategy_definitions": strategy_root / "strategy_definitions.parquet",
                "strategy_runs": strategy_root / "strategy_runs.parquet",
            }
            for view_name, path in strategy_views.items():
                if path.exists():
                    connection.execute(
                        f"CREATE OR REPLACE VIEW {view_name} AS "
                        f"SELECT * FROM read_parquet('{_duckdb_string(str(path))}', union_by_name=true)"
                    )
        return database_path

    def build_quality_frame(self) -> pl.DataFrame:
        data = self.read_ohlcv()
        if data.is_empty():
            return pl.DataFrame(
                schema={
                    "exchange": pl.String,
                    "market_type": pl.String,
                    "symbol": pl.String,
                    "timeframe": pl.String,
                    "rows": pl.Int64,
                    "duplicate_rows": pl.Int64,
                    "null_rows": pl.Int64,
                    "bad_price_rows": pl.Int64,
                    "bad_volume_rows": pl.Int64,
                    "gap_count": pl.Int64,
                }
            )

        duplicate_counts = (
            data.group_by(["exchange", "market_type", "symbol", "timeframe", "timestamp"])
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") > 1)
            .group_by(["exchange", "market_type", "symbol", "timeframe"])
            .agg((pl.sum("count") - pl.len()).alias("duplicate_rows"))
        )

        base = (
            data.group_by(["exchange", "market_type", "symbol", "timeframe"])
            .agg(
                pl.len().alias("rows"),
                pl.any_horizontal([pl.col(column).is_null() for column in ["timestamp", "open", "high", "low", "close", "volume"]]).sum().alias("null_rows"),
                ((pl.col("open") <= 0) | (pl.col("high") <= 0) | (pl.col("low") <= 0) | (pl.col("close") <= 0) | (pl.col("high") < pl.col("low"))).sum().alias("bad_price_rows"),
                (pl.col("volume") < 0).sum().alias("bad_volume_rows"),
            )
        )

        gaps = (
            data.sort(["exchange", "market_type", "symbol", "timeframe", "timestamp"])
            .with_columns(
                pl.col("timestamp").diff().over(["exchange", "market_type", "symbol", "timeframe"]).alias("delta_ms")
            )
            .with_columns(pl.col("delta_ms").median().over(["exchange", "market_type", "symbol", "timeframe"]).alias("expected_delta_ms"))
            .filter((pl.col("delta_ms").is_not_null()) & (pl.col("expected_delta_ms").is_not_null()) & (pl.col("delta_ms") > pl.col("expected_delta_ms")))
            .group_by(["exchange", "market_type", "symbol", "timeframe"])
            .agg(pl.len().alias("gap_count"))
        )

        return (
            base.join(duplicate_counts, on=["exchange", "market_type", "symbol", "timeframe"], how="left")
            .join(gaps, on=["exchange", "market_type", "symbol", "timeframe"], how="left")
            .with_columns(
                pl.col("duplicate_rows").fill_null(0),
                pl.col("gap_count").fill_null(0),
            )
            .sort(["exchange", "market_type", "timeframe", "symbol"])
        )

    def write_quality_report(self) -> Path:
        quality = self.build_quality_frame()
        self.catalog_root.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)
        quality.write_parquet(self.catalog_root / "ohlcv_quality.parquet")

        report_path = self.reports_root / "ohlcv_quality.md"
        lines = ["# OHLCV Quality Report", ""]
        if quality.is_empty():
            lines.append("No OHLCV data found.")
        else:
            lines.extend(
                [
                    "| exchange | market_type | symbol | timeframe | rows | duplicate_rows | null_rows | bad_price_rows | bad_volume_rows | gap_count |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in quality.iter_rows(named=True):
                lines.append(
                    f"| {row['exchange']} | {row['market_type']} | {row['symbol']} | {row['timeframe']} | "
                    f"{row['rows']} | {row['duplicate_rows']} | {row['null_rows']} | {row['bad_price_rows']} | "
                    f"{row['bad_volume_rows']} | {row['gap_count']} |"
                )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path


def _duckdb_string(value: str) -> str:
    return value.replace("'", "''")
