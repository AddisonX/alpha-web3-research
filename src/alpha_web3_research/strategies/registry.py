from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from alpha_web3_research.config import Settings
from alpha_web3_research.strategies.base import StrategyDefinition, StrategyRun


class StrategyRegistry:
    """Load versioned strategy definitions and index immutable historical run artifacts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.project_root = settings.project_root
        self.catalog_root = self.project_root / "strategies" / "catalog"
        self.output_root = settings.resolved_data_path() / "strategies"
        self.registry_path = self.output_root / "strategy_registry.json"
        self.definitions_parquet_path = self.output_root / "strategy_definitions.parquet"
        self.runs_path = self.output_root / "strategy_runs.json"
        self.runs_parquet_path = self.output_root / "strategy_runs.parquet"

    def definitions(self) -> list[StrategyDefinition]:
        files = sorted(self.catalog_root.glob("*.json"))
        definitions = [StrategyDefinition.model_validate_json(path.read_text(encoding="utf-8")) for path in files]
        keys = [definition.strategy_key for definition in definitions]
        if len(keys) != len(set(keys)):
            raise ValueError("strategy catalog contains duplicate strategy keys")
        return sorted(definitions, key=lambda item: (item.strategy_id, item.version))

    def get(self, strategy_id_or_key: str) -> StrategyDefinition:
        definitions = self.definitions()
        exact = [item for item in definitions if item.strategy_key == strategy_id_or_key]
        if exact:
            return exact[0]
        versions = [item for item in definitions if item.strategy_id == strategy_id_or_key]
        if not versions:
            raise KeyError(f"strategy not found: {strategy_id_or_key}")
        return max(versions, key=lambda item: item.version)

    def discover_runs(self, strategy_id_or_key: str | None = None) -> list[StrategyRun]:
        definitions = self.definitions()
        if strategy_id_or_key is not None:
            selected = self.get(strategy_id_or_key)
            definitions = [selected]
        runs: list[StrategyRun] = []
        for definition in definitions:
            for rule in definition.run_discovery:
                for path in sorted(self.project_root.glob(rule.pattern)):
                    run = self._run_from_artifact(definition, rule, path)
                    if run is not None:
                        runs.append(run)
        unique: dict[tuple[str, str, str], StrategyRun] = {}
        for run in runs:
            unique[(run.strategy_key, run.kind, run.run_id)] = run
        return sorted(unique.values(), key=lambda item: (item.created_at, item.strategy_key, item.run_id))

    def sync(self) -> tuple[list[StrategyDefinition], list[StrategyRun]]:
        definitions = self.definitions()
        runs = self.discover_runs()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps([item.model_dump(mode="json") for item in definitions], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        definition_rows = [
            {
                "strategy_key": item.strategy_key,
                "strategy_id": item.strategy_id,
                "version": item.version,
                "name": item.name,
                "description": item.description,
                "status": item.status,
                "direction": item.direction,
                "decision_frequency": item.decision_frequency,
                "selection_method": item.selection_method,
                "implementation_entrypoint": item.implementation_entrypoint,
                "model_contract": item.model_contract,
                "target_contract": item.target_contract,
                "execution_adapter": item.execution_adapter,
                "created_at": item.created_at,
                "predecessor": item.predecessor,
                "tags_json": json.dumps(item.tags, ensure_ascii=False),
                "source_paths_json": json.dumps(item.source_paths, ensure_ascii=False),
                "documentation_paths_json": json.dumps(item.documentation_paths, ensure_ascii=False),
            }
            for item in definitions
        ]
        pl.DataFrame(definition_rows).write_parquet(self.definitions_parquet_path)
        self.runs_path.write_text(
            json.dumps([item.model_dump(mode="json") for item in runs], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rows = [
            {
                **item.model_dump(exclude={"metadata"}),
                "metadata_json": json.dumps(item.metadata, ensure_ascii=False, sort_keys=True),
            }
            for item in runs
        ]
        if rows:
            pl.DataFrame(rows).write_parquet(self.runs_parquet_path)
        else:
            pl.DataFrame(
                schema={
                    "strategy_key": pl.String,
                    "run_id": pl.String,
                    "kind": pl.String,
                    "created_at": pl.String,
                    "primary_artifact": pl.String,
                    "artifact_root": pl.String,
                    "metadata_json": pl.String,
                }
            ).write_parquet(self.runs_parquet_path)
        return definitions, runs

    def _run_from_artifact(self, definition, rule, path: Path) -> StrategyRun | None:
        metadata: dict[str, Any] = {}
        if path.suffix == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                return None
            metadata = loaded
        if rule.metadata_strategy_class is not None:
            strategy_metadata = metadata.get(rule.metadata_strategy_class)
            if not isinstance(strategy_metadata, dict):
                return None
            metadata = strategy_metadata
        if not all(metadata.get(key) == value for key, value in rule.metadata_match.items()):
            return None
        run_id = str(metadata.get(rule.run_id_field, "")) if rule.run_id_field else ""
        if not run_id:
            run_id = path.parent.name if path.name in {"manifest.json", "config.json"} else path.stem
        created_value = metadata.get(rule.created_at_field) if rule.created_at_field else None
        created_at = normalize_created_at(created_value, path)
        return StrategyRun(
            strategy_key=definition.strategy_key,
            run_id=run_id,
            kind=rule.kind,
            created_at=created_at,
            primary_artifact=str(path.relative_to(self.project_root)),
            artifact_root=str(path.parent.relative_to(self.project_root)),
            metadata=compact_metadata(metadata),
        )


def normalize_created_at(value: Any, path: Path) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    if isinstance(value, str) and value:
        return value
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    preferred = [
        "timeframe",
        "horizon_bars",
        "label_column",
        "run_id",
        "backtest_start_ts",
        "backtest_end_ts",
        "notes",
        "chosen_entry_threshold",
        "chosen_exit_threshold",
    ]
    return {key: metadata[key] for key in preferred if key in metadata}
