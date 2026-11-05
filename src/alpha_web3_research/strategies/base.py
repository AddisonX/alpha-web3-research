from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field


class RunDiscoveryRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    pattern: str
    run_id_field: str | None = None
    created_at_field: str | None = None
    metadata_match: dict[str, Any] = Field(default_factory=dict)
    metadata_strategy_class: str | None = None


class StrategyDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    strategy_id: str
    version: int = Field(ge=1)
    name: str
    description: str
    status: str
    direction: str
    decision_frequency: str
    selection_method: str
    implementation_entrypoint: str
    model_contract: str
    target_contract: str
    execution_adapter: str
    created_at: str
    predecessor: str | None = None
    tags: list[str] = Field(default_factory=list)
    source_paths: list[str] = Field(default_factory=list)
    documentation_paths: list[str] = Field(default_factory=list)
    run_discovery: list[RunDiscoveryRule] = Field(default_factory=list)

    @computed_field
    @property
    def strategy_key(self) -> str:
        return f"{self.strategy_id}@v{self.version}"


class StrategyRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_key: str
    run_id: str
    kind: str
    created_at: str
    primary_artifact: str
    artifact_root: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TargetWeightStrategy(Protocol):
    strategy_key: str

    def target_weights(
        self,
        scores: np.ndarray,
        risks: np.ndarray,
        current_weights: np.ndarray,
        entry_threshold: float,
        exit_threshold: float,
        gross_exposure: float,
        max_asset_weight: float,
    ) -> np.ndarray: ...
