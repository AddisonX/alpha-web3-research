from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alpha_web3_research.config import Settings


@dataclass(frozen=True)
class ProductionModelResult:
    bundle_dir: Path
    manifest_path: Path
    latest_path: Path
    model_count: int


class ProductionModelPromoter:
    """Promote a validated walk-forward ensemble into an immutable live bundle."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.walk_root = settings.resolved_data_path() / "models" / "walk_forward"
        self.production_root = settings.resolved_data_path() / "models" / "production"

    def promote(self, experiment_dir: Path) -> ProductionModelResult:
        source = experiment_dir.expanduser()
        if not source.is_absolute():
            source = self.settings.project_root / source
        source = source.resolve()
        config_path = source / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"walk-forward config not found: {config_path}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_paths = sorted(source.glob("fold_*_model.txt"))
        if not model_paths:
            raise FileNotFoundError(f"no fold models found in: {source}")
        experiment_id = str(config["experiment_id"])
        bundle_dir = self.production_root / experiment_id
        bundle_dir.mkdir(parents=True, exist_ok=True)

        models: list[dict[str, str | int]] = []
        for model_path in model_paths:
            destination = bundle_dir / model_path.name
            shutil.copy2(model_path, destination)
            models.append(
                {
                    "file": destination.name,
                    "sha256": _sha256(destination),
                    "bytes": destination.stat().st_size,
                }
            )

        manifest = {
            "schema_version": 1,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "source_experiment": experiment_id,
            "source_config": config,
            "ensemble_method": "mean",
            "models": models,
        }
        manifest_path = bundle_dir / "manifest.json"
        _atomic_write_json(manifest_path, manifest)

        latest_path = self.production_root / "latest.json"
        _atomic_write_json(
            latest_path,
            {
                "schema_version": 1,
                "bundle": experiment_id,
                "manifest_sha256": _sha256(manifest_path),
            },
        )
        return ProductionModelResult(
            bundle_dir=bundle_dir,
            manifest_path=manifest_path,
            latest_path=latest_path,
            model_count=len(models),
        )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
