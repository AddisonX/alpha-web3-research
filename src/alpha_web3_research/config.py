from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or an optional env file."""

    model_config = SettingsConfigDict(
        env_prefix="ALPHA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    data_path: Path = Path("data")
    reports_path: Path = Path("reports")

    data_provider: str = "auto"
    exchange_id: str = "binanceusdm"
    market_type: str = "swap"
    quote_asset: str = "USDT"
    sandbox: bool = False
    enable_rate_limit: bool = True
    ccxt_timeout_ms: int = 30_000

    binance_api_key: str | None = None
    binance_secret: str | None = None

    def resolved_data_path(self) -> Path:
        return self._resolve(self.data_path)

    def resolved_reports_path(self) -> Path:
        return self._resolve(self.reports_path)

    def _resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.project_root / path

    def public_dump(self) -> dict[str, Any]:
        data = self.model_dump()
        data["binance_api_key"] = "***" if self.binance_api_key else None
        data["binance_secret"] = "***" if self.binance_secret else None
        data["data_path"] = str(self.resolved_data_path())
        data["reports_path"] = str(self.resolved_reports_path())
        data["project_root"] = str(self.project_root)
        return data


def load_settings(config_file: Path | None = None) -> Settings:
    project_root = find_project_root()
    kwargs: dict[str, Any] = {"project_root": project_root}
    if config_file is not None:
        kwargs["_env_file"] = config_file
    return Settings(**kwargs)


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for path in (current, *current.parents):
        if (path / "pyproject.toml").exists():
            return path
    return current
