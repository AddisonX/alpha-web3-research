"""Canonical instance locations; shared code never owns runtime files."""
from pathlib import Path

from src.strategies.base import DEFAULT_STRATEGY_ID


def default_instance_name(strategy_id: str = DEFAULT_STRATEGY_ID) -> str:
    return {
        DEFAULT_STRATEGY_ID: "tokyo-lightgbm",
        "return-1-reversal-long-only-v1": "return-1-reversal",
    }.get(strategy_id, strategy_id)


def instance_directory(root: Path, strategy_id: str = DEFAULT_STRATEGY_ID) -> Path:
    return root / "freqtrade/instances" / default_instance_name(strategy_id)
