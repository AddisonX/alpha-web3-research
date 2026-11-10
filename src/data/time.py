from __future__ import annotations

from datetime import datetime, timezone


def parse_timestamp_ms(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized.isdigit():
        raw = int(normalized)
        return raw if raw > 10_000_000_000 else raw * 1000
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    if "T" not in normalized and len(normalized) == 10:
        normalized = f"{normalized}T00:00:00+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def format_timestamp_ms(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()

