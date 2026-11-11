"""Offline readiness/import and explicitly requested CMC updates; never start trading."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.data.xgb15_caps import refresh_caps
from src.strategies.xgb15_assets import asset_paths, check_assets, export_research_bundle, import_bundle, read_pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "export", "import", "refresh-caps"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, help="Already exported runtime package directory")
    parser.add_argument("--year", type=int)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--as-of", help="Timezone-aware ISO timestamp, for offline checks only")
    args = parser.parse_args()
    now = datetime.fromisoformat(args.as_of.replace("Z", "+00:00")) if args.as_of else datetime.now(timezone.utc)
    if now.tzinfo is None:
        parser.error("--as-of must include timezone")
    if args.as_of and args.command != "check":
        parser.error("--as-of is only allowed for offline checks")
    root = args.root.resolve()
    try:
        if args.command == "export":
            if args.research_root is None or args.output is None or args.year is None:
                parser.error("export requires --research-root, --output and --year")
            result = export_research_bundle(args.research_root.resolve(), args.output.resolve(), args.year)
        elif args.command == "import":
            if args.source is None:
                parser.error("import requires --source")
            result = import_bundle(root, args.source, args.year or now.astimezone(timezone.utc).year)
        elif args.command == "refresh-caps":
            _, private = asset_paths(root)
            report = check_assets(root, now)
            result = refresh_caps(private, read_pairs(private / "pairs.json"), report["decision_timestamp"], now)
        else:
            result = check_assets(root, now)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get("ready") is False or result.get("complete_count", 60) < 60 else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"ok": False, "error": str(error), "worker_started": False}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
