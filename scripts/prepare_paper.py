"""Prepare published dry-run profiles offline; never start workers or overwrite data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from src.live.instances import prepare_instance

ROOT = Path(__file__).resolve().parents[1]


def verify_bundle(root: Path) -> dict:
    manifest = json.loads((root / "configs/paper/bundle_manifest.json").read_text())
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("published asset path escapes project root")
        content = path.read_bytes()
        if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"published asset checksum mismatch: {item['path']}")
    return manifest


def prepare_paper(root: Path = ROOT, names: list[str] | None = None) -> list[Path]:
    bundle = verify_bundle(root)
    profiles = {item["instance"]: item for item in bundle["instances"]}
    selected = list(profiles) if names is None else names
    if not selected or set(selected) - profiles.keys():
        raise ValueError("unknown or empty published profile selection")
    result = []
    for name in selected:
        profile = profiles[name]
        instance = prepare_instance(root, strategy_id=profile["strategy_id"], instance_name=name)
        if name == "xgb15-paper":
            source = root / "configs/paper/xgb15-paper/caps_seed.parquet"
            target = root / "data/imports/xgb15/runtime/caps.parquet"
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary = tempfile.mkstemp(dir=target.parent, suffix=".parquet")
                os.close(descriptor)
                try:
                    shutil.copyfile(source, temporary)
                    try:
                        os.link(temporary, target)  # Atomic publication, never replace existing observations.
                    except FileExistsError:
                        pass
                finally:
                    Path(temporary).unlink(missing_ok=True)
        result.append(instance)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", action="append", help="Prepare only this published instance; repeatable")
    args = parser.parse_args()
    print(json.dumps({"prepared": [str(path) for path in prepare_paper(names=args.instance)],
                      "workers_started": False}, indent=2))


if __name__ == "__main__":
    main()
