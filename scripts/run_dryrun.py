"""Run a named dry-run instance with isolated, structured runtime directories."""
import argparse
import json
import os
from pathlib import Path

from src.live.instances import prepare_instance, instance_path, run, startup_check
from src.strategies.base import DEFAULT_STRATEGY_ID
from src.strategies.reversal import STRATEGY_ID as REVERSAL_ID
from src.strategies.xgb15 import STRATEGY_ID as XGB15_ID


def main(default_strategy=DEFAULT_STRATEGY_ID, default_instance="tokyo-lightgbm"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=[DEFAULT_STRATEGY_ID, REVERSAL_ID, XGB15_ID], default=default_strategy)
    parser.add_argument("--pairs-path", type=Path)
    parser.add_argument("--instance", default=default_instance)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check-startup", action="store_true")
    args = parser.parse_args()
    instance = prepare_instance(strategy_id=args.strategy, instance_name=args.instance, source_pairs=args.pairs_path, structured=True)
    if args.prepare_only:
        print(instance)
        return
    runtime = instance_path(instance, "runtime")
    with (runtime / "supervisor.lock").open("a") as lock:
        try:
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            raise SystemExit("This instance already has a supervisor")
        (runtime / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        if args.check_startup:
            report = startup_check(instance)
            print(json.dumps(report, indent=2))
            raise SystemExit(0 if report["ready"] else 2)
        raise SystemExit(run(instance))


if __name__ == "__main__":
    main()
