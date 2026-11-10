"""Backward-compatible entry point for the existing reversal dry-run instance."""
# Re-export existing helper imports used by acceptance scripts and tests.
from src.live.instances import (
    prepare_instance, instance_info, instance_path, database_path,
    environment, trade_command, startup_check, run,
)


def main():
    from scripts.run_dryrun import main as run_main
    from src.strategies.reversal import STRATEGY_ID
    run_main(STRATEGY_ID, "return-1-reversal")


if __name__ == "__main__":
    main()
