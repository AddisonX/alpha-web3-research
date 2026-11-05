from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpha_web3_research.live.testnet import (
    load_testnet_credentials,
    preflight_testnet_account,
    validate_testnet_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate isolated Binance Futures Testnet execution")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--account", action="store_true", help="Check the signed Testnet account is funded and flat")
    args = parser.parse_args()
    validate_testnet_config(json.loads(args.config.read_text(encoding="utf-8")))
    if args.credentials is not None:
        load_testnet_credentials(args.credentials)
    if args.account:
        if args.credentials is None:
            parser.error("--account requires --credentials")
        print(json.dumps(preflight_testnet_account(args.config, args.credentials), sort_keys=True))
    else:
        print("Testnet routing and credential shape: OK; signed account check not run")


if __name__ == "__main__":
    main()
