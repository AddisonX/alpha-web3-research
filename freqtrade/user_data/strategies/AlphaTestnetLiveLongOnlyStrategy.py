from __future__ import annotations

import logging
import os
from pathlib import Path

from alpha_web3_research.live.testnet import validate_testnet_config
from AlphaCrossSectionalLiveLongOnlyStrategy import AlphaCrossSectionalLiveLongOnlyStrategy

logger = logging.getLogger(__name__)


class AlphaTestnetLiveLongOnlyStrategy(AlphaCrossSectionalLiveLongOnlyStrategy):
    """Allow entries only for the isolated, capped Binance Futures Testnet instance."""

    def _execution_mode_allowed(self) -> bool:
        try:
            validate_testnet_config(self.config)
            if not Path(os.environ["ALPHA_TESTNET_ARM_FILE"]).is_file():
                raise ValueError("testnet arm file is absent")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            logger.error("Testnet entry blocked: %s", exc)
            return False
        return True
