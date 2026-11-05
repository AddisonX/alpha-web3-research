from pathlib import Path

from alpha_web3_research.config import load_settings
from alpha_web3_research.research.pit_rank import PITRankResearchConfig, prepare_pit_rank_data


if __name__ == "__main__":
    settings = load_settings()
    config = PITRankResearchConfig.load(settings.project_root / "strategies/configs/pit_rank64_202501_202610.v1.json")
    prepare_pit_rank_data(settings, config, workers=12, progress=lambda message: print(message, flush=True))
