from __future__ import annotations

STABLE_BASE_ASSETS = {
    "BUSD",
    "DAI",
    "FDUSD",
    "TUSD",
    "USDC",
    "USDE",
    "USDP",
    "USDS",
    "USDT",
}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "4L", "4S", "5L", "5S")
NON_CRYPTO_BASE_ASSETS = {
    "AAPL",
    "AMD",
    "AMZN",
    "AVGO",
    "COIN",
    "CRCL",
    "GOOGL",
    "MCD",
    "META",
    "MSFT",
    "NFLX",
    "NVDA",
    "QQQ",
    "SPX",
    "TSLA",
}


def is_stable_base(asset: str) -> bool:
    return asset in STABLE_BASE_ASSETS


def is_leveraged_base(asset: str) -> bool:
    return asset.endswith(LEVERAGED_SUFFIXES)


def is_non_crypto_base(asset: str) -> bool:
    return asset in NON_CRYPTO_BASE_ASSETS
