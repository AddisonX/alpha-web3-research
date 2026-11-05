from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from alpha_web3_research.config import Settings
from alpha_web3_research.data.contracts import BarQuery, BarRequest, InstrumentKey
from alpha_web3_research.data.ohlcv import OHLCVFetcher
from alpha_web3_research.data.providers import create_ohlcv_fetcher
from alpha_web3_research.data.providers.binance import BinanceProvider
from alpha_web3_research.data.reader import MarketDataReader
from alpha_web3_research.data.storage import ParquetOHLCVStore
from alpha_web3_research.factors.hourly import HourlyMultiTimeframeFactorBuilder
from alpha_web3_research.labels.hourly import HourlyMultiHorizonLabelBuilder

HOUR = 3_600_000
START = 1_735_689_600_000
SYMBOL = "BTC/USDT:USDT"


class RecordedExchange:
    """Deterministic local exchange response, including a repeated boundary row."""
    id = "binanceusdm"
    timeframes = {"1h": "1h", "4h": "4h"}

    def __init__(self):
        self.rows = [
            [START + i * HOUR, 100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i, 10.0 + i % 7]
            for i in range(160)
        ]
        self.rows.insert(10, self.rows[9].copy())
        self.calls = []

    def load_markets(self):
        return {SYMBOL: {"id": "BTCUSDT", "swap": True, "linear": True,
                         "base": "BTC", "quote": "USDT", "settle": "USDT",
                         "contractSize": 1.0, "active": True}}

    def parse_timeframe(self, value):
        return {"1h": 3600, "4h": 14400}[value]

    def milliseconds(self):
        return START + 159 * HOUR + HOUR // 2

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls.append(since)
        return [row for row in self.rows if row[0] >= since][:limit]


def settings(tmp_path):
    return Settings(_env_file=None, project_root=tmp_path)


def frame(exchange=None):
    return OHLCVFetcher(exchange or RecordedExchange(), Settings(_env_file=None)).fetch_ohlcv_frame(
        SYMBOL, "1h", "2025-01-01", limit_per_request=17,
    )


def test_binance_adapter_matches_legacy_pagination_and_closed_bars(tmp_path):
    old_exchange, new_exchange = RecordedExchange(), RecordedExchange()
    provider = BinanceProvider(new_exchange, settings(tmp_path))
    batch = provider.fetch_bars(
        BarRequest(InstrumentKey("binanceusdm", "swap", SYMBOL), "1h", START),
        limit_per_request=17,
    )
    assert_frame_equal(frame(old_exchange).drop("fetched_at"), batch.frame.drop("fetched_at"))
    assert new_exchange.calls == old_exchange.calls
    assert batch.frame.height == 159
    assert batch.frame["timestamp"].max() == START + 158 * HOUR
    assert batch.provider_id == "binance_official_ccxt"
    instrument = provider.list_instruments()[0]
    assert instrument.native_symbol == "BTCUSDT"
    assert instrument.key.instrument_id != InstrumentKey("hyperliquid", "swap", SYMBOL).instrument_id
    assert not provider.capabilities().supports_live


def test_half_open_bounds_and_unclosed_opt_in(tmp_path):
    provider = BinanceProvider(RecordedExchange(), settings(tmp_path))
    key = InstrumentKey("binanceusdm", "swap", SYMBOL)
    batch = provider.fetch_bars(BarRequest(key, "1h", START + 2 * HOUR, START + 5 * HOUR))
    assert batch.frame["timestamp"].to_list() == [START + i * HOUR for i in range(2, 5)]
    assert provider.fetch_bars(BarRequest(key, "1h", START, include_unclosed=True)).frame.height == 160


@pytest.mark.parametrize("key,timeframe", [
    (InstrumentKey("hyperliquid", "swap", SYMBOL), "1h"),
    (InstrumentKey("binanceusdm", "spot", SYMBOL), "1h"),
    (InstrumentKey("binanceusdm", "swap", "UNKNOWN"), "1h"),
    (InstrumentKey("binanceusdm", "swap", SYMBOL), "7h"),
])
def test_reject_unsupported_before_fetch(tmp_path, key, timeframe):
    exchange = RecordedExchange()
    with pytest.raises(ValueError):
        BinanceProvider(exchange, settings(tmp_path)).fetch_bars(BarRequest(key, timeframe, START))
    assert exchange.calls == []


def test_factory_and_invalid_ranges(tmp_path):
    config = settings(tmp_path)
    assert isinstance(create_ohlcv_fetcher(RecordedExchange(), config), BinanceProvider)
    config.data_provider = "typo"
    with pytest.raises(ValueError, match="unsupported data_provider"):
        create_ohlcv_fetcher(RecordedExchange(), config)
    with pytest.raises(ValueError):
        BarRequest(InstrumentKey("binanceusdm", "swap", SYMBOL), "1h", START, START)
    with pytest.raises(ValueError):
        BarQuery("binanceusdm", "swap", "1h", start_ms=START, end_ms=START)


def test_storage_query_identity_bounds_and_idempotence(tmp_path):
    store = ParquetOHLCVStore(settings(tmp_path))
    original = frame()
    other = original.with_columns(pl.lit("hyperliquid").alias("exchange"),
                                  (pl.col("close") * 2).alias("close"))
    store.upsert_ohlcv(original)
    store.upsert_ohlcv(original)
    store.upsert_ohlcv(other)
    assert store.read_ohlcv().height == original.height * 2
    reader = MarketDataReader(store)
    query = BarQuery("binanceusdm", "swap", "1h", (SYMBOL,), START + HOUR, START + 4 * HOUR)
    assert_frame_equal(reader.read_bars(query), original.slice(1, 3))
    assert reader.read_bars(BarQuery("binanceusdm", "swap", "1h", ())).is_empty()
    assert reader.read_bars(BarQuery("missing", "swap", "1h")).schema["timestamp"] == pl.Int64
    assert_frame_equal(reader.read_bars(BarQuery("hyperliquid", "swap", "1h")), other)


def test_hourly_features_and_labels_equal_old_entry_and_isolate_new_market(tmp_path):
    config = settings(tmp_path)
    store = ParquetOHLCVStore(config)
    original = frame()
    store.upsert_ohlcv(original)
    factors = HourlyMultiTimeframeFactorBuilder(config, store)
    labels = HourlyMultiHorizonLabelBuilder(config, store)
    with patch.object(MarketDataReader, "read_bars", return_value=store.read_ohlcv()):
        expected_factors, _ = factors.build(universe_only=False, write=False)
        expected_labels, _ = labels.build(universe_only=False, write=False)
    store.upsert_ohlcv(original.with_columns(pl.lit("hyperliquid").alias("exchange"),
                                            (pl.col("close") * 5).alias("close")))
    actual_factors, _ = factors.build(universe_only=False, write=False)
    actual_labels, _ = labels.build(universe_only=False, write=False)
    assert_frame_equal(expected_factors.drop("generated_at"), actual_factors.drop("generated_at"))
    assert_frame_equal(expected_labels.drop("generated_at"), actual_labels.drop("generated_at"))
    assert actual_factors["exchange"].unique().to_list() == ["binanceusdm"]
