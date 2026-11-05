"""Market-scoped query boundary shared by hourly factors and labels."""
from alpha_web3_research.data.contracts import BarQuery
from alpha_web3_research.data.storage import ParquetOHLCVStore
import polars as pl


class MarketDataReader:
    def __init__(self, store: ParquetOHLCVStore) -> None:
        self.store = store

    def read_bars(self, query: BarQuery) -> pl.DataFrame:
        return self.store.read_bars(query)
