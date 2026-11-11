"""Research-compatible features; deliberately separate from legacy close-only lows."""
import numpy as np
import pandas as pd

from src.strategies.xgb15 import FEATURES, percentile_features


def build_xgb15_features(bars: pd.DataFrame, caps: pd.DataFrame, decision_timestamp: int, pairs: list[str] | None = None) -> pd.DataFrame:
    close_time = pd.Timestamp(decision_timestamp + 14_400_000, unit="ms", tz="UTC")
    last_date = (close_time - pd.Timedelta(hours=24)).floor("D").tz_localize(None) - pd.Timedelta(days=1)
    caps = caps.copy()
    caps["date"] = pd.to_datetime(caps.date).dt.tz_localize(None)
    caps = caps.loc[caps.date <= last_date].copy()
    if caps.duplicated(["symbol", "date"]).any():
        raise ValueError("duplicate cap observations")
    caps.loc[(caps.market_cap_usd <= 0) | (caps.marketcap_identity_error > .01), "market_cap_usd"] = np.nan
    if bars.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("duplicate candles")
    rows = []
    for symbol in sorted(pairs if pairs is not None else bars.symbol.unique()):
        row = {"symbol": symbol, **{f: np.nan for f in FEATURES}}
        history = caps.loc[caps.symbol == symbol].set_index("date").reindex(pd.date_range(last_date-pd.Timedelta(days=39), last_date))
        for n in (10, 20, 30, 40):
            h = history.iloc[-n:]
            ok = h.market_cap_usd.notna().all() and h.cmc_id.notna().all() and h.cmc_id.nunique() == 1
            row[f"market_cap_mean_{n}d"] = h.market_cap_usd.mean() if ok else np.nan
        all_bars = bars.loc[bars.symbol == symbol]
        p = all_bars.loc[(all_bars.timestamp <= decision_timestamp) & (all_bars.volume > 0) & (all_bars.close > 0)].sort_values("timestamp")
        if p.empty or p.timestamp.iloc[-1] != decision_timestamp:
            rows.append(row)
            continue
        segments = p.timestamp.diff().ne(14_400_000).cumsum()
        p = p.loc[segments == segments.iloc[-1]]
        if len(p) < 43:
            rows.append(row)
            continue
        ret = p.close / p.close.shift(1) - 1
        spread = (p.high - p.low) / p.close
        row.update({"mean_return_4": ret.rolling(4, min_periods=2).mean().iloc[-1],
               "range_to_quote_volume_2": (spread / ((p.close*p.volume).rolling(2, min_periods=2).mean()+1)).iloc[-1],
               "return_6": (p.close/p.close.shift(6)-1).iloc[-1],
               "turnover_relative_volume_42": (p.volume/p.volume.shift(1).rolling(42).mean()).iloc[-1],
               "volatility_4": ret.rolling(4, min_periods=2).std(ddof=1).iloc[-1]})
        for n in (12, 24):
            row[f"volume_change_{n}"] = (p.volume/p.volume.shift(n)-1).iloc[-1]
        for n in (3, 4):
            row[f"avg_range_{n}"] = spread.rolling(n, min_periods=2).mean().iloc[-1]
        for n in (4, 6):
            row[f"distance_to_low_{n}"] = (p.close/p.low.rolling(n, min_periods=2).min()-1).iloc[-1]
        rows.append(row)
    if not rows:
        raise ValueError("no closed contiguous factor history")
    return percentile_features(pd.DataFrame(rows))
