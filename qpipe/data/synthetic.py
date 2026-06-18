"""Synthetic GBM daily bars — for testing the pipeline without real data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qpipe.data.catalog import write_bars_df


def make_gbm_daily(years: int = 10, s0: float = 100.0, mu: float = 0.07, sigma: float = 0.25, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.utcnow().normalize()
    idx = pd.bdate_range(end - pd.DateOffset(years=years), end, tz="UTC")
    n = len(idx)
    dt = 1 / 252
    rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    close = s0 * np.exp(np.cumsum(rets))
    open_ = np.r_[s0, close[:-1]]
    spread = np.abs(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    vol = rng.integers(1e5, 5e6, n).astype(float)
    return pd.DataFrame(
        {"open": open_.round(2), "high": high.round(2), "low": low.round(2), "close": close.round(2), "volume": vol},
        index=idx,
    )


def ingest_synthetic(catalog_path: str, symbol: str = "TEST", venue: str = "XNAS", years: int = 10, seed: int = 7) -> int:
    return write_bars_df(catalog_path, symbol, venue, "1-DAY-LAST", make_gbm_daily(years=years, seed=seed))
