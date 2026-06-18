"""Download historical bars from Alpaca (free IEX feed) into the Nautilus catalog.

Requires env vars ALPACA_API_KEY and ALPACA_SECRET_KEY (any Alpaca account works;
market-data keys don't need a funded brokerage).
"""

from __future__ import annotations

import os

import pandas as pd

from qpipe.data.catalog import write_bars_df

_SPEC = {"1d": "1-DAY-LAST", "1h": "1-HOUR-LAST", "1m": "1-MINUTE-LAST"}


def ingest_alpaca(
    catalog_path: str,
    symbols: list[str],
    start: str,
    end: str | None = None,
    timeframe: str = "1d",  # 1d | 1h | 1m
    venue: str = "XNAS",
    adjustment: str = "all",  # split+dividend adjusted
) -> dict[str, int]:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars")

    tf = {
        "1d": TimeFrame.Day,
        "1h": TimeFrame.Hour,
        "1m": TimeFrame(1, TimeFrameUnit.Minute),
    }[timeframe]

    client = StockHistoricalDataClient(key, secret)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=tf,
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC") if end else None,
        adjustment=adjustment,
    )
    barset = client.get_stock_bars(req).df  # MultiIndex (symbol, timestamp)

    counts: dict[str, int] = {}
    for sym in symbols:
        if sym not in barset.index.get_level_values(0):
            print(f"  ! no data for {sym}, skipping")
            continue
        df = barset.xs(sym, level=0)[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
        counts[sym] = write_bars_df(catalog_path, sym, venue, _SPEC[timeframe], df)
        print(f"  {sym}: {counts[sym]} bars")
    return counts
