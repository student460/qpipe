"""Shared helpers for writing bar data into a Nautilus ParquetDataCatalog."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider


def write_bars_df(
    catalog_path: str | Path,
    symbol: str,
    venue: str,
    bar_spec: str,
    df: pd.DataFrame,
) -> int:
    """Write an OHLCV DataFrame (UTC DatetimeIndex; open/high/low/close/volume) to the catalog.

    Returns the number of bars written.
    """
    Path(catalog_path).mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(str(catalog_path))

    instrument = TestInstrumentProvider.equity(symbol, venue)
    bar_type = BarType.from_str(f"{instrument.id}-{bar_spec}-EXTERNAL")

    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    wrangler = BarDataWrangler(bar_type=bar_type, instrument=instrument)
    bars = wrangler.process(df)

    catalog.write_data([instrument])
    catalog.write_data(bars)
    return len(bars)
