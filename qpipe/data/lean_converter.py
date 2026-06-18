"""Convert QuantConnect LEAN daily/hourly equity zips into the Nautilus catalog.

LEAN layout (downloaded via `lean data download`):
    <lean-data-dir>/equity/usa/daily/aapl.zip   -> aapl.csv
    <lean-data-dir>/equity/usa/hour/aapl.zip    -> aapl.csv

CSV rows: "YYYYMMDD HH:MM,open,high,low,close,volume" with prices scaled x10000
(deci-cents). Timestamps are US/Eastern; we convert to UTC.

Note: minute data uses a different per-day layout and is intentionally not
supported here — pull minute bars from Alpaca instead (see alpaca_loader).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from qpipe.data.catalog import write_bars_df

_SCALE = 10_000.0
_SPEC = {"daily": "1-DAY-LAST", "hour": "1-HOUR-LAST"}


def read_lean_zip(zip_path: str | Path) -> pd.DataFrame:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name)
    df = pd.read_csv(
        io.BytesIO(raw),
        header=None,
        names=["time", "open", "high", "low", "close", "volume"],
    )
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d %H:%M")
    df = df.set_index("time")
    df.index = df.index.tz_localize("US/Eastern").tz_convert("UTC")
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] / _SCALE
    df["volume"] = df["volume"].astype(float)
    return df


def ingest_lean(
    lean_data_dir: str | Path,
    catalog_path: str,
    symbols: list[str],
    resolution: str = "daily",  # daily | hour
    venue: str = "XNAS",
) -> dict[str, int]:
    if resolution not in _SPEC:
        raise ValueError(f"resolution must be one of {list(_SPEC)} (minute: use Alpaca)")
    base = Path(lean_data_dir) / "equity" / "usa" / resolution
    counts: dict[str, int] = {}
    for sym in symbols:
        zp = base / f"{sym.lower()}.zip"
        if not zp.exists():
            print(f"  ! missing {zp}, skipping")
            continue
        df = read_lean_zip(zp)
        counts[sym] = write_bars_df(catalog_path, sym.upper(), venue, _SPEC[resolution], df)
        print(f"  {sym.upper()}: {counts[sym]} bars")
    return counts
