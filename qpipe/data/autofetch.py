"""Auto-resolve missing market data: catalog -> Alpaca -> yfinance.

Before a backtest/optimization runs, ensure_data() checks every symbol the
config needs. Already in the catalog = untouched. Missing = fetched from
Alpaca (if keys are configured) or yfinance as fallback, then written into
the catalog so it's never fetched again.

API keys: env vars ALPACA_API_KEY/ALPACA_SECRET_KEY, or the settings file
(~/.qpipe.json) written by the web UI's Data tab.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SETTINGS_FILE = Path.home() / ".qpipe.json"
DEFAULT_START = "2015-01-01"


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_settings(settings: dict) -> None:
    merged = load_settings() | settings
    SETTINGS_FILE.write_text(json.dumps(merged, indent=2))
    SETTINGS_FILE.chmod(0o600)  # keys in plaintext: owner-only


def _apply_keys_to_env() -> bool:
    """Returns True if Alpaca keys are available (env or settings file)."""
    s = load_settings()
    if s.get("alpaca_key") and not os.environ.get("ALPACA_API_KEY"):
        os.environ["ALPACA_API_KEY"] = s["alpaca_key"]
        os.environ["ALPACA_SECRET_KEY"] = s.get("alpaca_secret", "")
    return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))


def have_symbol(catalog_path: str, symbol: str, venue: str, bar_spec: str) -> bool:
    from qpipe.backtest.runner import load_market

    try:
        load_market(catalog_path, f"{symbol}.{venue}-{bar_spec}-EXTERNAL")
        return True
    except Exception:  # noqa: BLE001
        return False


def _fetch_yfinance(catalog_path: str, symbol: str, venue: str, start: str) -> int:
    import yfinance as yf

    from qpipe.data.catalog import write_bars_df

    df = yf.download(symbol.replace(".", "-"), start=start, auto_adjust=True, progress=False)  # BRK.B -> BRK-B
    if df is None or df.empty:
        raise ValueError(f"yfinance returned no data for {symbol}")
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    # guard against bad rows that violate OHLC invariants
    df = df[(df.high >= df[["open", "close", "low"]].max(axis=1)) & (df.low <= df[["open", "close"]].min(axis=1))]
    if df.empty:
        raise ValueError(f"no clean rows for {symbol}")
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return write_bars_df(catalog_path, symbol, venue, "1-DAY-LAST", df)


def _purge_symbol(catalog_path: str, symbol: str, venue: str, bar_spec: str) -> None:
    """Remove a symbol's bars + instrument from the catalog (before a force refetch,
    so overlapping date ranges don't create duplicate bars)."""
    import shutil

    root = Path(catalog_path) / "data"
    for d in (root / "bar" / f"{symbol}.{venue}-{bar_spec}-EXTERNAL",
              root / "equity" / f"{symbol}.{venue}"):
        if d.exists():
            shutil.rmtree(d)


def update_catalog(catalog_path: str, log=print) -> dict:
    """Top up every symbol already in the catalog to the latest available daily bar.

    Designed for a daily cron job: fetches only bars AFTER each symbol's last
    stored date (Alpaca if keys configured, yfinance fallback), appends to the
    catalog. Symbols already current are skipped.
    """
    import pandas as pd

    from qpipe.backtest.runner import load_market
    from qpipe.data.catalog import write_bars_df

    bar_root = Path(catalog_path) / "data" / "bar"
    if not bar_root.exists():
        return {}
    alpaca_ok = _apply_keys_to_env()
    status: dict = {}
    for d in sorted(bar_root.iterdir()):
        if not d.is_dir() or "1-DAY-LAST" not in d.name:
            continue
        instrument = d.name.split("-")[0]
        sym, venue = instrument.rsplit(".", 1)
        try:
            _, bars = load_market(catalog_path, d.name)
            last = pd.Timestamp(bars[-1].ts_event, unit="ns", tz="UTC")
        except Exception as e:  # noqa: BLE001
            status[sym] = f"FAILED: {e}"
            continue
        start = (last + pd.Timedelta(days=1)).date().isoformat()
        if pd.Timestamp(start, tz="UTC") >= pd.Timestamp.utcnow().normalize():
            status[sym] = "current"
            continue
        fetched = False
        if alpaca_ok:
            try:
                from qpipe.data.alpaca_loader import ingest_alpaca
                n = ingest_alpaca(catalog_path, [sym], start, venue=venue)
                if n.get(sym):
                    status[sym] = f"alpaca +{n[sym]}"
                    fetched = True
            except Exception:  # noqa: BLE001
                pass
        if not fetched:
            try:
                import yfinance as yf
                df = yf.download(sym.replace(".", "-"), start=start, auto_adjust=True, progress=False)
                if df is not None and not df.empty:
                    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
                        df.columns = df.columns.get_level_values(0)
                    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
                    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
                    df = df[df.index > last]  # strictly new bars only — no duplicates
                    if len(df):
                        write_bars_df(catalog_path, sym, venue, "1-DAY-LAST", df)
                        status[sym] = f"yfinance +{len(df)}"
                    else:
                        status[sym] = "current"
                else:
                    status[sym] = "current"
            except Exception as e:  # noqa: BLE001
                status[sym] = f"FAILED: {e}"
        log(f"  {sym}: {status[sym]}")
    load_market.cache_clear()
    return status


def ensure_data(catalog_path: str, run_cfg, start: str | None = None, log=print,
                force: bool = False) -> dict:
    """Make sure every symbol the run config needs exists in the catalog.

    Returns {symbol: "catalog" | "alpaca" | "yfinance" | "FAILED: <why>"}.
    Only daily bars are auto-fetched; intraday should be ingested explicitly.
    force=True refetches everything from `start` via yfinance (deepest history)
    after purging existing data — use it to extend history earlier than Alpaca's
    ~2016 IEX limit.
    """
    if run_cfg.bar_spec != "1-DAY-LAST":
        return {}
    s = load_settings()
    if start is None:
        start = s.get("autofetch_start") or DEFAULT_START
    symbols = list(dict.fromkeys([run_cfg.symbol] + list(run_cfg.universe)))
    if not force and s.get("auto_fetch", True) is False:
        log("auto-fetch disabled in Settings; using catalog only")
        return {sym: "catalog" for sym in symbols
                if have_symbol(catalog_path, sym, run_cfg.venue, run_cfg.bar_spec)}
    if force:
        status = {}
        for sym in symbols:
            try:
                log(f"refetching {sym} from {start} via yfinance…")
                _purge_symbol(catalog_path, sym, run_cfg.venue, run_cfg.bar_spec)
                _fetch_yfinance(catalog_path, sym, run_cfg.venue, start)
                status[sym] = "yfinance"
            except Exception as e:  # noqa: BLE001
                status[sym] = f"FAILED: {e}"
                log(f"  refetch failed for {sym}: {e}")
        from qpipe.backtest.runner import load_market
        load_market.cache_clear()
        return status
    missing = [s for s in symbols if not have_symbol(catalog_path, s, run_cfg.venue, run_cfg.bar_spec)]
    status = {s: "catalog" for s in symbols if s not in missing}
    if not missing:
        return status

    alpaca_ok = _apply_keys_to_env()
    for sym in missing:
        fetched = False
        if alpaca_ok:
            try:
                log(f"fetching {sym} from Alpaca…")
                from qpipe.data.alpaca_loader import ingest_alpaca
                n = ingest_alpaca(catalog_path, [sym], start, venue=run_cfg.venue)
                if n.get(sym):
                    status[sym] = "alpaca"
                    fetched = True
            except Exception as e:  # noqa: BLE001
                log(f"  alpaca failed for {sym}: {e}")
        if not fetched:
            try:
                log(f"fetching {sym} from yfinance…")
                _fetch_yfinance(catalog_path, sym, run_cfg.venue, start)
                status[sym] = "yfinance"
                fetched = True
            except Exception as e:  # noqa: BLE001
                status[sym] = f"FAILED: {e}"
                log(f"  yfinance failed for {sym}: {e}")
    # new data on disk: clear the per-process bar cache so it gets re-read
    from qpipe.backtest.runner import load_market
    load_market.cache_clear()
    return status
