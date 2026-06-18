"""Point-in-time shares outstanding from SEC EDGAR (free, no key).

Source: each 10-Q/10-K cover page reports shares outstanding; the XBRL API serves
the series with `filed` dates. Using the latest value FILED on or before a date
gives a legitimately point-in-time share count — combined with the day's close it
yields PIT market cap, so a strategy can rank a universe by market cap at each
rebalance with no lookahead and no hand-maintained membership file.

Caveats: cover-page counts are quarterly granularity (fine — share counts move
slowly; rank changes between filings are price-driven). Multi-class names
(GOOG/GOOGL, BRK) report per-class counts, so class-share market caps are
understated; either accept that or exclude them from candidates.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "qpipe research (self-hosted backtester) contact@example.com"}
TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/dei/EntityCommonStockSharesOutstanding.json"
# Fallback for multi-class filers (META, GOOGL...) that don't report the cover-page
# concept: weighted-average basic shares (the EPS denominator) aggregates all classes.
FALLBACK_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/WeightedAverageNumberOfSharesOutstandingBasic.json"
GAAP_ISSUED_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/CommonStockSharesIssued.json"


def _headers() -> dict:
    """SEC asks for a contact in the User-Agent; configurable in Settings."""
    try:
        from qpipe.data.autofetch import load_settings
        contact = load_settings().get("edgar_contact")
        if contact:
            return {"User-Agent": f"qpipe research {contact}"}
    except Exception:  # noqa: BLE001
        pass
    return UA


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def ticker_to_cik() -> dict[str, int]:
    data = _get(TICKER_URL)
    return {v["ticker"].upper(): v["cik_str"] for v in data.values()}


def fetch_shares(symbols: list[str], out_file: str | Path, log=print, adjust_splits: bool = True) -> dict:
    """Fetch PIT shares-outstanding series for symbols -> JSON file.

    Output format: {SYM: [[filed_date, shares], ...] ascending by filed date}.
    """
    ciks = ticker_to_cik()
    out: dict[str, list] = {}
    for sym in symbols:
        cik = ciks.get(sym.upper().replace(".", "-")) or ciks.get(sym.upper())
        if cik is None:
            log(f"  ! no CIK for {sym}, skipping")
            continue
        candidates = []
        for url, label in ((CONCEPT_URL, "cover-page"), (FALLBACK_URL, "wavg-basic"),
                           (GAAP_ISSUED_URL, "shares-issued")):
            try:
                data = _get(url.format(cik=cik))
                facts = data.get("units", {}).get("shares", [])
                by_filed: dict = {}
                for f in facts:
                    if not f.get("val"):
                        continue
                    cur = by_filed.get(f["filed"])
                    if cur is None or f.get("end", "") > cur[0]:
                        by_filed[f["filed"]] = (f.get("end", ""), int(f["val"]))
                if by_filed:
                    candidates.append((label, sorted((d, v) for d, (_, v) in by_filed.items())))
            except Exception:  # noqa: BLE001
                continue
            time.sleep(0.12)
        if candidates:
            # best = freshest last filing, then longest history
            label, series = max(candidates, key=lambda c: (c[1][-1][0], len(c[1])))
            if adjust_splits:
                series = _split_adjust(sym, series, log)
            out[sym] = [[d, v] for d, v in series]
            log(f"  {sym}: {len(series)} filings via {label} ({series[0][0]} .. {series[-1][0]})")
        else:
            log(f"  ! {sym}: no share data on EDGAR")
        time.sleep(0.15)  # SEC fair-use pacing
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(json.dumps(out, indent=1))
    log(f"wrote {out_file} ({len(out)} symbols)")
    return out


def _split_adjust(sym: str, series: list, log=print) -> list:
    """Scale as-filed share counts into today's (split-adjusted) units.

    Catalog prices are split-adjusted, so market cap = adj_close x shares only
    works if historical share counts are multiplied by every split that happened
    AFTER their filing date (mcap itself is split-invariant)."""
    try:
        import yfinance as yf
        spl = yf.Ticker(sym.replace(".", "-")).splits
        events = [(d.date().isoformat(), float(r)) for d, r in spl.items() if float(r) > 0]
    except Exception as e:  # noqa: BLE001
        log(f"  ! {sym}: split lookup failed ({e}); share counts left as-filed")
        return series
    if not events:
        return series
    out = []
    for filed, val in series:
        f = 1.0
        for d, r in events:
            if d > filed:
                f *= r
        out.append((filed, int(val * f)))
    return out


def shares_asof(series: list, date_iso: str) -> int | None:
    """Latest share count FILED on or before date_iso (point-in-time correct)."""
    val = None
    for filed, v in series:
        if filed <= date_iso:
            val = v
        else:
            break
    return val
