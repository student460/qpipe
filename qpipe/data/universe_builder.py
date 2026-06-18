"""Auto-derive the candidate pool for a market-cap-ranked universe. No hand-curated lists.

Method:
1. Seed = current S&P 500 members (Wikipedia).
2. Bulk point-in-time share counts per quarter from SEC EDGAR *frames* API
   (one request per quarter covers every filer). Multi-class filers missing from
   frames (GOOGL, META...) are filled per-ticker via qpipe.data.edgar.
3. Unadjusted quarterly closes from yfinance (batched) — raw price x as-filed
   shares = split-invariant market cap.
4. Every quarter's top (top_n + buffer) by market cap is collected; the UNION is
   the pool — every name that ever plausibly ranked. The strategy then computes
   the actual top-N at each rebalance from PIT data, exactly as it would live.

Caveat: seeding from the CURRENT S&P 500 misses ever-top-N names that have since
delisted — at top-10 scale this has been a non-issue for the past decade, but it
is mild survivorship bias; widen `buffer` to be safe.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import time
import urllib.request

from qpipe.data.edgar import UA, _headers, fetch_shares, shares_asof, ticker_to_cik

FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/CY{y}Q{q}I.json"
WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500_tickers() -> list[str]:
    import pandas as pd

    req = urllib.request.Request(WIKI_SP500, headers=_headers())
    html = urllib.request.urlopen(req, timeout=30).read().decode()
    table = pd.read_html(io.StringIO(html))[0]
    return [t.replace("-", ".") for t in table["Symbol"].astype(str).tolist()]


def _quarters(start_year: int) -> list[tuple[int, int]]:
    now = dt.date.today()
    out = []
    for y in range(start_year, now.year + 1):
        for q in (1, 2, 3, 4):
            if y == now.year and q > (now.month - 1) // 3:  # only completed quarters
                break
            out.append((y, q))
    return out


def build_universe(
    top_n: int = 10,
    buffer: int = 20,
    start_year: int = 2015,
    log=print,
) -> dict:
    """Returns {"pool": [...], "quarterly_top": {"2015Q1": [...], ...}}."""
    import pandas as pd
    import yfinance as yf

    seed = sp500_tickers()
    ciks = ticker_to_cik()
    cik_to_ticker = {ciks[t.replace(".", "-")] if t.replace(".", "-") in ciks else ciks.get(t): t
                     for t in seed if (t in ciks or t.replace(".", "-") in ciks)}
    cik_to_ticker = {c: t for c, t in cik_to_ticker.items() if c is not None}
    log(f"seed: {len(seed)} S&P 500 tickers, {len(cik_to_ticker)} matched to CIKs")

    # bulk shares per quarter
    shares_q: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    for y, q in _quarters(start_year):
        key = f"{y}Q{q}"
        try:
            req = urllib.request.Request(FRAMES_URL.format(y=y, q=q), headers=_headers())
            data = json.loads(urllib.request.urlopen(req, timeout=30).read())
            row = {}
            for f in data.get("data", []):
                t = cik_to_ticker.get(f["cik"])
                if t and f.get("val"):
                    row[t] = int(f["val"])
                    seen.add(t)
            shares_q[key] = row
            log(f"  {key}: {len(row)} seed filers")
        except Exception as e:  # noqa: BLE001
            log(f"  ! {key}: {e}")
        time.sleep(0.12)

    # multi-class filers absent from every frame: per-ticker fallback (as-filed units)
    missing = [t for t in cik_to_ticker.values() if t not in seen]
    log(f"filling {len(missing)} multi-class/missing tickers per-ticker…")
    fallback = fetch_shares(missing, "/dev/null", log=lambda m: None, adjust_splits=False) if missing else {}
    for key in shares_q:
        y, q = int(key[:4]), int(key[-1])
        qend = f"{y}-{q*3:02d}-28"
        for t, series in fallback.items():
            v = shares_asof(series, qend)
            if v:
                shares_q[key][t] = v

    # unadjusted quarterly closes, batched
    tickers = sorted({t for row in shares_q.values() for t in row})
    log(f"downloading quarterly closes for {len(tickers)} tickers…")
    px = yf.download([t.replace(".", "-") for t in tickers], start=f"{start_year}-01-01",
                     interval="1mo", auto_adjust=False, progress=False)["Close"]
    px.columns = [c.replace("-", ".") for c in px.columns]

    pool: set[str] = set()
    quarterly_top: dict[str, list[str]] = {}
    for key, row in shares_q.items():
        y, q = int(key[:4]), int(key[-1])
        month = pd.Timestamp(year=y, month=q * 3, day=1)
        if month not in px.index:
            continue
        closes = px.loc[month]
        caps = {t: row[t] * closes[t] for t in row if t in closes.index and pd.notna(closes[t])}
        top = sorted(caps, key=caps.get, reverse=True)[: top_n + buffer]
        quarterly_top[key] = top[:top_n]
        pool.update(top)
    log(f"pool: {len(pool)} names ever in the top {top_n}+{buffer}")
    return {"pool": sorted(pool), "quarterly_top": quarterly_top}
