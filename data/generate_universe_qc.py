# Run this in a QuantConnect RESEARCH NOTEBOOK to generate a precise
# point-in-time top-10-by-market-cap membership file from QC's fundamental data.
# Paste the printed JSON into quant-pipeline/data/universe_top10.json.
#
# Monthly snapshots, 2016 -> today. BRK.B is excluded (ticker parsing).

import json
from datetime import datetime

qb = QuantBook()

EXCLUDE = {"BRK.B", "BRK.A"}
entries = []
prev = None
for year in range(2016, datetime.now().year + 1):
    for month in range(1, 13):
        d = datetime(year, month, 1)
        if d > datetime.now():
            break
        # Top US equities by dollar volume as candidate pool, then rank by market cap
        candidates = qb.universe_history(
            qb.add_universe(qb.universe.dollar_volume.top(60)), d, d
        )
        caps = {}
        for day in candidates:
            for f in day:
                t = str(f.symbol.value)
                if t in EXCLUDE or f.market_cap is None:
                    continue
                caps[t] = f.market_cap
        top10 = sorted(caps, key=caps.get, reverse=True)[:10]
        if top10 and top10 != prev:
            entries.append({"from": d.strftime("%Y-%m-%d"), "symbols": top10})
            prev = top10

print(json.dumps(entries, indent=2))
