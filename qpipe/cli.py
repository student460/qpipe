"""qpipe command-line interface."""

from __future__ import annotations

import argparse
import json

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(prog="qpipe", description="Quant model IS/OOS optimization pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    ds = sub.add_parser("data-synthetic", help="Generate synthetic test data")
    ds.add_argument("--catalog", default="catalog")
    ds.add_argument("--symbol", default="TEST")
    ds.add_argument("--years", type=int, default=10)
    ds.add_argument("--seed", type=int, default=7)

    dl = sub.add_parser("data-lean", help="Convert LEAN zips (from `lean data download`) into the catalog")
    dl.add_argument("--lean-dir", required=True, help="LEAN data dir (contains equity/usa/...)")
    dl.add_argument("--catalog", default="catalog")
    dl.add_argument("--symbols", required=True, help="Comma-separated tickers")
    dl.add_argument("--resolution", default="daily", choices=["daily", "hour"])

    da = sub.add_parser("data-alpaca", help="Download bars from Alpaca into the catalog")
    da.add_argument("--catalog", default="catalog")
    da.add_argument("--symbols", required=True)
    da.add_argument("--start", required=True)
    da.add_argument("--end", default=None)
    da.add_argument("--timeframe", default="1d", choices=["1d", "1h", "1m"])

    du = sub.add_parser("data-update", help="Top up every catalog symbol to the latest daily bar (cron-friendly)")
    du.add_argument("--catalog", default="catalog")

    de = sub.add_parser("data-edgar", help="Fetch PIT shares outstanding from SEC EDGAR")
    de.add_argument("--symbols", required=True, help="Comma-separated tickers")
    de.add_argument("--out", default="data/shares.json")

    ub = sub.add_parser("universe-build", help="Auto-derive a market-cap universe pool (S&P500 + EDGAR), update a config")
    ub.add_argument("--config", required=True, help="Config name to update (universe field)")
    ub.add_argument("--top", type=int, default=10)
    ub.add_argument("--buffer", type=int, default=20)
    ub.add_argument("--start-year", type=int, default=2015)
    ub.add_argument("--configs-dir", default="configs")
    ub.add_argument("--shares-out", default="data/shares.json")

    bt = sub.add_parser("backtest", help="Run a single backtest")
    bt.add_argument("--config", required=True)
    bt.add_argument("--catalog", default="catalog")
    bt.add_argument("--params", default="{}", help='JSON params, e.g. \'{"fast_period": 10}\'')
    bt.add_argument("--start", default=None)
    bt.add_argument("--end", default=None)
    bt.add_argument("--balance", type=float, default=100_000)

    op = sub.add_parser("optimize", help="Randomized IS/OOS optimization")
    op.add_argument("--config", required=True)
    op.add_argument("--catalog", default="catalog")
    op.add_argument("--segment-months", type=int, default=6, choices=[3, 6, 9])
    op.add_argument("--windows", type=int, default=20)
    op.add_argument("--trials", type=int, default=50)
    op.add_argument("--jobs", type=int, default=4)
    op.add_argument("--gap-days", type=int, default=5)
    op.add_argument("--holdout-months", type=int, default=0)
    op.add_argument("--warmup-days", type=int, default=90, help="Calendar days of indicator warm-up before each window (no trading)")
    op.add_argument("--balance", type=float, default=100_000)
    op.add_argument("--seed", type=int, default=42)
    op.add_argument("--out", default="runs")

    sv = sub.add_parser("serve", help="Start the web UI")
    sv.add_argument("--catalog", default="catalog")
    sv.add_argument("--configs", default="configs")
    sv.add_argument("--runs", default="runs")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true", help="Auto-restart on code changes (dev)")

    mc = sub.add_parser("mcp", help="Start the MCP server (connect Claude to the pipeline)")
    mc.add_argument("--catalog", default="catalog")
    mc.add_argument("--configs", default="configs")
    mc.add_argument("--runs", default="runs")
    mc.add_argument("--host", default="127.0.0.1")
    mc.add_argument("--port", type=int, default=8765)

    args = p.parse_args()

    if args.cmd == "data-synthetic":
        from qpipe.data.synthetic import ingest_synthetic
        n = ingest_synthetic(args.catalog, args.symbol, years=args.years, seed=args.seed)
        print(f"Wrote {n} bars for {args.symbol} to {args.catalog}")

    elif args.cmd == "data-lean":
        from qpipe.data.lean_converter import ingest_lean
        ingest_lean(args.lean_dir, args.catalog, args.symbols.split(","), args.resolution)

    elif args.cmd == "data-alpaca":
        from qpipe.data.alpaca_loader import ingest_alpaca
        ingest_alpaca(args.catalog, args.symbols.split(","), args.start, args.end, args.timeframe)

    elif args.cmd == "data-update":
        from qpipe.data.autofetch import update_catalog
        s = update_catalog(args.catalog)
        fresh = sum(1 for v in s.values() if v.startswith(("alpaca", "yfinance")))
        print(f"{len(s)} symbols: {fresh} updated, {sum(1 for v in s.values() if v=='current')} current, "
              f"{sum(1 for v in s.values() if v.startswith('FAILED'))} failed")

    elif args.cmd == "data-edgar":
        from qpipe.data.edgar import fetch_shares
        fetch_shares(args.symbols.split(","), args.out)

    elif args.cmd == "universe-build":
        import yaml
        from pathlib import Path as _P
        from qpipe.data.universe_builder import build_universe
        from qpipe.data.edgar import fetch_shares
        uv = build_universe(top_n=args.top, buffer=args.buffer, start_year=args.start_year)
        pool = uv["pool"]
        print(f"\nderived pool ({len(pool)}): {', '.join(pool)}")
        qt = uv["quarterly_top"]
        for k in sorted(qt)[-4:]:
            print(f"  {k} top {args.top}: {', '.join(qt[k])}")
        print("\nfetching PIT share series for the pool…")
        fetch_shares(pool, args.shares_out)
        cfg_path = _P(args.configs_dir) / f"{args.config}.yaml"
        c = yaml.safe_load(cfg_path.read_text())
        primary = c.get("symbol") if c.get("symbol") in pool else pool[0]
        c["symbol"] = primary
        c["universe"] = [s for s in pool if s != primary]
        c.setdefault("fixed", {})["shares_file"] = args.shares_out
        c["fixed"].setdefault("universe_top_n", args.top)
        cfg_path.write_text(yaml.safe_dump(c, sort_keys=False))
        print(f"updated {cfg_path} (primary {primary}, universe {len(c['universe'])} names)")
        print("note: ingest prices for new symbols (auto-fetches on first backtest)")

    elif args.cmd == "backtest":
        from qpipe.config import RunConfig
        from qpipe.backtest.runner import run_backtest
        cfg = RunConfig.from_yaml(args.config)
        m = run_backtest(
            cfg, args.catalog, json.loads(args.params),
            pd.Timestamp(args.start, tz="UTC") if args.start else None,
            pd.Timestamp(args.end, tz="UTC") if args.end else None,
            balance=args.balance,
        )
        print(json.dumps(m, indent=2, default=str))

    elif args.cmd == "serve":
        from qpipe.web.app import serve
        print(f"qpipe UI -> http://{args.host}:{args.port}")
        serve(args.catalog, args.configs, args.runs, args.host, args.port, args.reload)

    elif args.cmd == "mcp":
        from qpipe.mcp_server import serve_mcp
        print(f"qpipe MCP -> http://{args.host}:{args.port}/mcp")
        serve_mcp(args.catalog, args.configs, args.runs, args.host, args.port)

    elif args.cmd == "optimize":
        from qpipe.config import RunConfig
        from qpipe.optimize.engine import optimize
        cfg = RunConfig.from_yaml(args.config)
        optimize(
            cfg, args.catalog,
            segment_months=args.segment_months, n_windows=args.windows, n_trials=args.trials,
            n_jobs=args.jobs, gap_days=args.gap_days, holdout_months=args.holdout_months,
            warmup_days=args.warmup_days, balance=args.balance, seed=args.seed, out_dir=args.out,
        )


if __name__ == "__main__":
    main()
