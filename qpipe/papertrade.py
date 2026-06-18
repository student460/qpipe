"""Paper trading on synthetic market continuations.

Calibrates a correlated geometric Brownian motion on each symbol's real history
(per-symbol drift + volatility, cross-asset correlation via Cholesky, volumes
bootstrapped from the historical distribution), generates unseen future bars,
and runs the strategy on them through the normal backtest engine.

Modes:
- live:   one path, paced at `speed` bars/second so you can watch it trade.
- monte:  N independently-seeded paths at full speed -> a DISTRIBUTION of
          outcomes (sharpe, max drawdown, pnl), which is the honest way to ask
          "what risk am I actually tolerating?"

The synthetic world preserves vol/correlation structure but has no real alpha in
it beyond drift — momentum strategies will look weaker here than on history.
That's informative: performance on GBM continuations ~ what you get if the
historical patterns you fit do NOT persist.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qpipe.config import RunConfig
from qpipe.backtest.runner import load_market, run_backtest
from qpipe.data.catalog import write_bars_df


def _hist_frames(catalog_path: str, run_cfg: RunConfig, window_years: float) -> dict[str, pd.DataFrame]:
    """Last `window_years` of real OHLCV per symbol (for calibration + warm-up)."""
    out = {}
    for bts in run_cfg.all_bar_type_strs:
        try:
            _, bars = load_market(catalog_path, bts)
        except ValueError:
            continue
        sym = bts.split("-")[0].rsplit(".", 1)[0]
        idx = pd.DatetimeIndex([pd.Timestamp(b.ts_event, unit="ns", tz="UTC") for b in bars])
        df = pd.DataFrame({"open": [float(b.open) for b in bars], "high": [float(b.high) for b in bars],
                           "low": [float(b.low) for b in bars], "close": [float(b.close) for b in bars],
                           "volume": [float(b.volume) for b in bars]}, index=idx)
        cutoff = df.index[-1] - pd.Timedelta(days=int(window_years * 365.25))
        out[sym] = df[df.index >= cutoff]
    return out


def generate_continuation(hist: dict[str, pd.DataFrame], horizon_years: float, seed: int) -> dict[str, pd.DataFrame]:
    """Correlated GBM continuation per symbol, volumes bootstrapped from history."""
    rng = np.random.default_rng(seed)
    syms = sorted(hist.keys())
    rets = pd.DataFrame({s: np.log(hist[s].close).diff() for s in syms}).dropna()
    mu = rets.mean().values
    cov = rets.cov().values
    # regularize then Cholesky (correlated daily shocks)
    L = np.linalg.cholesky(cov + np.eye(len(syms)) * 1e-12)
    n = int(252 * horizon_years)
    start = max(hist[s].index[-1] for s in syms) + pd.Timedelta(days=1)
    dates = pd.bdate_range(start, periods=n, tz="UTC")

    z = rng.standard_normal((n, len(syms)))
    shocks = z @ L.T + mu  # log returns
    out = {}
    for j, s in enumerate(syms):
        last = float(hist[s].close.iloc[-1])
        closes = last * np.exp(np.cumsum(shocks[:, j]))
        opens = np.r_[last, closes[:-1]]
        spread = np.abs(rng.normal(0, 0.004, n))
        highs = np.maximum(opens, closes) * (1 + spread)
        lows = np.minimum(opens, closes) * (1 - spread)
        vols = rng.choice(hist[s].volume.values, size=n)  # bootstrap real volume dist
        out[s] = pd.DataFrame({"open": opens.round(4), "high": highs.round(4),
                               "low": lows.round(4), "close": closes.round(4),
                               "volume": vols}, index=dates)
    return out


def run_paper(
    run_cfg: RunConfig,
    catalog_path: str,
    work_dir: str,
    horizon_years: float = 5.0,
    n_paths: int = 1,
    speed: float = 0,        # bars/second for path 1 (0 = unpaced)
    hist_window_years: float = 4.0,
    balance: float = 100_000.0,
    warmup_days: int = 380,
    seed: int = 42,
    params: dict | None = None,
    progress_cb=None,
    point_cb=None,           # live equity stream for path 1
) -> dict:
    def emit(e: dict) -> None:
        if progress_cb:
            progress_cb(e)

    hist = _hist_frames(catalog_path, run_cfg, hist_window_years + warmup_days / 365.25)
    if run_cfg.symbol not in hist:
        raise ValueError(f"no history for primary symbol {run_cfg.symbol}")
    emit({"type": "started", "n_paths": n_paths, "horizon_years": horizon_years,
          "symbols": sorted(hist.keys())})

    # SPY benchmark calibration: prefer SPY inside the synthetic world (shares the
    # correlated shocks); else an independent GBM from SPY's real history.
    spy_hist = None
    if "SPY" not in hist:
        try:
            _, spy_bars = load_market(catalog_path, f"SPY.{run_cfg.venue}-{run_cfg.bar_spec}-EXTERNAL")
            closes = pd.Series([float(b.close) for b in spy_bars[-int(252 * hist_window_years):]])
            lr = np.log(closes).diff().dropna()
            spy_hist = {"mu": float(lr.mean()), "sigma": float(lr.std()), "last": float(closes.iloc[-1])}
        except Exception:  # noqa: BLE001
            pass

    def _bench_curve(synth: dict, path_seed: int, dates_hint=None):
        """Synthetic SPY equity (scaled to balance) for one generated world."""
        if "SPY" in synth:
            c = synth["SPY"].close
            return [[int(ts.value // 1_000_000), round(float(balance * v / c.iloc[0]), 2)]
                    for ts, v in c.items()]
        if spy_hist is None:
            return None
        rng_b = np.random.default_rng(path_seed + 990_001)
        dates = dates_hint
        n = len(dates)
        lr = spy_hist["mu"] + spy_hist["sigma"] * rng_b.standard_normal(n)
        closes = spy_hist["last"] * np.exp(np.cumsum(lr))
        return [[int(ts.value // 1_000_000), round(float(balance * v / closes[0]), 2)]
                for ts, v in zip(dates, closes)]

    results = []
    benchmark = None
    for p in range(n_paths):
        synth = generate_continuation(hist, horizon_years, seed + p)
        sim_start = min(df.index[0] for df in synth.values())
        bench = _bench_curve(synth, seed + p, dates_hint=next(iter(synth.values())).index)
        if p == 0 and bench:
            benchmark = bench
            emit({"type": "benchmark", "points": bench})

        path_cat = Path(work_dir) / f"path_{p}"
        if path_cat.exists():
            shutil.rmtree(path_cat)
        for s, sdf in synth.items():
            full = pd.concat([hist[s], sdf])
            write_bars_df(str(path_cat), s, run_cfg.venue,
                          run_cfg.bar_spec.replace("-EXTERNAL", ""), full)
        load_market.cache_clear()

        pace = None
        if p == 0 and speed and speed > 0:
            delay = 1.0 / speed
            last_emit = [0.0]

            def pace(ts_ns: int, equity: float) -> None:
                if point_cb:
                    point_cb(ts_ns, equity)
                now = time.time()
                wait = delay - (now - last_emit[0])
                if wait > 0:
                    time.sleep(wait)
                last_emit[0] = time.time()
        elif p == 0:
            pace = point_cb

        m = run_backtest(run_cfg, str(path_cat), params or {}, sim_start, None,
                         balance=balance, warmup_days=warmup_days,
                         with_trades=(p == 0), point_cb=pace)
        bench_pnl = float((bench[-1][1] / bench[0][1]) - 1) * 100 if bench else None
        res = {"path": p, "sharpe": m.get("sharpe"), "psr": m.get("psr"),
               "pnl_pct": m.get("pnl_pct"), "max_drawdown": m.get("max_drawdown"),
               "total_positions": m.get("total_positions"), "error": m.get("error"),
               "spy_pnl_pct": round(bench_pnl, 2) if bench_pnl is not None else None,
               "excess_pnl_pct": (round(m["pnl_pct"] - bench_pnl, 2)
                                  if bench_pnl is not None and m.get("pnl_pct") is not None else None)}
        if p == 0:
            res["trades"] = m.get("trades", [])[-400:]
        results.append(res)
        emit({"type": "path_done", "i": p + 1, "n": n_paths,
              **{k: v for k, v in res.items() if k != "trades"}})

    vals = lambda k: sorted(r[k] for r in results if r.get(k) is not None)  # noqa: E731
    def pct(arr, q):
        return arr[min(len(arr) - 1, int(q * len(arr)))] if arr else None
    sh, dd, pnl = vals("sharpe"), vals("max_drawdown"), vals("pnl_pct")
    exc = vals("excess_pnl_pct")
    summary = {
        "n_paths": n_paths,
        "sharpe_p5": pct(sh, 0.05), "sharpe_median": pct(sh, 0.5), "sharpe_p95": pct(sh, 0.95),
        "maxdd_p5": pct(dd, 0.05), "maxdd_median": pct(dd, 0.5),
        "pnl_p5": pct(pnl, 0.05), "pnl_median": pct(pnl, 0.5), "pnl_p95": pct(pnl, 0.95),
        "pct_profitable": (sum(1 for v in pnl if v > 0) / len(pnl)) if pnl else None,
        "excess_median": pct(exc, 0.5),
        "pct_beat_spy": (sum(1 for v in exc if v > 0) / len(exc)) if exc else None,
    }
    emit({"type": "done", "summary": summary})
    return {"results": results, "summary": summary, "benchmark": benchmark}
