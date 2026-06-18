"""On-demand performance statistics computed from a stored equity curve.

All stats derive from daily strategy returns (and aligned SPY benchmark returns
where needed), so anything here can be computed for any past backtest.
"""

from __future__ import annotations

import math

import pandas as pd

ANN = 252

STATS: dict[str, str] = {
    "cagr": "Compound annual growth rate",
    "ann_volatility": "Annualized volatility of daily returns",
    "calmar": "CAGR / |max drawdown|",
    "beta_spy": "Beta vs SPY (daily returns regression)",
    "jensen_alpha_spy": "Jensen's alpha vs SPY, annualized (beta-adjusted)",
    "excess_alpha_spy": "Annualized return minus SPY's (not beta-adjusted)",
    "correlation_spy": "Correlation of daily returns with SPY",
    "up_capture_spy": "Avg return on SPY up-days / SPY's avg up-day return",
    "down_capture_spy": "Avg return on SPY down-days / SPY's avg down-day return",
    "skew": "Skewness of daily returns",
    "kurtosis": "Excess kurtosis of daily returns",
    "var_95": "1-day 95% Value at Risk (historical)",
    "cvar_95": "1-day 95% expected shortfall",
    "best_day": "Best single-day return",
    "worst_day": "Worst single-day return",
    "pct_days_positive": "Share of days with positive return",
    "longest_drawdown_days": "Longest peak-to-recovery stretch (calendar days)",
}


def _series(points: list) -> pd.Series:
    idx = pd.to_datetime([p[0] for p in points], unit="ms", utc=True)
    return pd.Series([p[1] for p in points], index=idx)


def compute_stat(name: str, points: list, bench_points: list | None = None) -> dict:
    """points / bench_points: [[ts_ms, value], ...]. Returns {name, value, fmt}."""
    eq = _series(points)
    r = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1

    b = None
    if bench_points:
        bench = _series(bench_points)
        df = pd.concat([r.rename("r"), bench.pct_change().dropna().rename("b")], axis=1).dropna()
        if len(df) > 10:
            b = df

    def need_bench():
        if b is None:
            raise ValueError("SPY benchmark data not available for this period")

    if name == "cagr":
        v, f = cagr, "pct"
    elif name == "ann_volatility":
        v, f = float(r.std() * math.sqrt(ANN)), "pct"
    elif name == "calmar":
        peak = eq.cummax()
        mdd = float(((eq - peak) / peak).min())
        v, f = (cagr / abs(mdd) if mdd else None), "num"
    elif name == "beta_spy":
        need_bench()
        v, f = float(b["r"].cov(b["b"]) / b["b"].var()), "num"
    elif name == "jensen_alpha_spy":
        need_bench()
        beta = float(b["r"].cov(b["b"]) / b["b"].var())
        v, f = float((b["r"].mean() - beta * b["b"].mean()) * ANN), "pct"
    elif name == "excess_alpha_spy":
        need_bench()
        bench_eq = _series(bench_points)
        bench_cagr = (bench_eq.iloc[-1] / bench_eq.iloc[0]) ** (1 / years) - 1
        v, f = cagr - bench_cagr, "pct"
    elif name == "correlation_spy":
        need_bench()
        v, f = float(b["r"].corr(b["b"])), "num"
    elif name == "up_capture_spy":
        need_bench()
        up = b[b["b"] > 0]
        v, f = (float(up["r"].mean() / up["b"].mean()) if len(up) else None), "num"
    elif name == "down_capture_spy":
        need_bench()
        dn = b[b["b"] < 0]
        v, f = (float(dn["r"].mean() / dn["b"].mean()) if len(dn) else None), "num"
    elif name == "skew":
        v, f = float(r.skew()), "num"
    elif name == "kurtosis":
        v, f = float(r.kurt()), "num"
    elif name == "var_95":
        v, f = float(r.quantile(0.05)), "pct"
    elif name == "cvar_95":
        v, f = float(r[r <= r.quantile(0.05)].mean()), "pct"
    elif name == "best_day":
        v, f = float(r.max()), "pct"
    elif name == "worst_day":
        v, f = float(r.min()), "pct"
    elif name == "pct_days_positive":
        v, f = float((r > 0).mean()), "pct"
    elif name == "longest_drawdown_days":
        peak = eq.cummax()
        at_peak = eq >= peak
        longest, cur_start = 0, None
        for ts, ok in at_peak.items():
            if ok:
                cur_start = None
            else:
                cur_start = cur_start or ts
                longest = max(longest, (ts - cur_start).days)
        v, f = longest, "int"
    else:
        raise ValueError(f"unknown stat: {name}")
    return {"name": name, "value": v, "fmt": f, "description": STATS.get(name, "")}
