"""Classic rolling walk-forward optimization with full grid search.

Segments roll forward through history: a fixed-length IS window is grid-searched
over the declared parameter space (ranges + steps), the best params are applied
to the immediately-following OOS window, then everything slides forward by
`step_months` (default = OOS length). The per-segment OOS results are stitched
into one continuous out-of-sample equity curve — the closest backtesting gets to
"how would this have traded live, re-optimized on schedule".
"""

from __future__ import annotations

import itertools
import json
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from qpipe.config import RunConfig
from qpipe.backtest.runner import data_range, objective_value, run_backtest

import os


def _max_combos() -> int:
    """Grid-size guardrail: Settings tab > env QPIPE_MAX_COMBOS > 20k default."""
    try:
        from qpipe.data.autofetch import load_settings
        v = load_settings().get("max_combos")
        if v:
            return int(v)
    except Exception:  # noqa: BLE001
        pass
    return int(os.environ.get("QPIPE_MAX_COMBOS", 20_000))


def grid_combos(run_cfg: RunConfig) -> list[dict]:
    """Expand space (ranges + steps) into every combination, constraints applied."""
    axes: list[tuple[str, list]] = []
    for p in run_cfg.space:
        if p.type == "int":
            step = int(p.step or 1)
            vals = list(range(int(p.low), int(p.high) + 1, step))
        elif p.type == "float":
            if not p.step:
                raise ValueError(f"param '{p.name}': float params need a step for grid search")
            n = int(round((p.high - p.low) / p.step)) + 1
            vals = [round(p.low + i * p.step, 10) for i in range(n) if p.low + i * p.step <= p.high + 1e-9]
        else:
            vals = list(p.choices or [])
        axes.append((p.name, vals))
    names = [n for n, _ in axes]
    combos = [dict(zip(names, vs)) for vs in itertools.product(*[v for _, v in axes])]
    combos = [c for c in combos if run_cfg.check_constraints(c)]
    cap = _max_combos()
    if len(combos) > cap:
        raise ValueError(
            f"grid has {len(combos):,} combos (cap {cap:,}); increase steps, disable "
            f"params, or raise the cap in Settings"
        )
    return combos


def _eval_combo(args: dict) -> dict:
    run_cfg = RunConfig(**args["run_cfg"])
    m = run_backtest(
        run_cfg, args["catalog_path"], args["params"],
        pd.Timestamp(args["start"], tz="UTC"), pd.Timestamp(args["end"], tz="UTC"),
        balance=args["balance"], warmup_days=args["warmup_days"],
    )
    return {"params": args["params"], "value": objective_value(m, run_cfg.objective),
            "sharpe": m.get("sharpe"), "psr": m.get("psr"), "pnl_pct": m.get("pnl_pct"),
            "max_drawdown": m.get("max_drawdown"), "positions": m.get("total_positions", 0)}


def walk_forward(
    run_cfg: RunConfig,
    catalog_path: str,
    is_months: int = 12,
    oos_months: int = 3,
    step_months: int | None = None,
    gap_days: int = 3,
    warmup_days: int = 90,
    balance: float = 100_000.0,
    n_jobs: int = 4,
    out_dir: str = "runs",
    progress_cb=None,
    metric_filter: str = "",  # e.g. "sharpe > 1 and psr > 0.6 and alpha > 0"
) -> dict:
    def emit(e: dict) -> None:
        if progress_cb:
            progress_cb(e)

    t0 = time.time()
    step_months = step_months or oos_months
    combos = grid_combos(run_cfg)
    cfg_dict = {
        "strategy_path": run_cfg.strategy_path, "config_class_path": run_cfg.config_class_path,
        "symbol": run_cfg.symbol, "venue": run_cfg.venue, "bar_spec": run_cfg.bar_spec,
        "fixed": run_cfg.fixed, "space": run_cfg.space, "constraints": run_cfg.constraints,
        "objective": run_cfg.objective, "universe": run_cfg.universe,
        "account_type": run_cfg.account_type, "fees": run_cfg.fees,
    }

    d0, d1 = data_range(catalog_path, run_cfg.bar_type_str)
    segments = []
    is_start = d0 + pd.Timedelta(days=warmup_days)
    while True:
        is_end = is_start + pd.DateOffset(months=is_months)
        oos_start = is_end + pd.Timedelta(days=gap_days)
        oos_end = oos_start + pd.DateOffset(months=oos_months)
        if oos_end > d1:
            break
        segments.append((is_start, is_end, oos_start, oos_end))
        is_start = is_start + pd.DateOffset(months=step_months)
    if not segments:
        raise ValueError("data range too short for the chosen IS/OOS lengths")

    run_id = f"wf_{run_cfg.symbol}_{is_months}is{oos_months}oos_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}"
    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    emit({"type": "started", "run_id": run_id, "n_segments": len(segments),
          "n_combos": len(combos), "data_start": str(d0.date()), "data_end": str(d1.date())})

    def _passes(t: dict) -> bool:
        if not t.get("positions"):
            return False
        if not metric_filter.strip():
            return True
        scope = {k: (v if v is not None else float("nan")) for k, v in t.items() if k != "params"}
        try:
            return bool(eval(metric_filter, {"__builtins__": {}}, scope))  # noqa: S307
        except Exception:
            return False

    # SPY benchmark return per IS window (for per-trial simple alpha)
    spy_bars = None
    try:
        from qpipe.backtest.runner import load_market
        _, spy_bars = load_market(catalog_path, f"SPY.{run_cfg.venue}-{run_cfg.bar_spec}-EXTERNAL")
    except Exception:  # noqa: BLE001
        pass

    def _spy_ann(s: pd.Timestamp, e: pd.Timestamp) -> float | None:
        if not spy_bars:
            return None
        win = [b for b in spy_bars if int(s.value) <= b.ts_event <= int(e.value)]
        if len(win) < 10:
            return None
        years = max((e - s).days / 365.25, 1e-9)
        return (float(win[-1].close) / float(win[0].close)) ** (1 / years) - 1

    seg_results, stitched, all_trials = [], [], []
    for si, (is_s, is_e, oos_s, oos_e) in enumerate(segments):
        seg_spy = _spy_ann(is_s, is_e)
        seg_years = max((is_e - is_s).days / 365.25, 1e-9)
        jobs = [{"run_cfg": cfg_dict, "catalog_path": catalog_path, "params": c,
                 "start": str(is_s.date()), "end": str(is_e.date()),
                 "balance": balance, "warmup_days": warmup_days} for c in combos]
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for fut in as_completed([ex.submit(_eval_combo, j) for j in jobs]):
                r = fut.result()
                # simple annualized alpha vs SPY over the same IS window
                if r.get("pnl_pct") is not None and seg_spy is not None:
                    strat_ann = (1 + r["pnl_pct"] / 100) ** (1 / seg_years) - 1
                    r["alpha"] = strat_ann - seg_spy
                else:
                    r["alpha"] = None
                results.append(r)
                emit({"type": "trial", "segment": si + 1, **r})
        valid = [r for r in results if r["value"] > -1e5]
        if not valid:
            seg_results.append({"segment": si, "error": "no_valid_combos",
                                "is_window": [str(is_s.date()), str(is_e.date())],
                                "oos_window": [str(oos_s.date()), str(oos_e.date())]})
            continue
        passing = [r for r in valid if _passes(r)]
        filter_relaxed = bool(metric_filter.strip()) and not passing
        best = max(passing or valid, key=lambda r: r["value"])
        for r in valid:
            all_trials.append({**r["params"], "_value": r["value"], "_window": si})

        oos = run_backtest(run_cfg, catalog_path, best["params"], oos_s, oos_e,
                           balance=balance, warmup_days=warmup_days, with_curve=True)
        if not oos.get("total_positions"):
            emit({"type": "log", "msg": f"segment {si+1}: OOS made ZERO trades — warmup_days "
                  f"({warmup_days}) is likely smaller than the strategy's longest lookback in bars"})
        # stitched curve must contain ONLY the OOS period — warm-up bars are flat
        # equity that would dilute every stitched statistic
        curve = [p for p in (oos.pop("equity_curve", None) or []) if p[0] >= oos_s.value // 1_000_000]
        # stitch: rescale this segment's OOS curve to continue from the previous end
        if curve:
            scale = (stitched[-1][1] / curve[0][1]) if stitched else (balance / curve[0][1])
            stitched.extend([[ts, round(v * scale, 2)] for ts, v in curve])
        seg_results.append({
            "segment": si,
            "is_window": [str(is_s.date()), str(is_e.date())],
            "oos_window": [str(oos_s.date()), str(oos_e.date())],
            "filter_relaxed": filter_relaxed,
            "n_passing": len(passing),
            "best_params": best["params"], "is_objective": best["value"],
            "oos_objective": objective_value(oos, run_cfg.objective, no_trade_value=0.0),
            "oos_metrics": {k: v for k, v in oos.items() if k != "trades"},
        })
        emit({"type": "segment_done", "i": si + 1, "n": len(segments),
              "filter_relaxed": filter_relaxed, "n_passing": len(passing),
              "is_window": seg_results[-1]["is_window"], "oos_window": seg_results[-1]["oos_window"],
              "best_params": best["params"], "is_objective": best["value"],
              "oos_objective": seg_results[-1]["oos_objective"]})

    # stats on the stitched OOS curve
    from qpipe.backtest.runner import _risk_stats
    eq = pd.Series([v for _, v in stitched],
                   index=[pd.Timestamp(t, unit="ms", tz="UTC") for t, _ in stitched]) if stitched else None
    risk = _risk_stats(eq, balance)
    r = eq.pct_change().dropna() if eq is not None else None
    sharpe = float(r.mean() / r.std() * (252 ** 0.5)) if r is not None and r.std() > 0 else None

    result = {
        "type": "walkforward", "run_id": run_id, "config": {**cfg_dict, "space": [vars(p) for p in run_cfg.space]},
        "settings": {"metric_filter": metric_filter,
                     "is_months": is_months, "oos_months": oos_months, "step_months": step_months,
                     "gap_days": gap_days, "warmup_days": warmup_days, "balance": balance,
                     "n_combos": len(combos), "n_segments": len(segments),
                     "data_start": str(d0.date()), "data_end": str(d1.date())},
        "segments": seg_results,
        "trials": all_trials,
        "stitched_curve": stitched,
        "summary": {
            "stitched_sharpe": sharpe,
            "stitched_pnl_pct": round((stitched[-1][1] / balance - 1) * 100, 2) if stitched else None,
            "stitched_max_drawdown": risk["max_drawdown"],
            "stitched_psr": risk["psr"],
            "median_is_objective": (sorted(s["is_objective"] for s in seg_results if "is_objective" in s) or [None])[len([s for s in seg_results if "is_objective" in s]) // 2] if any("is_objective" in s for s in seg_results) else None,
            "median_oos_objective": (sorted(s["oos_objective"] for s in seg_results if "oos_objective" in s) or [None])[len([s for s in seg_results if "oos_objective" in s]) // 2] if any("oos_objective" in s for s in seg_results) else None,
            "elapsed_secs": round(time.time() - t0, 1),
        },
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    emit({"type": "done", "run_id": run_id, "summary": result["summary"]})
    return result
