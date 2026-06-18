"""Randomized IS/OOS optimization engine.

For each randomly sampled window pair:
  1. Optuna optimizes the strategy's params on the IS window.
  2. The best IS params are evaluated once on the unseen OOS window.

Then cross-validation: every distinct winning param set is evaluated on ALL OOS
windows; the recommended set maximizes median OOS objective (tiebreak: lower
dispersion). This selects for robustness, not a single lucky window.
"""

from __future__ import annotations

import json
import statistics
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

from qpipe.config import RunConfig
from qpipe.backtest.runner import data_range, objective_value, run_backtest
from qpipe.optimize.windows import WindowPair, sample_windows

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------- worker fns

def _optimize_one_window(args: dict) -> dict:
    """Worker: Optuna study on one IS window, then OOS evaluation."""
    run_cfg = RunConfig(**args["run_cfg"])
    cat, w = args["catalog_path"], args["window"]
    is_start, is_end = pd.Timestamp(w["is_start"], tz="UTC"), pd.Timestamp(w["is_end"], tz="UTC")
    oos_start, oos_end = pd.Timestamp(w["oos_start"], tz="UTC"), pd.Timestamp(w["oos_end"], tz="UTC")
    space = run_cfg.space

    def objective(trial: optuna.Trial) -> float:
        params = {p.name: p.suggest(trial) for p in space}
        if not run_cfg.check_constraints(params):
            raise optuna.TrialPruned()
        m = run_backtest(run_cfg, cat, params, is_start, is_end,
                         balance=args["balance"], warmup_days=args["warmup_days"])
        return objective_value(m, run_cfg.objective)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args["seed"]),
    )
    study.optimize(objective, n_trials=args["n_trials"])

    best_params = study.best_params
    oos = run_backtest(run_cfg, cat, best_params, oos_start, oos_end,
                       balance=args["balance"], warmup_days=args["warmup_days"])
    trials = [
        {"params": t.params, "value": t.value}
        for t in study.trials
        if t.value is not None and t.value > -1e5
    ]
    return {
        "trials": trials,
        "window": w,
        "best_params": best_params,
        "is_objective": study.best_value,
        "oos_metrics": oos,
        "oos_objective": objective_value(oos, run_cfg.objective, no_trade_value=0.0),
        "n_trials": len(study.trials),
    }


def _evaluate_candidate(args: dict) -> dict:
    """Worker: evaluate one param set across all OOS windows."""
    run_cfg = RunConfig(**args["run_cfg"])
    cat, params = args["catalog_path"], args["params"]
    values = []
    for w in args["windows"]:
        m = run_backtest(
            run_cfg, cat, params,
            pd.Timestamp(w["oos_start"], tz="UTC"), pd.Timestamp(w["oos_end"], tz="UTC"),
            balance=args["balance"], warmup_days=args["warmup_days"],
        )
        values.append(objective_value(m, run_cfg.objective, no_trade_value=0.0))
    valid = [v for v in values if v > -1e5]
    return {
        "params": params,
        "oos_values": values,
        "median": statistics.median(valid) if valid else -1e6,
        "iqr": (statistics.quantiles(valid, n=4)[2] - statistics.quantiles(valid, n=4)[0]) if len(valid) >= 4 else None,
        "pct_positive": sum(v > 0 for v in valid) / len(valid) if valid else 0.0,
    }


# ---------------------------------------------------------------- main entry

def _cfg_dict(run_cfg: RunConfig) -> dict:
    return {
        "strategy_path": run_cfg.strategy_path,
        "config_class_path": run_cfg.config_class_path,
        "symbol": run_cfg.symbol,
        "venue": run_cfg.venue,
        "bar_spec": run_cfg.bar_spec,
        "fixed": run_cfg.fixed,
        "space": run_cfg.space,
        "constraints": run_cfg.constraints,
        "objective": run_cfg.objective,
    }


def optimize(
    run_cfg: RunConfig,
    catalog_path: str,
    segment_months: int = 6,
    n_windows: int = 20,
    n_trials: int = 50,
    n_jobs: int = 4,
    gap_days: int = 5,
    holdout_months: int = 0,
    warmup_days: int = 90,
    balance: float = 100_000.0,
    seed: int = 42,
    out_dir: str = "runs",
    progress_cb=None,
) -> dict:
    def emit(event: dict) -> None:
        if progress_cb:
            progress_cb(event)
    assert segment_months in (3, 6, 9), "segment_months must be 3, 6 or 9"
    t0 = time.time()
    run_id = f"{run_cfg.symbol}_{segment_months}m_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    d0, d1 = data_range(catalog_path, run_cfg.bar_type_str)
    # Windows start no earlier than data_start + warmup, so warm-up is never truncated
    w0 = d0 + pd.Timedelta(days=warmup_days)
    windows = sample_windows(w0, d1, segment_months, n_windows, gap_days, None, holdout_months, seed)
    print(f"Run {run_id}: data {d0.date()}..{d1.date()}, {n_windows} windows x {n_trials} trials, {n_jobs} workers")
    emit({"type": "started", "run_id": run_id, "n_windows": n_windows, "n_trials": n_trials,
          "data_start": str(d0.date()), "data_end": str(d1.date())})

    cfg = _cfg_dict(run_cfg)
    base = {"run_cfg": cfg, "catalog_path": catalog_path, "n_trials": n_trials,
            "balance": balance, "warmup_days": warmup_days}

    # Phase 1: per-window IS optimization + OOS validation
    window_results: list[dict] = []
    jobs = [{**base, "window": w.to_dict(), "seed": seed + i} for i, w in enumerate(windows)]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(_optimize_one_window, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            window_results.append(r)
            print(f"  [{i}/{len(jobs)}] IS {r['window']['is_start']}: "
                  f"IS={r['is_objective']:.3f} OOS={r['oos_objective']:.3f} params={r['best_params']}")
            emit({"type": "window_done", "i": i, "n": len(jobs), "window": r["window"],
                  "best_params": r["best_params"], "is_objective": r["is_objective"],
                  "oos_objective": r["oos_objective"]})

    # Phase 2: cross-validate distinct winners across all OOS windows
    emit({"type": "phase", "name": "cross_validation"})
    distinct = list({json.dumps(r["best_params"], sort_keys=True) for r in window_results})
    cand_jobs = [{**base, "params": json.loads(p), "windows": [w.to_dict() for w in windows]} for p in distinct]
    candidates: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for fut in as_completed([ex.submit(_evaluate_candidate, j) for j in cand_jobs]):
            candidates.append(fut.result())
    candidates.sort(key=lambda c: (-c["median"], c["iqr"] if c["iqr"] is not None else 1e9))
    best = candidates[0]

    # flat list of every IS trial across all windows (for the heat map)
    all_trials = []
    for wi, r in enumerate(window_results):
        for t in r.pop("trials", []):
            all_trials.append({**t["params"], "_value": t["value"], "_window": wi})

    is_vals = [r["is_objective"] for r in window_results if r["is_objective"] > -1e5]
    oos_vals = [r["oos_objective"] for r in window_results if r["oos_objective"] > -1e5]
    result = {
        "run_id": run_id,
        "config": {**cfg, "space": [vars(p) for p in run_cfg.space]},
        "settings": {
            "segment_months": segment_months, "n_windows": n_windows, "n_trials": n_trials,
            "gap_days": gap_days, "holdout_months": holdout_months, "warmup_days": warmup_days,
            "seed": seed, "balance": balance,
            "data_start": str(d0.date()), "data_end": str(d1.date()),
        },
        "window_results": window_results,
        "trials": all_trials,
        "candidates": candidates,
        "recommended": best,
        "summary": {
            "median_is_objective": statistics.median(is_vals) if is_vals else None,
            "median_oos_objective": statistics.median(oos_vals) if oos_vals else None,
            "recommended_params": best["params"],
            "recommended_median_oos": best["median"],
            "recommended_pct_positive": best["pct_positive"],
            "elapsed_secs": round(time.time() - t0, 1),
        },
    }

    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    model = {
        "strategy": run_cfg.strategy_path,
        "config_class": run_cfg.config_class_path,
        "symbol": run_cfg.symbol,
        "venue": run_cfg.venue,
        "bar_spec": run_cfg.bar_spec,
        "parameters": {**run_cfg.fixed, **best["params"]},
        "provenance": {
            "run_id": run_id, "objective": run_cfg.objective,
            "median_oos": best["median"], "pct_positive_oos": best["pct_positive"],
            "optimized_on": f"{d0.date()}..{d1.date()}",
        },
    }
    (out / "optimized_model.json").write_text(json.dumps(model, indent=2))

    from qpipe.optimize.report import write_report
    write_report(result, out / "report.md")
    print(f"Done in {result['summary']['elapsed_secs']}s -> {out}")
    emit({"type": "done", "run_id": run_id, "summary": result["summary"], "out_dir": str(out)})
    return result
