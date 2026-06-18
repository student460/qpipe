"""qpipe web UI — FastAPI backend.

Run with:  qpipe serve --catalog catalog --port 8000
Security note: this app edits and executes strategy code by design. Only expose
it on localhost or a private network (Tailscale); add auth before wider sharing.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time as _time
import urllib.request
import uuid
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import os

from qpipe.web.logbuf import LOG, log, log_exc

app = FastAPI(title="qpipe")


@app.get("/api/logs")
def get_logs(since: int = 0):
    items = list(LOG)
    return {"items": items[since:], "next": len(items)}


# ------------------------------------------------------------------- version
# Lightweight update check: compare the locally checked-out git commit to the
# latest commit on GitHub. Result cached 30 min so page loads don't hammer the
# GitHub API (unauthenticated limit is 60 req/hr). All failures degrade to
# "up to date" so the UI never blocks on network issues.

GITHUB_REPO = os.environ.get("QPIPE_GITHUB_REPO", "student640/qpipe")
GITHUB_BRANCH = os.environ.get("QPIPE_GITHUB_BRANCH", "main")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERSION_CACHE: dict = {"t": 0.0, "data": None}


def _local_sha() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except Exception:  # noqa: BLE001
        return None


def _remote_sha() -> str | None:
    url = f"https://api.github.com/repos/{GITHUB_REPO}/commits/{GITHUB_BRANCH}"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "qpipe-update-check",
            "Accept": "application/vnd.github.sha"})  # returns the bare SHA
        with urllib.request.urlopen(req, timeout=6) as resp:
            return resp.read().decode().strip() or None
    except Exception:  # noqa: BLE001
        return None


@app.get("/api/version")
def get_version():
    now = _time.time()
    cached = _VERSION_CACHE["data"]
    if cached and now - _VERSION_CACHE["t"] < 1800:
        return cached
    local, remote = _local_sha(), _remote_sha()
    data = {
        "local": local,
        "remote": remote,
        "behind": bool(local and remote and local != remote),
        "repo": GITHUB_REPO,
        "branch": GITHUB_BRANCH,
    }
    _VERSION_CACHE.update(t=now, data=data)
    return data

# Configured via serve() / env vars (env survives uvicorn --reload worker restarts)
STATE = {
    "catalog": os.environ.get("QPIPE_CATALOG", "catalog"),
    "configs_dir": Path(os.environ.get("QPIPE_CONFIGS", "configs")),
    "strategies_dir": Path(__file__).resolve().parent.parent / "strategies",
    "runs_dir": Path(os.environ.get("QPIPE_RUNS", "runs")),
}
JOBS: dict[str, dict] = {}  # job_id -> {status, events, result, error}


# ------------------------------------------------------------------- helpers

def _config_path(name: str) -> Path:
    p = (STATE["configs_dir"] / name).with_suffix(".yaml")
    if not p.resolve().is_relative_to(STATE["configs_dir"].resolve()):
        raise HTTPException(400, "bad name")
    return p


def _strategy_path(name: str) -> Path:
    p = (STATE["strategies_dir"] / name).with_suffix(".py")
    if not p.resolve().is_relative_to(STATE["strategies_dir"].resolve()):
        raise HTTPException(400, "bad name")
    return p


# -------------------------------------------------------------------- static

@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# --------------------------------------------------------------------- data

@app.get("/api/data")
def list_data():
    """Catalog inventory: every bar type with its date range and bar count."""
    from qpipe.backtest.runner import load_market

    def _symbol(bts: str) -> str:
        # "BRK.B.XNAS-1-DAY-LAST-EXTERNAL" -> instrument "BRK.B.XNAS" -> symbol "BRK.B"
        instrument = bts.split("-")[0]
        return instrument.rsplit(".", 1)[0]

    bar_root = Path(STATE["catalog"]) / "data" / "bar"
    items = []
    if bar_root.exists():
        for d in sorted(bar_root.iterdir()):
            if not d.is_dir():
                continue
            bts = d.name
            try:
                _, bars = load_market(STATE["catalog"], bts)
                items.append({
                    "bar_type": bts,
                    "symbol": _symbol(bts),
                    "bars": len(bars),
                    "start": str(pd.Timestamp(bars[0].ts_event, unit="ns", tz="UTC").date()),
                    "end": str(pd.Timestamp(bars[-1].ts_event, unit="ns", tz="UTC").date()),
                })
            except Exception as e:  # noqa: BLE001
                items.append({"bar_type": bts, "symbol": _symbol(bts), "error": str(e)})
    return {"catalog": STATE["catalog"], "items": items}


class SettingsBody(BaseModel):
    alpaca_key: str | None = None
    alpaca_secret: str | None = None
    anthropic_key: str | None = None
    openrouter_key: str | None = None
    local_base_url: str | None = None
    local_key: str | None = None
    llm_provider: str | None = None  # anthropic | openrouter | local
    llm_model: str | None = None
    auto_fetch: bool | None = None
    autofetch_start: str | None = None
    edgar_contact: str | None = None
    max_combos: int | None = None
    default_jobs: int | None = None
    default_warmup: int | None = None


@app.get("/api/settings")
def get_settings():
    import os
    from qpipe.data.autofetch import load_settings
    s = load_settings()
    return {"alpaca_key_set": bool(s.get("alpaca_key")),
            "anthropic_key_set": bool(s.get("anthropic_key")),
            "openrouter_key_set": bool(s.get("openrouter_key")),
            "llm_provider": s.get("llm_provider", "anthropic"),
            "llm_model": s.get("llm_model", ""),
            "local_base_url": s.get("local_base_url", "http://localhost:11434/v1"),
            "auto_fetch": s.get("auto_fetch", True),
            "autofetch_start": s.get("autofetch_start", "2015-01-01"),
            "edgar_contact": s.get("edgar_contact", ""),
            "max_combos": s.get("max_combos", int(os.environ.get("QPIPE_MAX_COMBOS", 20000))),
            "default_jobs": s.get("default_jobs", 8),
            "default_warmup": s.get("default_warmup", 380)}


@app.put("/api/settings")
def put_settings(body: SettingsBody):
    from qpipe.data.autofetch import save_settings
    save_settings({k: v for k, v in body.model_dump().items() if v is not None})
    return {"ok": True}


class FetchBody(BaseModel):
    config: str
    start: str = "2015-01-01"
    force: bool = False  # purge + refetch everything from `start` (yfinance)


@app.post("/api/data/fetch")
def fetch_data(body: FetchBody):
    """Fetch missing symbols, or with force=True refetch all from an earlier start."""
    from qpipe.config import RunConfig
    from qpipe.data.autofetch import ensure_data

    cfg = RunConfig.from_yaml(_config_path(body.config))
    return ensure_data(STATE["catalog"], cfg, start=body.start,
                       log=lambda m: log("info", "autofetch", m), force=body.force)


# ------------------------------------------------------------------- configs

@app.get("/api/configs")
def list_configs():
    return sorted(p.stem for p in STATE["configs_dir"].glob("*.yaml"))


@app.get("/api/configs/{name}")
def get_config(name: str):
    p = _config_path(name)
    if not p.exists():
        raise HTTPException(404)
    return {"name": name, "yaml": p.read_text(), "parsed": yaml.safe_load(p.read_text())}


class ConfigBody(BaseModel):
    yaml: str


@app.put("/api/configs/{name}")
def put_config(name: str, body: ConfigBody):
    yaml.safe_load(body.yaml)  # validate
    STATE["configs_dir"].mkdir(exist_ok=True)
    _config_path(name).write_text(body.yaml)
    return {"ok": True}


# ---------------------------------------------------------------- strategies

@app.get("/api/strategies")
def list_strategies():
    return sorted(p.stem for p in STATE["strategies_dir"].glob("*.py") if p.stem != "__init__")


@app.get("/api/strategies/{name}")
def get_strategy(name: str):
    p = _strategy_path(name)
    if not p.exists():
        raise HTTPException(404)
    return {"name": name, "code": p.read_text()}


class StrategyBody(BaseModel):
    code: str


@app.put("/api/strategies/{name}")
def put_strategy(name: str, body: StrategyBody):
    compile(body.code, name, "exec")  # syntax check before saving
    _strategy_path(name).write_text(body.code)
    return {"ok": True}


# ------------------------------------------------------------------ backtest

class BacktestBody(BaseModel):
    config: str
    params: dict = {}
    start: str | None = None
    end: str | None = None
    balance: float = 100_000.0
    warmup_days: int = 0


@app.post("/api/backtest")
def api_backtest(body: BacktestBody):
    """Launch a backtest job that streams live equity points; poll /api/backtest/{job_id}."""
    from qpipe.config import RunConfig
    from qpipe.backtest.runner import run_backtest

    cfg = RunConfig.from_yaml(_config_path(body.config))
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "status": "running", "points": [], "result": None, "error": None}
    JOBS[job_id] = job

    def cb(ts_ns: int, equity: float) -> None:
        job["points"].append([ts_ns // 1_000_000, round(equity, 2)])

    def work() -> None:
        import time

        try:
            from qpipe.data.autofetch import ensure_data
            ensure_data(STATE["catalog"], cfg, log=lambda m_: log("info", "autofetch", m_))
            m = run_backtest(
                cfg, STATE["catalog"], body.params,
                pd.Timestamp(body.start, tz="UTC") if body.start else None,
                pd.Timestamp(body.end, tz="UTC") if body.end else None,
                balance=body.balance, warmup_days=body.warmup_days,
                with_trades=True, point_cb=cb,
            )
            job["result"] = m
            job["status"] = "failed" if m.get("error") else "done"
            job["error"] = m.get("error")
            if m.get("error"):
                log("error", f"backtest:{body.config}",
                    f"{m['error']} (n_bars={m.get('n_bars')}) — likely missing catalog data for the config's symbols")
            if job["status"] == "done" and job["points"]:
                # beg/end equity + simple alpha vs SPY (annualized excess return, same window)
                pts = job["points"]
                beg, end_eq = pts[0][1], pts[-1][1]
                days = max((pts[-1][0] - pts[0][0]) / 86_400_000, 1)
                years = days / 365.25
                m["beg_equity"], m["end_equity"] = beg, end_eq
                strat_ann = (end_eq / beg) ** (1 / years) - 1 if beg > 0 and years > 0 else None
                try:
                    from qpipe.backtest.runner import load_market
                    _, spy = load_market(STATE["catalog"], f"SPY.{cfg.venue}-{cfg.bar_spec}-EXTERNAL")
                    t0, t1 = pts[0][0] * 1_000_000, pts[-1][0] * 1_000_000
                    win = [b for b in spy if t0 <= b.ts_event <= t1]
                    spy_ann = (float(win[-1].close) / float(win[0].close)) ** (1 / years) - 1 if len(win) > 2 else None
                    m["alpha_spy"] = (strat_ann - spy_ann) if (strat_ann is not None and spy_ann is not None) else None
                except Exception:  # noqa: BLE001
                    m["alpha_spy"] = None
            if job["status"] == "done":  # persist to history
                STATE["runs_dir"].mkdir(parents=True, exist_ok=True)
                rec = {"id": job_id, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "config": body.config, "params": body.params,
                       "start": body.start, "end": body.end, "balance": body.balance,
                       "metrics": {k: v for k, v in m.items() if k not in ("equity_curve", "trades")},
                       "trades": m.get("trades", []),
                       "equity_curve": job["points"]}
                with (STATE["runs_dir"] / "backtests.jsonl").open("a") as f:
                    f.write(json.dumps(rec) + "\n")
        except Exception as e:  # noqa: BLE001
            job["error"], job["status"] = log_exc(f"backtest:{body.config}", e), "failed"

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/backtest/{job_id}")
def backtest_status(job_id: str, since: int = 0):
    job = JOBS.get(job_id)
    if not job or "points" not in job:
        raise HTTPException(404)
    return {"status": job["status"], "error": job["error"],
            "points": job["points"][since:], "next": len(job["points"]),
            "result": job["result"] if job["status"] in ("done", "failed") else None}


# ------------------------------------------------------------------ optimize

class OptimizeBody(BaseModel):
    config: str
    segment_months: int = 6
    windows: int = 20
    trials: int = 50
    jobs: int = 4
    gap_days: int = 5
    holdout_months: int = 0
    warmup_days: int = 90
    balance: float = 100_000.0
    seed: int = 42


@app.post("/api/optimize")
def api_optimize(body: OptimizeBody):
    from qpipe.config import RunConfig
    from qpipe.optimize.engine import optimize

    cfg = RunConfig.from_yaml(_config_path(body.config))
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "status": "running", "events": [], "result": None, "error": None,
           "settings": body.model_dump()}
    JOBS[job_id] = job

    def cb(event: dict) -> None:
        job["events"].append(event)

    def work() -> None:
        try:
            from qpipe.data.autofetch import ensure_data
            ensure_data(STATE["catalog"], cfg,
                        log=lambda m: (cb({"type": "log", "msg": m}), log("info", "autofetch", m)))
            result = optimize(
                cfg, STATE["catalog"],
                segment_months=body.segment_months, n_windows=body.windows, n_trials=body.trials,
                n_jobs=body.jobs, gap_days=body.gap_days, holdout_months=body.holdout_months,
                warmup_days=body.warmup_days, balance=body.balance, seed=body.seed,
                out_dir=str(STATE["runs_dir"]), progress_cb=cb,
            )
            # keep the response light: drop bulky per-window OOS metrics
            for r in result["window_results"]:
                r.pop("oos_metrics", None)
            job["result"] = result
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001
            job["error"] = log_exc(f"optimize:{body.config}", e)
            job["status"] = "failed"

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs():
    return [{"id": j["id"], "status": j["status"], "settings": j["settings"]} for j in JOBS.values()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, since: int = 0):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404)
    return {"id": job["id"], "status": job["status"], "error": job["error"],
            "events": job["events"][since:], "next": len(job["events"]),
            "result": job["result"] if job["status"] == "done" else None}


# ------------------------------------------------------------ backtest history

def _bt_history() -> list[dict]:
    p = STATE["runs_dir"] / "backtests.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


@app.get("/api/backtests")
def list_backtests():
    out = []
    for rec in reversed(_bt_history()):
        m = rec["metrics"]
        out.append({"id": rec["id"], "ts": rec["ts"], "config": rec["config"],
                    "params": rec["params"], "start": rec["start"], "end": rec["end"],
                    "sharpe": m.get("sharpe"), "psr": m.get("psr"),
                    "max_drawdown": m.get("max_drawdown"), "pnl_pct": m.get("pnl_pct"),
                    "beg_equity": m.get("beg_equity"), "end_equity": m.get("end_equity"),
                    "alpha_spy": m.get("alpha_spy")})
    return out


@app.get("/api/backtests/{bt_id}")
def get_backtest(bt_id: str):
    for rec in _bt_history():
        if rec["id"] == bt_id:
            return rec
    raise HTTPException(404)


# -------------------------------------------------------------- advanced stats

@app.get("/api/stats")
def list_stats():
    from qpipe.analysis import STATS
    return STATS


def _bench_points(rec: dict) -> list | None:
    """SPY closes aligned to the backtest window, as [[ts_ms, close], ...]."""
    try:
        from qpipe.backtest.runner import load_market
        cfg_yaml = _config_path(rec["config"])
        import yaml as _y
        c = _y.safe_load(cfg_yaml.read_text())
        venue, spec = c.get("venue", "XNAS"), c.get("bar_spec", "1-DAY-LAST")
        _, spy = load_market(STATE["catalog"], f"SPY.{venue}-{spec}-EXTERNAL")
        t0, t1 = rec["equity_curve"][0][0] * 1_000_000, rec["equity_curve"][-1][0] * 1_000_000
        return [[b.ts_event // 1_000_000, float(b.close)] for b in spy if t0 <= b.ts_event <= t1]
    except Exception:  # noqa: BLE001
        return None


@app.get("/api/backtests/{bt_id}/benchmark")
def backtest_benchmark(bt_id: str):
    """SPY closes over the backtest window, for charting against the equity curve."""
    rec = next((r for r in _bt_history() if r["id"] == bt_id), None)
    if not rec:
        raise HTTPException(404)
    pts = _bench_points(rec)
    if not pts:
        raise HTTPException(400, "SPY data not in catalog for this period")
    return {"points": pts, "balance": rec.get("balance", 100_000.0)}


@app.get("/api/backtests/{bt_id}/stat/{name}")
def backtest_stat(bt_id: str, name: str):
    from qpipe.analysis import compute_stat
    rec = next((r for r in _bt_history() if r["id"] == bt_id), None)
    if not rec:
        raise HTTPException(404)
    try:
        return compute_stat(name, rec["equity_curve"], _bench_points(rec))
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------- walk-forward

class WalkForwardBody(BaseModel):
    config: str
    is_months: int = 12
    oos_months: int = 3
    step_months: int | None = None
    gap_days: int = 3
    warmup_days: int = 90
    balance: float = 100_000.0
    jobs: int = 4
    space: dict | None = None  # optional override of the config's search space
    metric_filter: str = ""    # e.g. "sharpe > 1 and psr > 0.6 and alpha > 0"


@app.post("/api/walkforward/preview")
def wf_preview(body: WalkForwardBody):
    """Combo count + segment count before launching."""
    from qpipe.config import RunConfig, ParamSpec
    from qpipe.optimize.walkforward import grid_combos
    from qpipe.backtest.runner import data_range

    cfg = RunConfig.from_yaml(_config_path(body.config))
    if body.space:
        cfg.space = [ParamSpec(name=k, **v) for k, v in body.space.items()]
    try:
        n_combos = len(grid_combos(cfg))
    except ValueError as e:
        raise HTTPException(400, str(e))
    d0, d1 = data_range(STATE["catalog"], cfg.bar_type_str)
    span_months = max((d1 - d0).days / 30.44 - body.warmup_days / 30.44, 0)
    step = body.step_months or body.oos_months
    n_segments = max(int((span_months - body.is_months - body.oos_months) / step) + 1, 0)
    total = n_combos * n_segments

    # measure real per-backtest cost: time 2 sample combos on the first IS window
    est_seconds = None
    per_bt_ms = None
    if total and n_segments:
        try:
            import time as _t
            from qpipe.optimize.walkforward import grid_combos as _gc
            from qpipe.backtest.runner import run_backtest
            combos = _gc(cfg)
            samples = [combos[0], combos[len(combos) // 2]][: max(1, min(2, len(combos)))]
            is_s = d0 + pd.Timedelta(days=body.warmup_days)
            is_e = is_s + pd.DateOffset(months=body.is_months)
            # warm-up run first: catalog loading is a one-time cost per worker,
            # not a per-backtest cost — timing it would inflate the estimate ~100x
            run_backtest(cfg, STATE["catalog"], samples[0], is_s, is_e,
                         balance=body.balance, warmup_days=body.warmup_days)
            t0 = _t.perf_counter()
            for p in samples:
                run_backtest(cfg, STATE["catalog"], p, is_s, is_e,
                             balance=body.balance, warmup_days=body.warmup_days)
            per_bt = (_t.perf_counter() - t0) / len(samples)
            per_bt_ms = round(per_bt * 1000, 1)
            est_seconds = round(total * per_bt / max(body.jobs, 1) * 1.15)  # 15% pool overhead
        except Exception:  # noqa: BLE001
            pass
    # warm-up adequacy: largest bar-count-looking value among fixed params / space highs
    warmup_warning = None
    lookbacks = [v for v in cfg.fixed.values() if isinstance(v, (int, float)) and 60 <= v <= 2000]
    lookbacks += [p.high for p in cfg.space if p.type == "int" and p.high and 60 <= p.high <= 2000]
    if lookbacks:
        need_cal = int(max(lookbacks) * 365 / 252 * 1.1)
        if body.warmup_days < need_cal:
            warmup_warning = (f"warm-up {body.warmup_days}d ≈ {int(body.warmup_days*252/365)} bars, but the "
                              f"longest lookback is {int(max(lookbacks))} bars — OOS windows will make ZERO "
                              f"trades. Use warmup_days ≥ {need_cal}.")
    return {"n_combos": n_combos, "n_segments": n_segments, "total_backtests": total,
            "per_backtest_ms": per_bt_ms, "est_seconds": est_seconds,
            "warmup_warning": warmup_warning}


@app.post("/api/walkforward")
def api_walkforward(body: WalkForwardBody):
    from qpipe.config import RunConfig, ParamSpec
    from qpipe.optimize.walkforward import walk_forward

    cfg = RunConfig.from_yaml(_config_path(body.config))
    if body.space:
        cfg.space = [ParamSpec(name=k, **v) for k, v in body.space.items()]
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "status": "running", "events": [], "result": None, "error": None,
           "settings": body.model_dump()}
    JOBS[job_id] = job

    def work() -> None:
        try:
            from qpipe.data.autofetch import ensure_data
            ensure_data(STATE["catalog"], cfg, log=lambda m: log("info", "autofetch", m))
            result = walk_forward(
                cfg, STATE["catalog"], is_months=body.is_months, oos_months=body.oos_months,
                step_months=body.step_months, gap_days=body.gap_days,
                warmup_days=body.warmup_days, balance=body.balance, n_jobs=body.jobs,
                out_dir=str(STATE["runs_dir"]), progress_cb=job["events"].append,
                metric_filter=body.metric_filter,
            )
            job["result"] = result
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001
            job["error"] = log_exc(f"walkforward:{body.config}", e)
            job["status"] = "failed"

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


# ---------------------------------------------------------------- paper trading

class PaperBody(BaseModel):
    config: str
    horizon_years: float = 5.0
    paths: int = 1
    speed: float = 0  # bars/sec for the live path; 0 = full speed
    hist_window_years: float = 4.0
    balance: float = 100_000.0
    warmup_days: int = 380
    seed: int = 42
    params: dict = {}


@app.post("/api/paper")
def api_paper(body: PaperBody):
    from qpipe.config import RunConfig
    from qpipe.papertrade import run_paper

    cfg = RunConfig.from_yaml(_config_path(body.config))
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "status": "running", "events": [], "points": [],
           "result": None, "error": None, "settings": body.model_dump()}
    JOBS[job_id] = job

    def work() -> None:
        try:
            from qpipe.data.autofetch import ensure_data
            ensure_data(STATE["catalog"], cfg, log=lambda m: log("info", "autofetch", m))
            result = run_paper(
                cfg, STATE["catalog"], str(STATE["runs_dir"] / ".paper" / job_id),
                horizon_years=body.horizon_years, n_paths=body.paths, speed=body.speed,
                hist_window_years=body.hist_window_years, balance=body.balance,
                warmup_days=body.warmup_days, seed=body.seed, params=body.params,
                progress_cb=job["events"].append,
                point_cb=lambda ts, eq: job["points"].append([ts // 1_000_000, round(eq, 2)]),
            )
            job["result"] = result
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001
            job["error"] = log_exc(f"paper:{body.config}", e)
            job["status"] = "failed"

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/paper/{job_id}")
def paper_status(job_id: str, since: int = 0, esince: int = 0):
    job = JOBS.get(job_id)
    if not job or "points" not in job:
        raise HTTPException(404)
    return {"status": job["status"], "error": job["error"],
            "points": job["points"][since:], "next": len(job["points"]),
            "events": job["events"][esince:], "enext": len(job["events"]),
            "result": job["result"] if job["status"] in ("done", "failed") else None}


# ------------------------------------------------------------- agent assistant

class AgentMsg(BaseModel):
    text: str
    context: dict | None = None  # UI snapshot auto-attached by the front-end


@app.post("/api/agent/session")
def agent_new_session():
    from qpipe.web.claude_agent import start_session
    return {"session_id": start_session()}


@app.post("/api/agent/{sid}")
def agent_send(sid: str, body: AgentMsg):
    from qpipe.web.claude_agent import SESSIONS, send_message
    if sid not in SESSIONS:
        raise HTTPException(404)
    if SESSIONS[sid]["busy"]:
        raise HTTPException(409, "agent is still working")
    send_message(sid, body.text, STATE, context=body.context)
    return {"ok": True}


@app.get("/api/agent/{sid}")
def agent_poll(sid: str, since: int = 0):
    from qpipe.web.claude_agent import SESSIONS
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404)
    return {"busy": s["busy"], "display": s["display"][since:], "next": len(s["display"])}


# ------------------------------------------------------------------ past runs

@app.get("/api/runs")
def list_runs():
    runs = []
    if STATE["runs_dir"].exists():
        for d in sorted(STATE["runs_dir"].iterdir(), reverse=True):
            if (d / "result.json").exists():
                runs.append(d.name)
    return runs


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    p = STATE["runs_dir"] / run_id / "result.json"
    if not p.exists():
        raise HTTPException(404)
    return json.loads(p.read_text())


def serve(catalog: str = "catalog", configs: str = "configs", runs: str = "runs",
          host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    import uvicorn

    STATE["catalog"] = os.environ["QPIPE_CATALOG"] = str(Path(catalog).resolve())
    STATE["configs_dir"] = Path(configs).resolve()
    STATE["runs_dir"] = Path(runs).resolve()
    os.environ["QPIPE_CONFIGS"] = str(STATE["configs_dir"])
    os.environ["QPIPE_RUNS"] = str(STATE["runs_dir"])
    print(f"  catalog: {STATE['catalog']}\n  configs: {STATE['configs_dir']}\n  runs:    {STATE['runs_dir']}")
    if reload:  # dev mode: auto-restart on code changes
        uvicorn.run("qpipe.web.app:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port)
