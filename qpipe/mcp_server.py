"""MCP server: lets Claude (Desktop, Code, Cowork) read and control the pipeline.

Start:   qpipe mcp --catalog catalog --port 8765
Connect: add an MCP server with URL http://<host>:8765/mcp (streamable HTTP).
Friends on your Tailnet use http://<tailscale-name>:8765/mcp.

Tools mirror the web API: browse/edit strategies and configs, run backtests,
launch and monitor randomized IS/OOS optimizations, read past run results.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("qpipe")

STATE = {
    "catalog": "catalog",
    "configs_dir": Path("configs"),
    "strategies_dir": Path(__file__).resolve().parent / "strategies",
    "runs_dir": Path("runs"),
}
JOBS: dict[str, dict] = {}


# ---------------------------------------------------------------- strategies

@mcp.tool()
def list_strategies() -> list[str]:
    """List available strategy module names."""
    return sorted(p.stem for p in STATE["strategies_dir"].glob("*.py") if p.stem != "__init__")


@mcp.tool()
def read_strategy(name: str) -> str:
    """Read the Python source of a strategy module."""
    return (STATE["strategies_dir"] / f"{name}.py").read_text()


@mcp.tool()
def write_strategy(name: str, code: str) -> str:
    """Create or overwrite a strategy module (syntax-checked before saving)."""
    compile(code, name, "exec")
    (STATE["strategies_dir"] / f"{name}.py").write_text(code)
    return f"saved qpipe/strategies/{name}.py"


# ------------------------------------------------------------------- configs

@mcp.tool()
def list_configs() -> list[str]:
    """List run-config names (strategy + symbols + parameter search space)."""
    return sorted(p.stem for p in STATE["configs_dir"].glob("*.yaml"))


@mcp.tool()
def read_config(name: str) -> str:
    """Read a run-config YAML."""
    return (STATE["configs_dir"] / f"{name}.yaml").read_text()


@mcp.tool()
def write_config(name: str, yaml_text: str) -> str:
    """Create or overwrite a run-config YAML (validated before saving)."""
    import yaml as _yaml
    _yaml.safe_load(yaml_text)
    STATE["configs_dir"].mkdir(exist_ok=True)
    (STATE["configs_dir"] / f"{name}.yaml").write_text(yaml_text)
    return f"saved configs/{name}.yaml"


# ----------------------------------------------------------------- backtests

@mcp.tool()
def run_backtest(
    config: str,
    params: dict | None = None,
    start: str | None = None,
    end: str | None = None,
    balance: float = 100_000.0,
    warmup_days: int = 0,
) -> dict:
    """Run a single backtest. Returns metrics: sharpe, psr (probabilistic Sharpe),
    sortino, max_drawdown, pnl_pct, win_rate, total orders/positions.
    `params` overrides strategy defaults, e.g. {"fast_period": 10}."""
    from qpipe.config import RunConfig
    from qpipe.backtest.runner import run_backtest as _run

    cfg = RunConfig.from_yaml(STATE["configs_dir"] / f"{config}.yaml")
    m = _run(
        cfg, STATE["catalog"], params or {},
        pd.Timestamp(start, tz="UTC") if start else None,
        pd.Timestamp(end, tz="UTC") if end else None,
        balance=balance, warmup_days=warmup_days,
    )
    return m


@mcp.tool()
def start_optimization(
    config: str,
    segment_months: int = 6,
    windows: int = 20,
    trials: int = 50,
    jobs: int = 4,
    warmup_days: int = 90,
    holdout_months: int = 0,
    balance: float = 100_000.0,
    seed: int = 42,
) -> dict:
    """Launch a randomized IS/OOS walk-forward optimization in the background.
    segment_months must be 3, 6 or 9. Returns a job_id; poll with job_status."""
    from qpipe.config import RunConfig
    from qpipe.optimize.engine import optimize

    cfg = RunConfig.from_yaml(STATE["configs_dir"] / f"{config}.yaml")
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "status": "running", "events": [], "result": None, "error": None}
    JOBS[job_id] = job

    def work() -> None:
        try:
            result = optimize(
                cfg, STATE["catalog"], segment_months=segment_months, n_windows=windows,
                n_trials=trials, n_jobs=jobs, warmup_days=warmup_days,
                holdout_months=holdout_months, balance=balance, seed=seed,
                out_dir=str(STATE["runs_dir"]), progress_cb=job["events"].append,
            )
            job["result"] = result["summary"] | {"run_id": result["run_id"]}
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001
            job["error"], job["status"] = str(e), "failed"

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@mcp.tool()
def job_status(job_id: str) -> dict:
    """Status of an optimization job: progress events, and the recommended
    parameters + summary once done."""
    job = JOBS.get(job_id)
    if not job:
        return {"error": "unknown job_id"}
    done_windows = sum(1 for e in job["events"] if e.get("type") == "window_done")
    total = next((e["n_windows"] for e in job["events"] if e.get("type") == "started"), None)
    return {"status": job["status"], "windows_done": done_windows, "windows_total": total,
            "last_events": job["events"][-5:], "result": job["result"], "error": job["error"]}


# ---------------------------------------------------------------- past runs

@mcp.tool()
def list_runs() -> list[str]:
    """List past optimization run IDs (newest first)."""
    if not STATE["runs_dir"].exists():
        return []
    return [d.name for d in sorted(STATE["runs_dir"].iterdir(), reverse=True)
            if (d / "result.json").exists()]


@mcp.tool()
def read_run(run_id: str, full: bool = False) -> dict:
    """Read an optimization run's results. Default returns summary + candidate
    table; full=True includes every per-window result."""
    r = json.loads((STATE["runs_dir"] / run_id / "result.json").read_text())
    if full:
        return r
    return {"run_id": r["run_id"], "settings": r["settings"], "summary": r["summary"],
            "candidates": r["candidates"], "config": {k: r["config"][k] for k in
            ("strategy_path", "symbol", "objective")}}


@mcp.tool()
def read_report(run_id: str) -> str:
    """Read the markdown report for an optimization run."""
    return (STATE["runs_dir"] / run_id / "report.md").read_text()


def serve_mcp(catalog: str = "catalog", configs: str = "configs", runs: str = "runs",
              host: str = "127.0.0.1", port: int = 8765) -> None:
    STATE["catalog"] = catalog
    STATE["configs_dir"] = Path(configs)
    STATE["runs_dir"] = Path(runs)
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="streamable-http")
