"""In-app Claude agent: chat panel with pipeline tools + background agent loop.

Requires an Anthropic API key (saved via the UI to ~/.qpipe.json as
"anthropic_key", or env ANTHROPIC_API_KEY). Each chat session runs its agentic
loop in a background thread so the UI stays responsive; long optimizations are
managed via the start_optimization / job_status / wait tools.
"""

from __future__ import annotations

import json
import threading
import time
import uuid

SESSIONS: dict[str, dict] = {}
MAX_TURNS = 60
MODEL = "claude-sonnet-4-6"

SYSTEM = """You are the resident quant assistant inside qpipe, a self-hosted strategy
development pipeline built on NautilusTrader (v1.228+). You help the user develop
strategies, interpret backtests, and manage randomized IS/OOS walk-forward optimizations.

================ STRATEGY CODE CONTRACT (follow exactly) ================
Strategies live in qpipe/strategies/<name>.py (write_strategy tool). Template skeleton:

    from decimal import Decimal
    from nautilus_trader.config import StrategyConfig
    from nautilus_trader.indicators import SimpleMovingAverage  # all indicators import from here
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.trading.strategy import Strategy

    class MyStratConfig(StrategyConfig, frozen=True):   # msgspec struct: types required
        instrument_id: InstrumentId      # single-symbol strategies (auto-injected by runner)
        bar_type: BarType                # auto-injected
        my_param: int = 10               # every tunable must be a typed field with default
        trade_size: int = 100
        trade_start_ns: int = 0          # REQUIRED convention: no orders before this ts

    class MyStrat(Strategy):
        def __init__(self, config): super().__init__(config); ...
        def on_start(self):
            self.instrument = self.cache.instrument(self.config.instrument_id)
            self.register_indicator_for_bars(self.config.bar_type, self.sma)  # auto-updates
            self.subscribe_bars(self.config.bar_type)
        def on_bar(self, bar: Bar):
            if not self.sma.initialized: return
            if bar.ts_event < self.config.trade_start_ns: return   # warm-up: no trading
            # orders:
            o = self.order_factory.market(instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY, quantity=self.instrument.make_qty(Decimal(100)))
            self.submit_order(o)
            # close: self.close_all_positions(self.config.instrument_id)
        def on_stop(self):
            self.close_all_positions(self.config.instrument_id)
            self.unsubscribe_bars(self.config.bar_type)

Multi-symbol strategies instead declare `bar_types: list[str]` (auto-injected from config
universe) and parse with BarType.from_str; see etf_rotation (read it before writing one —
it also shows account-aware equity sizing, weekly/monthly rebalance keying, and the
membership_file convention for point-in-time universes). State facts:
- portfolio: self.portfolio.is_net_long(iid), .net_position(iid), .account(venue)
- positions: self.cache.positions_open(); pos.signed_qty, pos.avg_px_open
- MARGIN accounts: balance_total is NOT reduced by buys -> equity = balance + unrealized
  PnL, never balance + market value (causes leverage spiral). CASH: balance + market value.
- Integer shares only (instrument.make_qty); no fractional.
- Bars are daily by default (bar_spec 1-DAY-LAST); ts via bar.ts_event (ns). Convert:
  from nautilus_trader.core.datetime import unix_nanos_to_dt.
ALWAYS read sma_cross or etf_rotation first as a syntax reference, write the strategy,
then run a quick backtest to prove it executes (total_positions > 0) before declaring done.

================ CONFIG YAML CONTRACT ================
configs/<name>.yaml (write_config tool):
    strategy: qpipe.strategies.<module>:<Class>
    config_class: qpipe.strategies.<module>:<Class>Config
    symbol: SPY            # primary; venue: XNAS; bar_spec: 1-DAY-LAST
    account_type: CASH     # or MARGIN
    fees: none             # none | ibkr | per_share:<usd>
    universe: [QQQ, IWM]   # extra symbols (multi-asset only)
    fixed: {param: value}  # passed to config as-is
    space:                 # Optuna search space — only params here are optimized
      my_param: {type: int, low: 5, high: 50, step: 1}
      w: {type: float, low: 0.0, high: 0.4, step: 0.05}
      mode: {type: categorical, choices: [a, b]}
    constraints: ["fast < slow"]   # python exprs over params; violations pruned
    objective: sharpe      # sharpe | psr | sortino | pnl_pct | win_rate
Missing daily data auto-fetches (Alpaca -> yfinance) before runs.

================ OPTIMIZATION METHODOLOGY ================
start_optimization samples N random 3/6/9-month IS windows over the data range, Optuna
optimizes params on each IS window, validates best params on the following OOS window
(gap_days purge between), then cross-validates every distinct winner across ALL OOS
windows; recommendation = best median OOS objective. warmup_days must cover the longest
indicator lookback in calendar days (252 trading days -> use ~380). Poll job_status with
wait(seconds) between polls for long runs; report progress.

Be vigilant about overfitting: always compare median IS vs median OOS and say so plainly.
A big drop = fitting noise. Fixed universes of today's winners = survivorship bias.
Keep responses concise. When you change code or configs, state exactly what you changed.

Each user message may end with a <ui_context> JSON block auto-attached by the web app.
It tells you what the user is LOOKING AT right now: the open project (= config name),
the active tab, the current form values on that screen, and jobs they launched this
session (newest last, with job_ids). When the user says "this", "this one", "the run",
or similar, resolve it against this context before asking for clarification — e.g. on
the Walk-forward tab, "this run" means the most recent walkforward job_id listed."""


def _tools_schema(state: dict) -> list[dict]:
    return [
        {"name": "list_configs", "description": "List run-config names.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "read_config", "description": "Read a run-config YAML.",
         "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
        {"name": "write_config", "description": "Create/overwrite a run-config YAML.",
         "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "yaml_text": {"type": "string"}}, "required": ["name", "yaml_text"]}},
        {"name": "list_strategies", "description": "List strategy module names.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "read_strategy", "description": "Read strategy Python source.",
         "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
        {"name": "write_strategy", "description": "Create/overwrite a strategy module (syntax-checked).",
         "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "code": {"type": "string"}}, "required": ["name", "code"]}},
        {"name": "run_backtest", "description": "Run one backtest synchronously; returns metrics.",
         "input_schema": {"type": "object", "properties": {
             "config": {"type": "string"}, "params": {"type": "object"},
             "start": {"type": "string"}, "end": {"type": "string"},
             "balance": {"type": "number"}, "warmup_days": {"type": "integer"}}, "required": ["config"]}},
        {"name": "start_optimization", "description": "Launch background IS/OOS optimization; returns job_id.",
         "input_schema": {"type": "object", "properties": {
             "config": {"type": "string"}, "segment_months": {"type": "integer", "enum": [3, 6, 9]},
             "windows": {"type": "integer"}, "trials": {"type": "integer"}, "jobs": {"type": "integer"},
             "warmup_days": {"type": "integer"}, "holdout_months": {"type": "integer"}}, "required": ["config"]}},
        {"name": "job_status", "description": "Status/progress/result of an optimization job.",
         "input_schema": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}},
        {"name": "list_runs", "description": "List past optimization run IDs.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "read_run", "description": "Read a past run's summary and candidates.",
         "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]}},
        {"name": "list_backtest_history", "description": "List past backtests with headline metrics.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "wait", "description": "Sleep N seconds (max 120) before continuing — use while polling long jobs.",
         "input_schema": {"type": "object", "properties": {"seconds": {"type": "integer"}}, "required": ["seconds"]}},
    ]


def _exec_tool(name: str, args: dict, state: dict) -> str:
    """Execute a tool against the same internals the REST API uses."""
    from qpipe.config import RunConfig
    import pandas as pd

    configs, runs_dir, catalog = state["configs_dir"], state["runs_dir"], state["catalog"]
    strategies = state["strategies_dir"]
    try:
        if name == "list_configs":
            return json.dumps(sorted(p.stem for p in configs.glob("*.yaml")))
        if name == "read_config":
            return (configs / f"{args['name']}.yaml").read_text()
        if name == "write_config":
            import yaml
            yaml.safe_load(args["yaml_text"])
            (configs / f"{args['name']}.yaml").write_text(args["yaml_text"])
            return f"saved configs/{args['name']}.yaml"
        if name == "list_strategies":
            return json.dumps(sorted(p.stem for p in strategies.glob("*.py") if p.stem != "__init__"))
        if name == "read_strategy":
            return (strategies / f"{args['name']}.py").read_text()
        if name == "write_strategy":
            compile(args["code"], args["name"], "exec")
            (strategies / f"{args['name']}.py").write_text(args["code"])
            return f"saved qpipe/strategies/{args['name']}.py"
        if name == "run_backtest":
            from qpipe.backtest.runner import run_backtest
            from qpipe.data.autofetch import ensure_data
            cfg = RunConfig.from_yaml(configs / f"{args['config']}.yaml")
            ensure_data(catalog, cfg, log=lambda m: None)
            m = run_backtest(
                cfg, catalog, args.get("params") or {},
                pd.Timestamp(args["start"], tz="UTC") if args.get("start") else None,
                pd.Timestamp(args["end"], tz="UTC") if args.get("end") else None,
                balance=args.get("balance") or 100_000.0,
                warmup_days=args.get("warmup_days") or 0,
            )
            return json.dumps(m, default=str)
        if name == "start_optimization":
            from qpipe.web.app import OptimizeBody, api_optimize
            body = OptimizeBody(config=args["config"],
                                segment_months=args.get("segment_months", 6),
                                windows=args.get("windows", 20), trials=args.get("trials", 50),
                                jobs=args.get("jobs", 4), warmup_days=args.get("warmup_days", 90),
                                holdout_months=args.get("holdout_months", 0))
            return json.dumps(api_optimize(body))
        if name == "job_status":
            from qpipe.web.app import JOBS
            job = JOBS.get(args["job_id"])
            if not job:
                return "unknown job_id"
            done = sum(1 for e in job.get("events", []) if e.get("type") == "window_done")
            out = {"status": job["status"], "windows_done": done, "error": job["error"]}
            if job["status"] == "done" and job.get("result"):
                out["summary"] = job["result"]["summary"]
            return json.dumps(out, default=str)
        if name == "list_runs":
            return json.dumps([d.name for d in sorted(runs_dir.iterdir(), reverse=True)
                               if (d / "result.json").exists()] if runs_dir.exists() else [])
        if name == "read_run":
            r = json.loads((runs_dir / args["run_id"] / "result.json").read_text())
            return json.dumps({"summary": r["summary"], "settings": r["settings"],
                               "candidates": r["candidates"]}, default=str)
        if name == "list_backtest_history":
            from qpipe.web.app import list_backtests
            return json.dumps(list_backtests(), default=str)
        if name == "wait":
            time.sleep(min(int(args["seconds"]), 120))
            return "waited"
        return f"unknown tool {name}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def _llm_settings() -> dict:
    """Provider config from settings file. Providers:
    anthropic (default) | openrouter | local (any OpenAI-compatible base_url)."""
    import os
    from qpipe.data.autofetch import load_settings
    s = load_settings()
    provider = s.get("llm_provider", "anthropic")
    cfg = {"provider": provider,
           "model": s.get("llm_model") or {"anthropic": MODEL,
                                           "openrouter": "anthropic/claude-sonnet-4.6",
                                           "local": "local-model"}[provider]}
    if provider == "anthropic":
        cfg["key"] = os.environ.get("ANTHROPIC_API_KEY") or s.get("anthropic_key")
        if not cfg["key"]:
            raise ValueError("No Anthropic API key set (Assistant settings).")
    elif provider == "openrouter":
        cfg["key"] = s.get("openrouter_key")
        cfg["base_url"] = "https://openrouter.ai/api/v1"
        if not cfg["key"]:
            raise ValueError("No OpenRouter API key set (Assistant settings).")
    else:  # local
        cfg["key"] = s.get("local_key") or "none"
        cfg["base_url"] = s.get("local_base_url") or "http://localhost:11434/v1"  # Ollama default
    return cfg


def _openai_tools(state: dict) -> list[dict]:
    return [{"type": "function", "function": {
        "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in _tools_schema(state)]


def start_session() -> str:
    sid = uuid.uuid4().hex[:10]
    SESSIONS[sid] = {"messages": [], "display": [], "busy": False}
    return sid


def _loop_anthropic(s: dict, state: dict, cfg: dict) -> None:
    import anthropic
    client = anthropic.Anthropic(api_key=cfg["key"])
    tools = _tools_schema(state)
    for _ in range(MAX_TURNS):
        resp = client.messages.create(model=cfg["model"], max_tokens=4096, system=SYSTEM,
                                      messages=s["messages"], tools=tools)
        s["messages"].append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "text" and block.text.strip():
                s["display"].append({"role": "assistant", "text": block.text})
            elif block.type == "tool_use":
                s["display"].append({"role": "tool", "text": f"{block.name}({json.dumps(block.input)[:200]})"})
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": _exec_tool(block.name, block.input, state)[:30000]})
        if resp.stop_reason == "tool_use":
            s["messages"].append({"role": "user", "content": results})
            continue
        break


def _loop_openai(s: dict, state: dict, cfg: dict) -> None:
    """OpenRouter / local OpenAI-compatible servers (Ollama, LM Studio, vLLM...)."""
    from openai import OpenAI
    client = OpenAI(api_key=cfg["key"], base_url=cfg["base_url"])
    tools = _openai_tools(state)
    msgs = [{"role": "system", "content": SYSTEM}] + s["messages"]
    for _ in range(MAX_TURNS):
        resp = client.chat.completions.create(model=cfg["model"], messages=msgs,
                                              tools=tools, max_tokens=4096)
        m = resp.choices[0].message
        msgs.append({"role": "assistant", "content": m.content,
                     "tool_calls": [tc.model_dump() for tc in (m.tool_calls or [])] or None})
        if m.content and m.content.strip():
            s["display"].append({"role": "assistant", "text": m.content})
        if m.tool_calls:
            for tc in m.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                s["display"].append({"role": "tool", "text": f"{tc.function.name}({tc.function.arguments[:200]})"})
                msgs.append({"role": "tool", "tool_call_id": tc.id,
                             "content": _exec_tool(tc.function.name, args, state)[:30000]})
            continue
        break
    s["messages"] = msgs[1:]  # persist without system prompt


def send_message(sid: str, text: str, state: dict, context: dict | None = None) -> None:
    """Append a user message (with optional auto-attached UI context) and run the loop."""
    s = SESSIONS[sid]
    content = text
    if context:
        content = f"{text}\n\n<ui_context>{json.dumps(context)}</ui_context>"
    s["messages"].append({"role": "user", "content": content})
    s["display"].append({"role": "user", "text": text})
    s["busy"] = True

    def loop() -> None:
        try:
            cfg = _llm_settings()
            if cfg["provider"] == "anthropic":
                _loop_anthropic(s, state, cfg)
            else:
                _loop_openai(s, state, cfg)
        except Exception as e:  # noqa: BLE001
            from qpipe.web.logbuf import log_exc
            s["display"].append({"role": "error", "text": log_exc("assistant", e)})
        finally:
            s["busy"] = False

    threading.Thread(target=loop, daemon=True).start()
