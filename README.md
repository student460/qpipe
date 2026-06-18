# qpipe — Quant Model Development Pipeline

Strategy idea → randomized in-sample/out-of-sample optimization over 10 years → optimized model. Built on [NautilusTrader](https://nautilustrader.io) (requires Python 3.11+).

## Install

```bash
cd quant-pipeline
python -m venv .venv && source .venv/bin/activate   # or use uv
pip install -e .
```

## 1. Get data into the catalog

```bash
# Synthetic data (test the pipeline)
qpipe data-synthetic --catalog catalog --symbol TEST --years 10

# QuantConnect (daily/hourly — cheap per-file via QCC, survivorship-bias-free)
lean data download                      # QC's CLI wizard, downloads LEAN zips
qpipe data-lean --lean-dir ~/lean/data --symbols AAPL,MSFT --resolution daily

# Alpaca (free; use for minute bars)
export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...
qpipe data-alpaca --symbols AAPL,MSFT --start 2016-06-01 --timeframe 1d
```

## 2. Define a strategy

Copy `qpipe/strategies/sma_cross.py` as a template. Conventions:
- Config inherits `StrategyConfig` with `instrument_id`, `bar_type`, your tunables, and `trade_start_ns: int = 0` (skip trading before this timestamp — enables leak-free indicator warm-up).
- Declare the search space in a YAML config (see `configs/sma_cross.yaml`).

## 3. Single backtest

```bash
qpipe backtest --config configs/sma_cross.yaml --params '{"fast_period":10,"slow_period":50}'
```

## 4. Optimize

```bash
qpipe optimize --config configs/sma_cross.yaml --segment-months 6 \
    --windows 20 --trials 50 --jobs 8 --holdout-months 6
```

What happens: N random 3/6/9-month IS windows are sampled over the data range, Optuna optimizes params on each IS window (parallel workers), each winner is validated on the unseen OOS window that follows it, then every distinct winner is cross-validated on **all** OOS windows. The recommendation maximizes *median OOS objective* — robustness, not one lucky window.

Outputs in `runs/<run_id>/`:
- `optimized_model.json` — strategy + frozen params + provenance (the deliverable)
- `report.md` — recommendation, IS-vs-OOS overfitting check, per-window tables
- `result.json` — full raw results

## Interpreting the report

The **IS vs OOS** medians are the headline. IS will always look good (it's optimized). If median OOS collapses toward zero or negative, the strategy has no edge — the optimizer fit noise. Sanity check: on synthetic GBM data this pipeline correctly reports IS ≈ 1.0, OOS ≈ 0.

## Web UI

```bash
qpipe serve --catalog catalog          # -> http://127.0.0.1:8000
```

Tabs: **Strategy** (code editor, syntax-checked saves), **Parameters** (QC-wizard-style table: name/min/max/step + constraints + objective), **Backtest** (equity chart, Sharpe, probabilistic Sharpe, max drawdown, etc.), **Optimize** (launch, live per-window progress, recommended model), **Past Runs**.

## Claude / MCP server

```bash
qpipe mcp --catalog catalog            # -> http://127.0.0.1:8765/mcp
```

Exposes the pipeline as MCP tools (read/write strategies & configs, run backtests, launch/monitor optimizations, read results). Connect from Claude Desktop/Cowork via Settings → Connectors → Add custom connector with URL `http://localhost:8765/mcp`. Friends on your Tailscale network use `http://<your-machine-name>:8765/mcp` (start with `--host 0.0.0.0`).

**Security**: both servers can edit and execute code by design. Bind to localhost or your private Tailnet only; never expose to the public internet.

## Docker (run anywhere)

```bash
docker compose up --build -d     # web UI :8000, MCP :8765
docker compose logs -f           # tail logs
docker compose down              # stop
```

Catalog, configs, runs, strategy code, and API-key settings are bind-mounted, so
everything you do in the containers persists in this folder — and the same folder
works whether you run natively or in Docker. API keys can also be passed as env
vars (`ALPACA_API_KEY=… docker compose up`). On the Windows box: install Docker
Desktop (WSL2 backend), clone/copy this folder, same command. Add Tailscale on the
host and friends reach `http://<machine>:8000` / `:8765/mcp` over the tailnet —
do not port-forward these to the public internet (the app executes code by design).

## Roadmap (see ../PLAN.md)

Remaining: auth for multi-user sharing, point-in-time universe via QC fundamentals.
