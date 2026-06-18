# qpipe — quick start (and context for your Claude)

You've been sent **qpipe**: a self-hosted quant strategy pipeline built on NautilusTrader.
You write a trading strategy (or let Claude write it), backtest it on 10 years of data,
then stress-test parameters with randomized in-sample/out-of-sample optimization and
rolling walk-forward analysis. Paste this whole file into Claude for context — it explains
how everything works.

## Setup (one time, ~5 min)

1. Install Docker Desktop (docker.com), start it.
2. Unzip `quant-pipeline.zip`, then in Terminal:
   ```bash
   cd quant-pipeline
   docker compose up --build
   ```
3. Open http://localhost:8000 — that's the app.
4. (Optional) Data tab → enter your own Alpaca API keys (free account at alpaca.markets)
   for market data. Without keys it falls back to yfinance automatically.

## Using it

- **Projects** home page: each card = one strategy config. "+ New project" scaffolds one.
- **Strategy** tab: the Python strategy code (NautilusTrader). `sma_cross` and
  `etf_rotation` are working examples to copy.
- **Parameters** tab: which params are fixed vs optimizable (ranges/steps/constraints),
  and the objective (sharpe, psr, sortino...).
- **Backtest** tab: run it, watch the equity curve draw live vs SPY, get Sharpe,
  probabilistic Sharpe, max drawdown, trade-by-trade log, advanced stats dropdown.
- **Optimize** tab: randomized walk-forward — N random 3/6/9-month in-sample windows,
  params optimized on each, validated on unseen out-of-sample windows, cross-validated.
  The recommendation maximizes *median OOS* performance. The heat map should be a flat,
  single-ish color — a lone bright island means overfitting.
- **Walk-forward** tab: classic rolling IS/OOS grid search with a stitched out-of-sample
  equity curve — the closest thing to "how would this have traded live".
- **Logs** tab: full tracebacks when something fails.
- **Rule of thumb**: in-sample results always look good; only out-of-sample counts.

## Connecting YOUR Claude to it (recommended)

The container also runs an MCP server at `http://localhost:8765/mcp`. Claude connected to
it can read/write strategies and configs, run backtests, launch and monitor optimizations,
and read all results — so you can just say "write me a mean-reversion strategy on QQQ,
backtest it, and tell me if it survives out-of-sample".

Claude desktop app → Settings → Connectors → Add custom connector.
Custom connectors require HTTPS, so localhost http will be rejected. Easiest fix:
```bash
tailscale serve --bg http://localhost:8765    # install tailscale.com first
tailscale serve status                         # shows your https URL
```
Use `https://<your-machine>.<your-tailnet>.ts.net/mcp/` (note trailing slash) as the URL.

There's also a built-in Assistant panel (✳ button in the app) — it needs its own model:
an Anthropic API key (pay-per-token), an OpenRouter key, or a free local model via
Ollama (`ollama pull qwen3.6:27b` if you have the GPU for it).

## Safety notes

- This app executes Python code by design. Keep it on localhost or a private
  Tailscale network. Never port-forward 8000/8765 to the open internet.
- Backtests are simulations: no slippage modeling, integer shares, and fixed universes
  of today's popular tickers carry survivorship bias. Don't trade real money on a
  pretty backtest — trust the out-of-sample numbers, and even then, be skeptical.
- Nothing here is financial advice; it's a research tool.
