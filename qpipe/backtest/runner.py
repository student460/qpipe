"""Single-backtest runner: catalog bars + strategy + params -> metrics dict."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from qpipe.backtest.fees import make_fee_model
from qpipe.config import RunConfig, load_class

_NS = 1_000_000_000


@lru_cache(maxsize=8)
def load_market(catalog_path: str, bar_type_str: str):
    """Load (instrument, bars) once per process; cached for reuse across trials."""
    catalog = ParquetDataCatalog(catalog_path)
    bar_type = BarType.from_str(bar_type_str)
    instrument_id = bar_type.instrument_id
    instruments = catalog.instruments(instrument_ids=[str(instrument_id)])
    if not instruments:
        raise ValueError(f"Instrument {instrument_id} not in catalog {catalog_path}")
    bars = catalog.bars([bar_type_str])
    if not bars:
        raise ValueError(f"No bars for {bar_type_str} in catalog {catalog_path}")
    return instruments[0], bars


def data_range(catalog_path: str, bar_type_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    _, bars = load_market(catalog_path, bar_type_str)
    return (
        pd.Timestamp(bars[0].ts_event, unit="ns", tz="UTC"),
        pd.Timestamp(bars[-1].ts_event, unit="ns", tz="UTC"),
    )


def _slice(bars, start: pd.Timestamp | None, end: pd.Timestamp | None):
    if start is None and end is None:
        return bars
    s = int(start.value) if start is not None else 0
    e = int(end.value) if end is not None else 2**63 - 1
    return [b for b in bars if s <= b.ts_event <= e]


def _risk_stats(equity: pd.Series | None, balance: float) -> dict[str, Any]:
    """Max drawdown and probabilistic Sharpe (PSR vs SR=0, Bailey & Lopez de Prado)
    from a mark-to-market equity series."""
    out: dict[str, Any] = {"max_drawdown": None, "psr": None, "equity_curve": None}
    if equity is None or len(equity) < 3:
        return out
    r = equity.pct_change().dropna()
    peak = equity.cummax()
    out["max_drawdown"] = float(((equity - peak) / peak).min())
    if len(r) < 3:
        return out
    r = r[r.abs() > 0] if (r != 0).sum() >= 3 else r  # PSR on active days
    sd = r.std()
    if sd and sd > 0:
        sr = float(r.mean() / sd)
        n = len(r)
        skew, kurt = float(r.skew()), float(r.kurt()) + 3.0  # raw kurtosis
        denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
        if denom > 0:
            z = sr * math.sqrt(n - 1) / math.sqrt(denom)
            out["psr"] = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    out["equity_curve"] = [
        [int(pd.Timestamp(ts).value // 1_000_000), round(float(v), 2)] for ts, v in equity.items()
    ]
    return out


def _grab(stats: dict, needle: str) -> float | None:
    for k, v in stats.items():
        if needle.lower() in k.lower():
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return v if math.isfinite(v) else None
    return None


def run_backtest(
    run_cfg: RunConfig,
    catalog_path: str,
    params: dict[str, Any],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    balance: float = 100_000.0,
    warmup_days: int = 0,
    with_curve: bool = False,
    with_trades: bool = False,
    point_cb=None,  # live equity callback: (ts_ns, equity) -> None
) -> dict[str, Any]:
    """Run one backtest over [start, end] and return a flat metrics dict.

    warmup_days prepends bars before `start` so indicators initialize, while the
    strategy is told (via the `trade_start_ns` config convention) not to trade
    until `start`. No PnL accrues during warm-up, so metrics stay leakage-free.
    """
    data_start = start - pd.Timedelta(days=warmup_days) if (start is not None and warmup_days) else start

    instruments, all_bars = [], []
    for bts in run_cfg.all_bar_type_strs:
        try:
            inst, bars = load_market(catalog_path, bts)
        except ValueError:
            continue  # universe symbol missing from catalog: skip
        instruments.append(inst)
        all_bars.extend(_slice(bars, data_start, end))
    if not instruments:
        return {"error": "no_instruments", "n_bars": 0}
    instrument = instruments[0]  # primary
    all_bars.sort(key=lambda b: b.ts_init)
    if len(all_bars) < 10:
        return {"error": "insufficient_bars", "n_bars": len(all_bars)}

    engine = BacktestEngine(
        config=BacktestEngineConfig(logging=LoggingConfig(bypass_logging=True))
    )
    engine.add_venue(
        venue=Venue(run_cfg.venue),
        oms_type=OmsType.NETTING,
        account_type=AccountType[run_cfg.account_type.upper()],
        base_currency=USD,
        starting_balances=[Money(balance, USD)],
        fee_model=make_fee_model(run_cfg.fees),
    )
    for inst in instruments:
        engine.add_instrument(inst)
    engine.add_data(all_bars)

    strategy_cls = load_class(run_cfg.strategy_path)
    config_cls = load_class(run_cfg.config_class_path)
    # Auto-injected fields are filtered to what the config class declares;
    # user fixed/space params pass through unfiltered (typos should error loudly).
    declared = set(getattr(config_cls, "__struct_fields__", ()))
    auto = {
        "instrument_id": instrument.id,
        "bar_type": BarType.from_str(run_cfg.bar_type_str),
        "bar_types": [bts for bts in run_cfg.all_bar_type_strs],
    }
    if start is not None and warmup_days:
        auto["trade_start_ns"] = int(start.value)
    cfg_kwargs = {k: v for k, v in auto.items() if k in declared}
    cfg_kwargs.update({**run_cfg.fixed, **params})
    strategy = strategy_cls(config=config_cls(**cfg_kwargs))

    from qpipe.backtest.equity_tracker import EquityTracker, EquityTrackerConfig
    tracker = EquityTracker(EquityTrackerConfig(
        bar_types=run_cfg.all_bar_type_strs, account_type=run_cfg.account_type,
        record_start_ns=int(start.value) if (start is not None and warmup_days) else 0))
    tracker.on_point = point_cb
    engine.add_actor(tracker)
    # flag truncated warm-up: not enough data before `start` to fill the requested warm-up
    warmup_truncated = bool(
        start is not None and warmup_days and all_bars
        and all_bars[0].ts_event > int((start - pd.Timedelta(days=warmup_days * 0.9)).value)
    )
    engine.add_strategy(strategy)

    try:
        engine.run()
        result = engine.get_result()
        ret, pnl = result.stats_returns, result.stats_pnls.get("USD", {})
        metrics = {
            "sharpe": _grab(ret, "sharpe"),
            "sortino": _grab(ret, "sortino"),
            "volatility": _grab(ret, "volatility"),
            "pnl_total": _grab(pnl, "PnL (total)"),
            "pnl_pct": _grab(pnl, "%"),
            "win_rate": _grab(pnl, "win rate"),
            "expectancy": _grab(pnl, "expectancy"),
            "total_orders": result.total_orders,
            "total_positions": result.total_positions,
            "n_bars": len(all_bars),
            "warmup_truncated": warmup_truncated,
        }
        equity = (
            pd.Series(
                [v for _, v in tracker.curve],
                index=[pd.Timestamp(t, unit="ns", tz="UTC") for t, _ in tracker.curve],
            )
            if tracker.curve else None
        )
        risk = _risk_stats(equity, balance)
        metrics["max_drawdown"] = risk["max_drawdown"]
        metrics["psr"] = risk["psr"]
        if with_curve:
            metrics["equity_curve"] = risk["equity_curve"]
        if with_trades:
            trades = []
            for o in engine.cache.orders():
                qty = o.filled_qty.as_double()
                if qty <= 0 or o.avg_px is None:
                    continue
                px = float(o.avg_px)
                trades.append({
                    "ts": int(o.ts_last // 1_000_000),
                    "symbol": str(o.instrument_id.symbol),
                    "side": "BUY" if o.side == OrderSide.BUY else "SELL",
                    "qty": qty,
                    "price": round(px, 4),
                    "value": round(qty * px, 2),
                })
            trades.sort(key=lambda t: t["ts"])
            metrics["trades"] = trades
    finally:
        engine.dispose()
    return metrics


def objective_value(metrics: dict[str, Any], objective: str, no_trade_value: float = -1e6) -> float:
    """Map a metrics dict to a single maximizable scalar.

    no_trade_value: score when the strategy never traded. Use a large penalty
    during IS optimization (we want configs that trade), but 0.0 when evaluating
    on OOS windows (staying flat is a legitimate, neutral outcome).
    """
    if metrics.get("error") or not metrics.get("total_positions"):
        return no_trade_value
    val = metrics.get(objective)
    return val if val is not None and math.isfinite(val) else no_trade_value
