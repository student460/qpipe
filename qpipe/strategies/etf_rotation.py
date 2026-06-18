"""Multi-factor momentum ETF rotation — port of the QuantConnect 'CreativeGreenBat' algo.

Logic (faithful to the original):
- Composite momentum score per ETF from 5 lookbacks (1w, 1m, 3m, 6m, 12m trading days),
  weighted by configurable weights; trend weight = max(0, 1 - sum of the others).
- Volume confirmation factor: mean(volume, last `volume_lookback`) / mean(volume, full
  trend window), clamped to [volume_min, volume_max]; multiplies the composite.
- Weekly (first bar of a new ISO week, i.e. Monday) or monthly (first bar of a new
  month) rebalance into the top `num_positions` ETFs, equal weight.

Port notes vs QC:
- Daily bars (the QC version used minute resolution only for scheduling).
- An ETF is only scored once it has `trend_momentum` daily bars of history (same as
  the QC backtest-mode requirement of all 5 timeframes).
- Equal-weight targets are sized from account equity (cash + position value) with
  integer shares; use a realistic starting balance (Nautilus equities don't do
  fractional shares, so $1000 across 4 ETFs won't fill — see README note).
"""

from __future__ import annotations

from collections import deque

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.trading.strategy import Strategy


class EtfRotationConfig(StrategyConfig, frozen=True):
    bar_types: list[str]
    num_positions: int = 4
    week_momentum: int = 5
    short_momentum: int = 21
    medium_momentum: int = 63
    long_momentum: int = 126
    trend_momentum: int = 252
    week_weight: float = 0.0
    short_weight: float = 0.2
    medium_weight: float = 0.2
    long_weight: float = 0.3
    volume_min: float = 0.8
    volume_max: float = 1.2
    volume_lookback: int = 21
    rebalance_frequency: str = "weekly"  # weekly | monthly
    trade_start_ns: int = 0
    # Optional point-in-time universe: JSON file of [{"from": "YYYY-MM-DD", "symbols": [...]}, ...]
    # sorted ascending. At each rebalance only symbols active on that date are scored/held.
    membership_file: str = ""
    # Dynamic market-cap universe (preferred, no static file): JSON from qpipe.data.edgar
    # {SYM: [[filed_date, shares], ...]}. At each rebalance the universe is the top
    # `universe_top_n` candidates by PIT market cap = shares(latest filing <= today) x last close.
    shares_file: str = ""
    universe_top_n: int = 10


class EtfRotation(Strategy):
    def __init__(self, config: EtfRotationConfig) -> None:
        super().__init__(config)
        self._bar_types = [BarType.from_str(s) for s in config.bar_types]
        self._primary = self._bar_types[0].instrument_id
        self._hist: dict = {}  # instrument_id -> deque[(close, volume)]
        self._last_key = None
        self._membership: list | None = None  # [(date, frozenset[str])] ascending
        if config.membership_file:
            import json
            from datetime import date as _date
            entries = json.loads(open(config.membership_file).read())
            self._membership = sorted(
                (( _date.fromisoformat(e["from"]), frozenset(e["symbols"]) ) for e in entries),
                key=lambda x: x[0],
            )
        self._shares: dict | None = None  # SYM -> [[filed_date, shares], ...]
        if config.shares_file:
            import json
            self._shares = json.loads(open(config.shares_file).read())

    def _mcap_top_n(self, dt) -> frozenset:
        """Point-in-time top-N by market cap: latest FILED share count x last close."""
        from qpipe.data.edgar import shares_asof
        d = dt.date().isoformat()
        caps = {}
        for iid, h in self._hist.items():
            sym = str(iid.symbol)
            series = self._shares.get(sym)
            if not series or not h:
                continue
            sh = shares_asof(series, d)
            if sh:
                caps[sym] = sh * h[-1][0]
        top = sorted(caps, key=caps.get, reverse=True)[: self.config.universe_top_n]
        return frozenset(top)

    def _active_symbols(self, dt) -> frozenset | None:
        """Membership set in force on date dt (None = no membership filter)."""
        if self._membership is None:
            return None
        active = None
        for d, syms in self._membership:
            if d <= dt.date():
                active = syms
            else:
                break
        return active if active is not None else frozenset()

    # ------------------------------------------------------------- lifecycle

    def on_start(self) -> None:
        maxlen = self.config.trend_momentum
        for bt in self._bar_types:
            self._hist[bt.instrument_id] = deque(maxlen=maxlen)
            self.subscribe_bars(bt)

    def on_stop(self) -> None:
        for bt in self._bar_types:
            self.close_all_positions(bt.instrument_id)
            self.unsubscribe_bars(bt)

    # ------------------------------------------------------------------ data

    def on_bar(self, bar: Bar) -> None:
        iid = bar.bar_type.instrument_id
        self._hist[iid].append((float(bar.close), float(bar.volume)))

        if iid != self._primary or bar.ts_event < self.config.trade_start_ns:
            return
        dt = unix_nanos_to_dt(bar.ts_event)
        if self.config.rebalance_frequency == "monthly":
            key = (dt.year, dt.month)
            due = key != self._last_key
        else:
            key = dt.isocalendar()[:2]  # (year, week)
            due = dt.weekday() == 0 and key != self._last_key
        if due and self._last_key is not None:
            active = self._mcap_top_n(dt) if self._shares else self._active_symbols(dt)
            self._rebalance(active)
        self._last_key = key if (due or self._last_key is None) else self._last_key

    # ------------------------------------------------------------- rebalance

    def _composite_scores(self, active: frozenset | None = None) -> dict:
        c = self.config
        periods = {
            "week": (c.week_momentum, c.week_weight),
            "short": (c.short_momentum, c.short_weight),
            "medium": (c.medium_momentum, c.medium_weight),
            "long": (c.long_momentum, c.long_weight),
            "trend": (c.trend_momentum, max(0.0, 1.0 - (c.week_weight + c.short_weight + c.medium_weight + c.long_weight))),
        }
        scores: dict = {}
        for iid, h in self._hist.items():
            if active is not None and str(iid.symbol) not in active:
                continue  # not in the point-in-time universe right now
            if len(h) < c.trend_momentum:
                continue  # require full history (QC backtest behavior)
            closes = [x[0] for x in h]
            vols = [x[1] for x in h]
            total_w = sum(w for _, w in periods.values()) or 1.0
            comp = sum(
                w / total_w * (closes[-1] - closes[-n]) / closes[-n]
                for n, w in periods.values()
            )
            avg_vol = sum(vols) / len(vols)
            recent = vols[-c.volume_lookback:]
            vf = 1.0 if avg_vol == 0 else min(max((sum(recent) / len(recent)) / avg_vol, c.volume_min), c.volume_max)
            scores[iid] = comp * vf
        return scores

    def _equity(self) -> float:
        """Account equity, computed correctly for both account types.

        CASH: buys deduct cash, so equity = cash + market value of positions.
        MARGIN: balance is NOT reduced by purchases, so adding full market value
        would double-count and compound into a leverage spiral — equity is
        balance + unrealized PnL (market value minus cost basis) instead.
        """
        from nautilus_trader.accounting.accounts.cash import CashAccount

        account = self.portfolio.account(self._primary.venue)
        balance = account.balance_total(USD).as_double()
        is_cash = isinstance(account, CashAccount)
        value = 0.0
        for pos in self.cache.positions_open():
            if pos.instrument_id not in self._hist or not self._hist[pos.instrument_id]:
                continue
            last = self._hist[pos.instrument_id][-1][0]
            sq = float(pos.signed_qty)
            value += sq * last if is_cash else sq * (last - float(pos.avg_px_open))
        return max(balance + value, 0.0)

    def _rebalance(self, active: frozenset | None = None) -> None:
        scores = self._composite_scores(active)
        if not scores:
            return
        top = sorted(scores, key=scores.get, reverse=True)[: self.config.num_positions]
        weight = 1.0 / len(top)
        equity = self._equity()

        orders = []
        for iid in self._hist:
            last_close = self._hist[iid][-1][0] if self._hist[iid] else None
            current = float(self.portfolio.net_position(iid))
            target = int(weight * equity / last_close) if (iid in top and last_close) else 0
            diff = target - current
            if diff == 0:
                continue
            instrument = self.cache.instrument(iid)
            orders.append(
                self.order_factory.market(
                    instrument_id=iid,
                    order_side=OrderSide.BUY if diff > 0 else OrderSide.SELL,
                    quantity=instrument.make_qty(abs(diff)),
                )
            )
        # Sells first so cash is freed before buys (matters for CASH accounts)
        for o in sorted(orders, key=lambda o: 0 if o.side == OrderSide.SELL else 1):
            self.submit_order(o)
