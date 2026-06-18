"""CreativeGreenBat — SPY Top-10 Market-Cap Filter → Top-N Momentum Rotation.

Logic:
1.  At every rebalance, filter the full universe down to the TOP_MCAP_COUNT (default 10)
    symbols by current market cap (shares_outstanding × last close).
2.  Score each of those 10 by a composite multi-factor momentum (same 5-lookback
    formula as the original EtfRotation / QC CreativeGreenBat).
3.  Hold the top NUM_POSITIONS (default 2) from that scored set, equal-weight.

=== FILTER PARAMS (top of file — easy to tweak / wire into Optuna) ===
All tuneable knobs live in CreativeGreenBatConfig; defaults match the QC original.

Survivorship note: `shares_outstanding` values are baked into each instrument's
metadata at ingest time. Combined with last-close they give a *proxy* market cap
that is point-in-time for price but NOT for share-count changes. This is still
substantially better than a fixed hand-picked list, but a full PIT share-count
file would eliminate the residual bias entirely.
"""

from __future__ import annotations

from collections import deque

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.trading.strategy import Strategy


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← all filter / momentum / weight knobs live here
# ─────────────────────────────────────────────────────────────────────────────

class CreativeGreenBatConfig(StrategyConfig, frozen=True):
    bar_types: list[str]

    # ── Universe-filter params ────────────────────────────────────────────────
    top_mcap_count: int = 10      # how many top-market-cap stocks to keep
    num_positions: int = 2        # how many of those to actually hold (top momentum)

    # ── Momentum lookback windows (trading days) ──────────────────────────────
    week_momentum: int = 5
    short_momentum: int = 21
    medium_momentum: int = 63
    long_momentum: int = 126
    trend_momentum: int = 252     # also controls the minimum history required

    # ── Composite-score weights (trend_weight = max(0, 1 - sum of the rest)) ──
    week_weight: float = 0.0
    short_weight: float = 0.2
    medium_weight: float = 0.2
    long_weight: float = 0.3
    # trend_weight is derived: max(0, 1 - week - short - medium - long)

    # ── Volume-confirmation filter ────────────────────────────────────────────
    volume_lookback: int = 21     # recent-volume window
    volume_min: float = 0.8       # clamp floor for volume factor
    volume_max: float = 1.2       # clamp ceiling for volume factor

    # ── Rebalance schedule ────────────────────────────────────────────────────
    rebalance_frequency: str = "weekly"   # "weekly" | "monthly"

    trade_start_ns: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

class CreativeGreenBat(Strategy):
    """SPY top-10-by-market-cap filter → top-N momentum rotation."""

    def __init__(self, config: CreativeGreenBatConfig) -> None:
        super().__init__(config)
        self._bar_types = [BarType.from_str(s) for s in config.bar_types]
        self._primary = self._bar_types[0].instrument_id   # drives rebalance clock
        # instrument_id -> deque of (close, volume)
        self._hist: dict = {}
        self._last_key = None

    # ──────────────────────────────────────────── lifecycle

    def on_start(self) -> None:
        maxlen = self.config.trend_momentum
        for bt in self._bar_types:
            self._hist[bt.instrument_id] = deque(maxlen=maxlen)
            self.subscribe_bars(bt)

    def on_stop(self) -> None:
        for bt in self._bar_types:
            self.close_all_positions(bt.instrument_id)
            self.unsubscribe_bars(bt)

    # ──────────────────────────────────────────── data

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
            key = dt.isocalendar()[:2]          # (ISO-year, ISO-week)
            due = dt.weekday() == 0 and key != self._last_key

        if due and self._last_key is not None:
            self._rebalance()

        if due or self._last_key is None:
            self._last_key = key

    # ──────────────────────────────────────────── step 1: market-cap filter

    def _top_mcap_iids(self) -> list:
        """Return up to top_mcap_count instrument IDs ranked by proxy market cap.

        Proxy market cap = shares_outstanding (from instrument metadata) × last close.
        Instruments with no history or no shares_outstanding metadata are excluded.
        """
        c = self.config
        caps: list[tuple[float, object]] = []

        for iid, h in self._hist.items():
            if not h:
                continue
            last_close = h[-1][0]
            instrument = self.cache.instrument(iid)
            if instrument is None:
                continue
            # NautilusTrader equity instruments expose `shares_outstanding`; fall back
            # to 0 so those symbols sort to the bottom rather than crashing.
            shares = getattr(instrument, "shares_outstanding", None) or 0
            if shares <= 0:
                # No metadata — include but rank last so they can still participate
                # if the whole universe lacks share data (common in test fixtures).
                market_cap = 0.0
            else:
                market_cap = shares * last_close
            caps.append((market_cap, iid))

        caps.sort(key=lambda x: x[0], reverse=True)
        return [iid for _, iid in caps[: c.top_mcap_count]]

    # ──────────────────────────────────────────── step 2: momentum scores

    def _composite_scores(self, candidate_iids: list) -> dict:
        """Compute multi-factor momentum scores for the given candidate set."""
        c = self.config
        trend_w = max(0.0, 1.0 - (c.week_weight + c.short_weight + c.medium_weight + c.long_weight))
        periods = [
            (c.week_momentum,  c.week_weight),
            (c.short_momentum, c.short_weight),
            (c.medium_momentum, c.medium_weight),
            (c.long_momentum,  c.long_weight),
            (c.trend_momentum, trend_w),
        ]
        total_w = sum(w for _, w in periods) or 1.0

        scores: dict = {}
        for iid in candidate_iids:
            h = self._hist.get(iid)
            if h is None or len(h) < c.trend_momentum:
                continue    # require full trend_momentum bars (QC warm-up equivalent)

            closes = [x[0] for x in h]
            vols   = [x[1] for x in h]

            # Composite momentum
            comp = sum(
                w / total_w * (closes[-1] - closes[-n]) / closes[-n]
                for n, w in periods
            )

            # Volume confirmation factor
            avg_vol = sum(vols) / len(vols)
            if avg_vol == 0:
                vf = 1.0
            else:
                recent_avg = sum(vols[-c.volume_lookback:]) / c.volume_lookback
                vf = min(max(recent_avg / avg_vol, c.volume_min), c.volume_max)

            scores[iid] = comp * vf

        return scores

    # ──────────────────────────────────────────── equity helper

    def _equity(self) -> float:
        """Account equity (works for both CASH and MARGIN accounts)."""
        from nautilus_trader.accounting.accounts.cash import CashAccount

        account  = self.portfolio.account(self._primary.venue)
        balance  = account.balance_total(USD).as_double()
        is_cash  = isinstance(account, CashAccount)
        value    = 0.0
        for pos in self.cache.positions_open():
            h = self._hist.get(pos.instrument_id)
            if not h:
                continue
            last = h[-1][0]
            sq   = float(pos.signed_qty)
            value += sq * last if is_cash else sq * (last - float(pos.avg_px_open))
        return max(balance + value, 0.0)

    # ──────────────────────────────────────────── rebalance

    def _rebalance(self) -> None:
        c = self.config

        # ── Step 1: market-cap filter → top_mcap_count stocks ────────────────
        mcap_top = self._top_mcap_iids()
        if not mcap_top:
            return

        # ── Step 2: momentum score the market-cap candidates ─────────────────
        scores = self._composite_scores(mcap_top)
        if not scores:
            return

        # ── Step 3: pick top num_positions by momentum score ─────────────────
        top = sorted(scores, key=scores.get, reverse=True)[: c.num_positions]
        weight = 1.0 / len(top)
        equity = self._equity()

        # ── Step 4: size and submit orders (sells before buys) ────────────────
        orders = []
        for iid, h in self._hist.items():
            if not h:
                continue
            last_close = h[-1][0]
            current    = float(self.portfolio.net_position(iid))
            target     = int(weight * equity / last_close) if iid in top else 0
            diff       = target - current
            if diff == 0:
                continue
            instrument = self.cache.instrument(iid)
            if instrument is None:
                continue
            orders.append(
                self.order_factory.market(
                    instrument_id=iid,
                    order_side=OrderSide.BUY if diff > 0 else OrderSide.SELL,
                    quantity=instrument.make_qty(abs(diff)),
                )
            )

        for o in sorted(orders, key=lambda o: 0 if o.side == OrderSide.SELL else 1):
            self.submit_order(o)
