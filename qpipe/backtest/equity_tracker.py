"""Actor that marks portfolio equity to market on every primary bar.

Gives a true equity curve (cash/balance + open-position value) regardless of how
often positions close — used for max drawdown, PSR, and the UI chart.
"""

from __future__ import annotations

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType


class EquityTrackerConfig(ActorConfig, frozen=True):
    bar_types: list[str]
    account_type: str = "MARGIN"
    record_start_ns: int = 0  # skip warm-up bars: curve/stats cover the test window only


class EquityTracker(Actor):
    def __init__(self, config: EquityTrackerConfig) -> None:
        super().__init__(config)
        self._bar_types = [BarType.from_str(s) for s in config.bar_types]
        self._primary = self._bar_types[0].instrument_id
        self._is_cash = config.account_type.upper() == "CASH"
        self._last: dict = {}
        self.curve: list[tuple[int, float]] = []  # (ts_ns, equity)
        self.on_point = None  # optional live callback: (ts_ns, equity) -> None

    def on_start(self) -> None:
        for bt in self._bar_types:
            self.subscribe_bars(bt)

    def on_bar(self, bar: Bar) -> None:
        self._last[bar.bar_type.instrument_id] = float(bar.close)
        if bar.bar_type.instrument_id != self._primary:
            return
        if bar.ts_event < self.config.record_start_ns:
            return  # warm-up: indicators are filling, equity is flat — don't record
        account = self.portfolio.account(self._primary.venue)
        if account is None:
            return
        equity = account.balance_total(USD).as_double()
        for pos in self.cache.positions_open():
            px = self._last.get(pos.instrument_id)
            if px is None:
                continue
            sq = float(pos.signed_qty)
            if self._is_cash:
                equity += sq * px  # cash balance excludes holdings
            else:
                equity += sq * (px - float(pos.avg_px_open))  # margin: unrealized PnL
        self.curve.append((bar.ts_event, equity))
        if self.on_point:
            self.on_point(bar.ts_event, equity)
