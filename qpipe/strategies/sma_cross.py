"""Example strategy: SMA crossover (long when fast > slow, flat otherwise)."""

from __future__ import annotations

from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators import SimpleMovingAverage
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy


class SMACrossConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    fast_period: int = 10
    slow_period: int = 50
    trade_size: int = 100
    trade_start_ns: int = 0  # warm-up convention: no orders before this timestamp


class SMACross(Strategy):
    def __init__(self, config: SMACrossConfig) -> None:
        super().__init__(config)
        self.fast = SimpleMovingAverage(config.fast_period)
        self.slow = SimpleMovingAverage(config.slow_period)
        self.instrument = None

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        self.register_indicator_for_bars(self.config.bar_type, self.fast)
        self.register_indicator_for_bars(self.config.bar_type, self.slow)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.fast.initialized or not self.slow.initialized:
            return
        if bar.ts_event < self.config.trade_start_ns:
            return  # warm-up period: indicators update, no trading

        is_long = self.portfolio.is_net_long(self.config.instrument_id)

        if self.fast.value > self.slow.value and not is_long:
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.instrument.make_qty(Decimal(self.config.trade_size)),
            )
            self.submit_order(order)
        elif self.fast.value <= self.slow.value and is_long:
            self.close_all_positions(self.config.instrument_id)

    def on_stop(self) -> None:
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)
