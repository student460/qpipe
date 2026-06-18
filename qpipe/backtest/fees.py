"""Broker fee models for backtests.

Configure per run-config YAML:
    fees: ibkr          # IB Fixed pricing, US stocks
    fees: none          # commission-free (Alpaca-like). Default.
    fees: per_share:0.005
"""

from __future__ import annotations

from nautilus_trader.backtest.models import FeeModel
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import Order


class IBKRFixedFeeModel(FeeModel):
    """Interactive Brokers US stocks, Fixed pricing:
    $0.005/share, min $1.00 per order, max 1% of trade value."""

    def get_commission(self, order: Order, fill_qty: Quantity, fill_px: Price, instrument: Instrument) -> Money:
        qty = fill_qty.as_double()
        notional = qty * fill_px.as_double()
        fee = min(max(0.005 * qty, 1.00), 0.01 * notional) if notional > 0 else 0.0
        return Money(fee, USD)


class PerShareFeeModel(FeeModel):
    def __init__(self, per_share: float) -> None:
        super().__init__()
        self._per_share = per_share

    def get_commission(self, order: Order, fill_qty: Quantity, fill_px: Price, instrument: Instrument) -> Money:
        return Money(self._per_share * fill_qty.as_double(), USD)


def make_fee_model(spec: str) -> FeeModel | None:
    spec = (spec or "none").strip().lower()
    if spec == "none":
        return None
    if spec == "ibkr":
        return IBKRFixedFeeModel()
    if spec.startswith("per_share:"):
        return PerShareFeeModel(float(spec.split(":", 1)[1]))
    raise ValueError(f"Unknown fee model: {spec}")
