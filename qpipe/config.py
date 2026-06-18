"""Run configuration: strategy reference, fixed params, and the parameter search space.

A run config is a YAML file like:

    strategy: qpipe.strategies.sma_cross:SMACross
    config_class: qpipe.strategies.sma_cross:SMACrossConfig
    symbol: AAPL
    venue: XNAS
    bar_spec: 1-DAY-LAST
    fixed:
      trade_size: 100
    space:
      fast_period: {type: int, low: 5, high: 50}
      slow_period: {type: int, low: 20, high: 200}
    constraints:
      - "fast_period < slow_period"
    objective: sharpe          # sharpe | sortino | pnl_pct | calmar
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ParamSpec:
    name: str
    type: str  # int | float | categorical
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list[Any] | None = None

    def suggest(self, trial) -> Any:
        if self.type == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high), step=int(self.step or 1))
        if self.type == "float":
            return trial.suggest_float(self.name, self.low, self.high, step=self.step, log=self.log)
        if self.type == "categorical":
            return trial.suggest_categorical(self.name, self.choices)
        raise ValueError(f"Unknown param type: {self.type}")


@dataclass
class RunConfig:
    strategy_path: str
    config_class_path: str
    symbol: str
    venue: str
    bar_spec: str
    fixed: dict[str, Any] = field(default_factory=dict)
    space: list[ParamSpec] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    objective: str = "sharpe"
    universe: list[str] = field(default_factory=list)  # extra symbols for multi-asset strategies
    account_type: str = "MARGIN"  # MARGIN | CASH
    fees: str = "none"  # none | ibkr | per_share:<usd>

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text())
        space = [ParamSpec(name=k, **v) for k, v in raw.get("space", {}).items()]
        return cls(
            strategy_path=raw["strategy"],
            config_class_path=raw["config_class"],
            symbol=raw["symbol"],
            venue=raw.get("venue", "XNAS"),
            bar_spec=raw.get("bar_spec", "1-DAY-LAST"),
            fixed=raw.get("fixed", {}),
            space=space,
            constraints=raw.get("constraints", []),
            objective=raw.get("objective", "sharpe"),
            universe=raw.get("universe", []),
            account_type=raw.get("account_type", "MARGIN"),
            fees=raw.get("fees", "none"),
        )

    @property
    def bar_type_str(self) -> str:
        return f"{self.symbol}.{self.venue}-{self.bar_spec}-EXTERNAL"

    @property
    def all_bar_type_strs(self) -> list[str]:
        """Primary + universe bar types (deduplicated, primary first)."""
        out = [self.bar_type_str]
        for s in self.universe:
            bts = f"{s}.{self.venue}-{self.bar_spec}-EXTERNAL"
            if bts not in out:
                out.append(bts)
        return out

    def check_constraints(self, params: dict[str, Any]) -> bool:
        scope = {**self.fixed, **params}
        return all(eval(c, {"__builtins__": {}}, scope) for c in self.constraints)  # noqa: S307


def load_class(path: str):
    """Load 'module.sub:ClassName'."""
    module_path, _, cls_name = path.partition(":")
    return getattr(importlib.import_module(module_path), cls_name)
