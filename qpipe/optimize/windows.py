"""Random in-sample / out-of-sample window sampling over the data range."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WindowPair:
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "is_start": str(self.is_start.date()),
            "is_end": str(self.is_end.date()),
            "oos_start": str(self.oos_start.date()),
            "oos_end": str(self.oos_end.date()),
        }


def sample_windows(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    segment_months: int,
    n_windows: int,
    gap_days: int = 5,
    oos_months: int | None = None,
    holdout_months: int = 0,
    seed: int = 42,
) -> list[WindowPair]:
    """Sample n random contiguous IS windows, each followed by a gap then an OOS window.

    holdout_months reserves the most recent months: no window may touch them.
    """
    oos_months = oos_months or segment_months
    usable_end = data_end - pd.DateOffset(months=holdout_months)
    latest_is_start = (
        usable_end - pd.DateOffset(months=segment_months + oos_months) - pd.Timedelta(days=gap_days)
    )
    if latest_is_start <= data_start:
        raise ValueError(
            f"Data range too short for {segment_months}m IS + {oos_months}m OOS "
            f"(+{holdout_months}m holdout)"
        )

    rng = random.Random(seed)
    span_days = (latest_is_start - data_start).days
    pairs = []
    for _ in range(n_windows):
        is_start = data_start + pd.Timedelta(days=rng.randint(0, span_days))
        is_end = is_start + pd.DateOffset(months=segment_months)
        oos_start = is_end + pd.Timedelta(days=gap_days)
        oos_end = oos_start + pd.DateOffset(months=oos_months)
        pairs.append(WindowPair(is_start, is_end, oos_start, oos_end))
    return pairs
