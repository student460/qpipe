"""In-memory ring buffer of app events/errors, served at /api/logs."""

from __future__ import annotations

import time
import traceback
from collections import deque

LOG: deque = deque(maxlen=2000)


def log(level: str, source: str, msg: str) -> None:
    LOG.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "level": level,
                "source": source, "msg": msg})


def log_exc(source: str, e: Exception) -> str:
    """Log an exception with full traceback; returns the short message."""
    log("error", source, f"{e}\n{traceback.format_exc()}")
    return str(e)
