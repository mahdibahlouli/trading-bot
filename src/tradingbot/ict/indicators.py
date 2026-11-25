from __future__ import annotations

import pandas as pd


def bar_delta(bars: pd.DataFrame) -> pd.Timedelta:
    """The bar length, taken as the most common gap between bar open times."""
    if len(bars) < 2:
        return pd.Timedelta(minutes=1)
    return bars.index.to_series().diff().mode().iloc[0]


def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's Average True Range. The value at bar ``i`` uses bars up to and including ``i``."""
    prev_close = bars["close"].shift(1)
    tr = pd.concat(
        [bars["high"] - bars["low"], (bars["high"] - prev_close).abs(), (bars["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().rename(f"atr{n}")
