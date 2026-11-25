"""Normalized bars shared by every data source.

Every source is turned into the same schema so rules and research code never depend on a
venue. Swapping venue means swapping the ``source`` argument of :func:`load_bars`, nothing else.

Bar schema: UTC ``ts`` (bar open time) as the index, ``open high low close volume``,
plus optional extra columns (``spread``, ``ticks``, ``trades`` ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from ..config import data_dir
from ..storage import read_parquet_glob
from ..timeutils import ensure_utc_index, utc_timestamp

OHLCV = ["open", "high", "low", "close", "volume"]


def ticks_to_bars(ticks: pd.DataFrame, freq: str = "1min", price: str = "bid") -> pd.DataFrame:
    """Build OHLC bars from bid/ask ticks.

    ``ticks`` needs a UTC ``ts`` column (or index) and ``bid``/``ask`` columns, optionally
    ``bid_volume``/``ask_volume``. ``price`` picks ``bid``, ``ask`` or ``mid`` for the OHLC.
    Bars with no ticks are left out (no forward fill), so gaps stay visible.
    """
    t = ensure_utc_index(ticks)
    if price == "mid":
        p = (t["bid"] + t["ask"]) / 2
    else:
        p = t[price]
    g = p.resample(freq, label="left", closed="left")
    bars = pd.DataFrame(
        {"open": g.first(), "high": g.max(), "low": g.min(), "close": g.last()}
    )
    vol_cols = [c for c in ("bid_volume", "ask_volume") if c in t.columns]
    if vol_cols:
        bars["volume"] = t[vol_cols].sum(axis=1).resample(freq, label="left", closed="left").sum()
    else:
        bars["volume"] = 0.0
    spread = (t["ask"] - t["bid"]).resample(freq, label="left", closed="left")
    bars["spread"] = spread.mean()
    bars["spread_max"] = spread.max()
    bars["ticks"] = p.resample(freq, label="left", closed="left").count()
    bars = bars[bars["ticks"] > 0]
    bars["ticks"] = bars["ticks"].astype("int64")
    return bars


def resample_bars(bars: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample normalized bars to a higher timeframe (for example ``"5min"`` or ``"15min"``)."""
    b = ensure_utc_index(bars)
    agg: dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last"}
    for col, how in (("volume", "sum"), ("ticks", "sum"), ("trades", "sum"),
                     ("spread", "mean"), ("spread_max", "max")):
        if col in b.columns:
            agg[col] = how
    out = b.resample(freq, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open"])


@dataclass
class QualityReport:
    bars: int
    missing_minutes: int
    duplicate_bars: int
    zero_volume_bars: int
    bad_ohlc: int
    spikes: int
    weekend_bars: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def quality_report(
    bars: pd.DataFrame,
    freq: str = "1min",
    spike_k: float = 10.0,
    expect_weekends: bool = False,
) -> QualityReport:
    """Count common data problems.

    ``missing_minutes`` only counts gaps inside trading periods: gaps longer than 2 hours are
    treated as market closures. ``spikes`` counts bars whose range exceeds ``spike_k`` times the
    rolling median range.
    """
    raw = bars
    if not isinstance(raw.index, pd.DatetimeIndex) and "ts" in raw.columns:
        dup = int(raw["ts"].duplicated().sum())
    else:
        dup = int(raw.index.duplicated().sum())
    b = ensure_utc_index(raw)
    b = b[~b.index.duplicated()]
    step = pd.Timedelta(freq)
    diffs = b.index.to_series().diff().dropna()
    gaps = diffs[(diffs > step) & (diffs <= pd.Timedelta(hours=2))]
    missing = int(((gaps // step) - 1).sum()) if len(gaps) else 0
    rng = b["high"] - b["low"]
    med = rng.rolling(500, min_periods=20).median()
    spikes = int((rng > spike_k * med).sum())
    bad = int(
        ((b["high"] < b[["open", "close"]].max(axis=1)) | (b["low"] > b[["open", "close"]].min(axis=1))).sum()
    )
    zero_vol = int((b["volume"] <= 0).sum()) if "volume" in b.columns else 0
    weekend = 0
    if not expect_weekends:
        ny = b.index.tz_convert("America/New_York")
        # FX/metals close Friday 17:00 NY and reopen Sunday 17:00 NY.
        fri_late = (ny.dayofweek == 4) & (ny.hour >= 17)
        sat = ny.dayofweek == 5
        sun_early = (ny.dayofweek == 6) & (ny.hour < 17)
        weekend = int((fri_late | sat | sun_early).sum())
    return QualityReport(len(b), missing, dup, zero_vol, bad, spikes, weekend)


# --------------------------------------------------------------------------------------
# Source registry: one loader per venue, all returning the normalized schema.

Loader = Callable[[str, str, Path], pd.DataFrame]
_LOADERS: dict[str, Loader] = {}


def register_source(name: str):
    def deco(fn: Loader) -> Loader:
        _LOADERS[name] = fn
        return fn
    return deco


def sources() -> list[str]:
    return sorted(_LOADERS)


@register_source("dukascopy")
def _load_dukascopy(symbol: str, timeframe: str, root: Path) -> pd.DataFrame:
    return read_parquet_glob(root / "bars" / "dukascopy" / symbol / "1m" / "*.parquet")


def _binance_loader(market: str) -> Loader:
    def load(symbol: str, timeframe: str, root: Path) -> pd.DataFrame:
        return read_parquet_glob(root / "raw" / "binance" / market / symbol / "klines_1m" / "*.parquet")
    return load


for _m in ("spot", "futures_um", "futures_cm"):
    register_source(f"binance_{_m}")(_binance_loader(_m))


def load_bars(
    source: str,
    symbol: str,
    timeframe: str = "1min",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Load stored 1-minute bars from ``source`` and resample to ``timeframe``.

    ``source`` is one of :func:`sources` (``dukascopy``, ``binance_futures_um`` ...).
    """
    if source not in _LOADERS:
        raise KeyError(f"unknown source {source!r}; known: {sources()}")
    root = root or data_dir()
    df = _LOADERS[source](symbol, timeframe, root)
    df = ensure_utc_index(df)
    df = df[~df.index.duplicated(keep="last")]
    if start is not None:
        df = df[df.index >= utc_timestamp(start)]
    if end is not None:
        df = df[df.index < utc_timestamp(end)]
    if timeframe not in ("1min", "1m", "1T"):
        df = resample_bars(df, timeframe)
    return df
