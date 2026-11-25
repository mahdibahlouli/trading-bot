"""Free Dukascopy tick data (XAUUSD and other FX/CFD symbols).

Each hour of ticks is an LZMA-compressed ``.bi5`` file at::

    https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM0}/{DD}/{HH}h_ticks.bi5

where ``MM0`` is the month counted from 00 (January = 00). Each record is 20 bytes,
big-endian: ms since the hour (uint32), ask (uint32), bid (uint32), ask volume (float32),
bid volume (float32). Prices are integers divided by a per-symbol point divisor
(1000 for XAUUSD [verify for other symbols]). An empty file means no ticks that hour.

Prices are Dukascopy's own feed, not your broker's, so expect small differences.
"""

from __future__ import annotations

import logging
import lzma
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..storage import write_parquet
from .bars import ticks_to_bars

log = logging.getLogger(__name__)

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
RECORD = np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("ask_volume", ">f4"), ("bid_volume", ">f4")])
DEFAULT_DIVISORS = {"XAUUSD": 1000, "XAGUSD": 1000, "EURUSD": 100000, "GBPUSD": 100000, "USDJPY": 1000}


def hour_url(symbol: str, hour: datetime) -> str:
    return f"{BASE_URL}/{symbol.upper()}/{hour.year:04d}/{hour.month - 1:02d}/{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"


def decode_bi5(blob: bytes, hour: datetime, divisor: float) -> pd.DataFrame:
    """Decode one hour of ``.bi5`` data into a tick DataFrame with a UTC ``ts`` column."""
    cols = ["ts", "bid", "ask", "bid_volume", "ask_volume"]
    if not blob:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols}).astype({"ts": "datetime64[ns, UTC]"})
    raw = lzma.decompress(blob)
    if len(raw) % RECORD.itemsize:
        raise ValueError(f"corrupt bi5 for {hour}: {len(raw)} bytes")
    rec = np.frombuffer(raw, dtype=RECORD)
    base = pd.Timestamp(hour).tz_localize("UTC") if pd.Timestamp(hour).tz is None else pd.Timestamp(hour).tz_convert("UTC")
    return pd.DataFrame(
        {
            "ts": base + pd.to_timedelta(rec["ms"].astype("int64"), unit="ms"),
            "bid": rec["bid"].astype("float64") / divisor,
            "ask": rec["ask"].astype("float64") / divisor,
            "bid_volume": rec["bid_volume"].astype("float64"),
            "ask_volume": rec["ask_volume"].astype("float64"),
        }
    )


class DukascopyDownloader:
    def __init__(
        self,
        root: Path,
        divisors: dict[str, float] | None = None,
        workers: int = 4,
        session: requests.Session | None = None,
        retries: int = 4,
    ):
        self.root = Path(root)
        self.divisors = {**DEFAULT_DIVISORS, **(divisors or {})}
        self.workers = workers
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "tradingbot-research/0.1")
        self.retries = retries

    def tick_path(self, symbol: str, day: date) -> Path:
        return self.root / "raw" / "dukascopy" / symbol / "ticks" / f"date={day.isoformat()}.parquet"

    def bar_path(self, symbol: str, day: date) -> Path:
        return self.root / "bars" / "dukascopy" / symbol / "1m" / f"date={day.isoformat()}.parquet"

    def _fetch(self, url: str) -> bytes:
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                r = self.session.get(url, timeout=30)
                if r.status_code == 404:
                    return b""
                r.raise_for_status()
                return r.content
            except requests.RequestException as exc:
                if attempt == self.retries:
                    raise
                log.warning("retry %s after %s", url, exc)
                time.sleep(delay)
                delay *= 2
        return b""

    def download_day(self, symbol: str, day: date) -> pd.DataFrame:
        symbol = symbol.upper()
        if symbol not in self.divisors:
            raise KeyError(f"no point divisor for {symbol}; add it under [dukascopy.point_divisors]")
        divisor = self.divisors[symbol]
        hours = [datetime(day.year, day.month, day.day, h, tzinfo=timezone.utc) for h in range(24)]
        with ThreadPoolExecutor(self.workers) as pool:
            blobs = list(pool.map(lambda h: self._fetch(hour_url(symbol, h)), hours))
        frames = [decode_bi5(b, h, divisor) for b, h in zip(blobs, hours) if b]
        if not frames:
            return decode_bi5(b"", hours[0], divisor)
        return pd.concat(frames, ignore_index=True)

    def download(self, symbol: str, start: date, end: date, overwrite: bool = False) -> int:
        """Download ``[start, end)`` day by day, writing ticks and 1-minute bars.

        Days already on disk are skipped, so the command can be stopped and resumed.
        Returns the number of days written.
        """
        symbol = symbol.upper()
        written = 0
        day = start
        while day < end:
            tp = self.tick_path(symbol, day)
            if overwrite or not tp.exists():
                ticks = self.download_day(symbol, day)
                write_parquet(ticks, tp)
                if len(ticks):
                    bars = ticks_to_bars(ticks).reset_index()
                    write_parquet(bars, self.bar_path(symbol, day))
                log.info("%s %s: %d ticks", symbol, day, len(ticks))
                written += 1
            day += timedelta(days=1)
        return written
