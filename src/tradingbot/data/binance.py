"""Free Binance public archive (https://data.binance.vision).

Files live at::

    https://data.binance.vision/data/{market_path}/{period}/{datatype}/{SYMBOL}/[{interval}/]{name}.zip

with a ``.CHECKSUM`` file (SHA-256) next to each zip. ``market_path`` is ``spot``,
``futures/um`` (USDT-margined perps) or ``futures/cm``. ``period`` is ``monthly`` or ``daily``.

Format quirks handled here:
* futures CSVs have a header row; older spot CSVs do not;
* spot timestamps switched from milliseconds to microseconds in 2025.

Downloading data does not need a Binance account. Whether you may *trade* on Binance
depends on your country; that is a separate question.
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from ..storage import write_parquet

log = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data"
MARKET_PATHS = {"spot": "spot", "futures_um": "futures/um", "futures_cm": "futures/cm"}

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
    "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
AGG_COLS = ["agg_id", "price", "qty", "first_id", "last_id", "transact_time", "is_buyer_maker", "is_best_match"]


def archive_url(market: str, period: str, datatype: str, symbol: str, stamp: str, interval: str | None = None) -> str:
    mp = MARKET_PATHS[market]
    symbol = symbol.upper()
    if datatype == "klines":
        if not interval:
            raise ValueError("klines need an interval")
        return f"{BASE_URL}/{mp}/{period}/klines/{symbol}/{interval}/{symbol}-{interval}-{stamp}.zip"
    return f"{BASE_URL}/{mp}/{period}/{datatype}/{symbol}/{symbol}-{datatype}-{stamp}.zip"


def _to_utc(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values).astype("int64")
    # > 1e14 means microseconds (spot from 2025), otherwise milliseconds.
    unit = "us" if len(v) and v.iloc[0] > 10**14 else "ms"
    return pd.to_datetime(v, unit=unit, utc=True)


def _read_csv(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        text = zf.read(name)
    first = text.split(b"\n", 1)[0].split(b",")[0].strip()
    has_header = not first.lstrip(b"-").replace(b".", b"").isdigit()
    return pd.read_csv(io.BytesIO(text), header=0 if has_header else None)


def parse_klines(blob: bytes) -> pd.DataFrame:
    df = _read_csv(blob)
    df = df.iloc[:, : len(KLINE_COLS)]
    df.columns = KLINE_COLS[: df.shape[1]]
    out = pd.DataFrame({"ts": _to_utc(df["open_time"])})
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"):
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    out["trades"] = pd.to_numeric(df["trades"]).astype("int64")
    return out


def parse_agg_trades(blob: bytes) -> pd.DataFrame:
    df = _read_csv(blob)
    df.columns = AGG_COLS[: df.shape[1]]
    is_bm = df["is_buyer_maker"]
    if is_bm.dtype != bool:
        is_bm = is_bm.astype(str).str.lower().isin(["true", "1"])
    return pd.DataFrame(
        {
            "ts": _to_utc(df["transact_time"]),
            "agg_id": df["agg_id"].astype("int64"),
            "price": df["price"].astype("float64"),
            "qty": df["qty"].astype("float64"),
            "first_id": df["first_id"].astype("int64"),
            "last_id": df["last_id"].astype("int64"),
            "is_buyer_maker": is_bm.astype(bool),
        }
    )


TRADE_COLS = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]


def parse_trades(blob: bytes) -> pd.DataFrame:
    df = _read_csv(blob)
    df.columns = TRADE_COLS[: df.shape[1]]
    is_bm = df["is_buyer_maker"]
    if is_bm.dtype != bool:
        is_bm = is_bm.astype(str).str.lower().isin(["true", "1"])
    return pd.DataFrame(
        {
            "ts": _to_utc(df["time"]),
            "id": df["id"].astype("int64"),
            "price": df["price"].astype("float64"),
            "qty": df["qty"].astype("float64"),
            "is_buyer_maker": is_bm.astype(bool),
        }
    )


PARSERS = {"klines": parse_klines, "aggTrades": parse_agg_trades, "trades": parse_trades}


def months_between(start: date, end: date) -> list[str]:
    """Month stamps ``YYYY-MM`` covering ``[start, end)``."""
    out, y, m = [], start.year, start.month
    while date(y, m, 1) < end:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


class BinanceArchive:
    def __init__(self, root: Path, market: str = "futures_um", session: requests.Session | None = None, retries: int = 4):
        if market not in MARKET_PATHS:
            raise ValueError(f"market must be one of {list(MARKET_PATHS)}")
        self.root = Path(root)
        self.market = market
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "tradingbot-research/0.1")
        self.retries = retries

    def out_path(self, symbol: str, datatype: str, stamp: str, interval: str | None = None) -> Path:
        sub = f"klines_{interval}" if datatype == "klines" else datatype
        return self.root / "raw" / "binance" / self.market / symbol.upper() / sub / f"{stamp}.parquet"

    def _get(self, url: str) -> bytes | None:
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                r = self.session.get(url, timeout=120)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.content
            except requests.RequestException as exc:
                if attempt == self.retries:
                    raise
                log.warning("retry %s after %s", url, exc)
                time.sleep(delay)
                delay *= 2
        return None

    def fetch_verified(self, url: str) -> bytes | None:
        """Download a zip and check it against its published SHA-256. ``None`` if missing."""
        blob = self._get(url)
        if blob is None:
            return None
        checksum = self._get(url + ".CHECKSUM")
        if checksum:
            expected = checksum.decode().split()[0].strip().lower()
            actual = hashlib.sha256(blob).hexdigest()
            if expected != actual:
                raise IOError(f"checksum mismatch for {url}")
        else:
            log.warning("no checksum published for %s", url)
        return blob

    def download(
        self,
        symbol: str,
        start: date,
        end: date,
        datatype: str = "klines",
        interval: str = "1m",
        overwrite: bool = False,
    ) -> int:
        """Download monthly files for ``[start, end)``; months not yet published fall back to daily files.

        Returns the number of files written. Re-running skips months already on disk.
        """
        parser = PARSERS[datatype]
        iv = interval if datatype == "klines" else None
        written = 0
        for stamp in months_between(start, end):
            path = self.out_path(symbol, datatype, stamp, iv)
            if path.exists() and not overwrite:
                continue
            blob = self.fetch_verified(archive_url(self.market, "monthly", datatype, symbol, stamp, iv))
            if blob is not None:
                df = parser(blob)
                partial = path.with_name(f"{stamp}.partial.parquet")
                if partial.exists():
                    partial.unlink()
            else:
                df = self._month_from_daily(symbol, datatype, stamp, iv, parser, end)
                if df is None:
                    log.info("%s %s %s: not published", symbol, datatype, stamp)
                    continue
                # A month built from daily files may be incomplete; name it so it gets refetched.
                path = path.with_name(f"{stamp}.partial.parquet")
            write_parquet(df, path)
            log.info("%s %s %s: %d rows", symbol, datatype, stamp, len(df))
            written += 1
        return written

    def _month_from_daily(self, symbol, datatype, stamp, interval, parser, end: date):
        y, m = map(int, stamp.split("-"))
        frames = []
        for day in pd.date_range(date(y, m, 1), periods=31, freq="D"):
            if day.month != m or day.date() >= min(end, date.today()):
                break
            blob = self.fetch_verified(archive_url(self.market, "daily", datatype, symbol, day.strftime("%Y-%m-%d"), interval))
            if blob is not None:
                frames.append(parser(blob))
        return pd.concat(frames, ignore_index=True) if frames else None
