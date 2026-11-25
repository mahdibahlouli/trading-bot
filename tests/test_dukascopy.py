import lzma
import struct
from datetime import date, datetime, timezone

import pandas as pd

from tradingbot.data.dukascopy import DukascopyDownloader, decode_bi5, hour_url
from conftest import FakeResponse, FakeSession


def bi5(records):
    raw = b"".join(struct.pack(">3I2f", *r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def test_hour_url_uses_zero_based_month():
    url = hour_url("xauusd", datetime(2024, 1, 2, 10, tzinfo=timezone.utc))
    assert url.endswith("/XAUUSD/2024/00/02/10h_ticks.bi5")


def test_decode_bi5_prices_and_times():
    hour = datetime(2024, 3, 5, 14, tzinfo=timezone.utc)
    blob = bi5([(0, 2050125, 2050000, 1.5, 2.0), (61_500, 2051000, 2050800, 0.5, 0.25)])
    df = decode_bi5(blob, hour, 1000)
    assert list(df["bid"]) == [2050.0, 2050.8]
    assert list(df["ask"]) == [2050.125, 2051.0]
    assert df["ts"].iloc[1] == pd.Timestamp("2024-03-05 14:01:01.500", tz="UTC")
    assert df["ts"].dt.tz is not None


def test_decode_empty_hour():
    df = decode_bi5(b"", datetime(2024, 3, 9, tzinfo=timezone.utc), 1000)
    assert df.empty and "bid" in df.columns


def test_download_writes_ticks_and_bars(tmp_path):
    day = date(2024, 3, 5)
    url = hour_url("XAUUSD", datetime(2024, 3, 5, 14, tzinfo=timezone.utc))
    blob = bi5([(0, 2050100, 2050000, 1, 1), (30_000, 2050600, 2050500, 1, 1), (65_000, 2050300, 2050200, 1, 1)])
    sess = FakeSession({url: FakeResponse(blob)})
    d = DukascopyDownloader(tmp_path, session=sess, workers=2)
    assert d.download("XAUUSD", day, date(2024, 3, 6)) == 1
    ticks = pd.read_parquet(d.tick_path("XAUUSD", day))
    bars = pd.read_parquet(d.bar_path("XAUUSD", day))
    assert len(ticks) == 3
    assert len(bars) == 2
    first = bars.iloc[0]
    assert (first["open"], first["high"], first["low"], first["close"]) == (2050.0, 2050.5, 2050.0, 2050.5)
    assert first["ticks"] == 2
    # Resumable: a second run skips the day.
    n_calls = len(sess.calls)
    assert d.download("XAUUSD", day, date(2024, 3, 6)) == 0
    assert len(sess.calls) == n_calls
