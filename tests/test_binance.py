import hashlib
import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from tradingbot.data.binance import BinanceArchive, archive_url, months_between, parse_agg_trades, parse_klines
from conftest import FakeResponse, FakeSession

HEADER = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
ROW_MS = "1704067200000,42000.1,42010,41990,42005,12.5,1704067259999,525000,321,6.1,256000,0\n"
ROW_US = "1735689600000000,93000,93010,92990,93005,1.5,1735689659999999,139000,42,0.7,65000,0\n"


def zipped(name, text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def test_urls():
    assert archive_url("futures_um", "monthly", "klines", "btcusdt", "2024-01", "1m") == (
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    )
    assert archive_url("spot", "daily", "aggTrades", "ETHUSDT", "2024-01-02").endswith(
        "/spot/daily/aggTrades/ETHUSDT/ETHUSDT-aggTrades-2024-01-02.zip"
    )


def test_months_between():
    assert months_between(date(2023, 11, 15), date(2024, 2, 1)) == ["2023-11", "2023-12", "2024-01"]


def test_parse_klines_with_header_ms():
    df = parse_klines(zipped("a.csv", HEADER + ROW_MS))
    assert df["ts"].iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert df["close"].iloc[0] == 42005 and df["trades"].iloc[0] == 321


def test_parse_klines_no_header_microseconds():
    df = parse_klines(zipped("a.csv", ROW_US))
    assert df["ts"].iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")


def test_parse_agg_trades():
    text = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n" \
           "1,42000.5,0.01,10,12,1704067200123,true\n2,42001,0.5,13,13,1704067200456,false\n"
    df = parse_agg_trades(zipped("a.csv", text))
    assert list(df["is_buyer_maker"]) == [True, False]
    assert df["ts"].iloc[1] == pd.Timestamp("2024-01-01 00:00:00.456", tz="UTC")


def _routes(url, blob, good_checksum=True):
    digest = hashlib.sha256(blob if good_checksum else b"x").hexdigest()
    return {url: FakeResponse(blob), url + ".CHECKSUM": FakeResponse(f"{digest}  file.zip".encode())}


def test_download_monthly_with_checksum(tmp_path):
    blob = zipped("BTCUSDT-1m-2024-01.csv", HEADER + ROW_MS)
    url = archive_url("futures_um", "monthly", "klines", "BTCUSDT", "2024-01", "1m")
    a = BinanceArchive(tmp_path, session=FakeSession(_routes(url, blob)))
    assert a.download("BTCUSDT", date(2024, 1, 1), date(2024, 2, 1)) == 1
    out = pd.read_parquet(a.out_path("BTCUSDT", "klines", "2024-01", "1m"))
    assert len(out) == 1


def test_checksum_mismatch_raises(tmp_path):
    blob = zipped("x.csv", HEADER + ROW_MS)
    url = archive_url("futures_um", "monthly", "klines", "BTCUSDT", "2024-01", "1m")
    a = BinanceArchive(tmp_path, session=FakeSession(_routes(url, blob, good_checksum=False)))
    with pytest.raises(IOError):
        a.download("BTCUSDT", date(2024, 1, 1), date(2024, 2, 1))


def test_falls_back_to_daily_files_as_partial(tmp_path):
    blob = zipped("x.csv", HEADER + ROW_MS)
    url = archive_url("futures_um", "daily", "klines", "BTCUSDT", "2024-01-01", "1m")
    a = BinanceArchive(tmp_path, session=FakeSession(_routes(url, blob)))
    assert a.download("BTCUSDT", date(2024, 1, 1), date(2024, 1, 3)) == 1
    assert (a.out_path("BTCUSDT", "klines", "2024-01", "1m").with_name("2024-01.partial.parquet")).exists()
