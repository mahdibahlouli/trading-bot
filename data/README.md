# data/

Everything here is downloaded or recorded, and ignored by Git.

```
data/
  raw/
    dukascopy/XAUUSD/ticks/date=YYYY-MM-DD.parquet
    binance/futures_um/BTCUSDT/klines_1m/YYYY-MM.parquet
    binance/futures_um/BTCUSDT/aggTrades/YYYY-MM.parquet
    hyperliquid/<stream>/date=YYYY-MM-DD/<part>.parquet
  bars/
    dukascopy/XAUUSD/1m/date=YYYY-MM-DD.parquet
```

All timestamps on disk are UTC (`ts`, timezone-aware). Convert to New York time only inside rules.
