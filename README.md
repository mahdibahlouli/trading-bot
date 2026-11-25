# trading-bot

Research and data tooling for turning ICT-style discretionary trading into tested, automated rules.
It follows the plan in the project's `automation-plan.md` (Phase 0 and the start of Phases 1–2).

**What this repo does not do:** place orders, hold API keys, or log in to any account. Everything
uses free public data. Execution comes later, only after the plan's go/no-go gates, and only on a
venue that is legal where you trade from.

## Setup

Python 3.11 or newer.

```bash
git clone https://github.com/mahdibahlouli/trading-bot.git
cd trading-bot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                             # all tests run offline
```

Settings live in `config/default.toml`. Put your own overrides in `config/local.toml`
(ignored by Git). Data goes to `data/` (also ignored); set `TB_DATA_DIR` to put it elsewhere,
for example on an external drive.

## 1. Hyperliquid recorder (start this first)

The on-chain "smart money" ideas need history that you can't download for free, so the plan
says to start recording on day one and test after about 3 months.

```bash
tb record hyperliquid                      # coins, intervals etc. from config
tb record hyperliquid --coins BTC,ETH,SOL --top 300 --fill-addresses 50
```

It runs until you stop it (Ctrl+C or SIGTERM flushes buffers first) and survives network drops.
What it saves, under `data/raw/hyperliquid/<stream>/date=YYYY-MM-DD/`:

| Stream | How often | Contents |
|---|---|---|
| `asset_ctxs` | 60 s | funding, open interest, mark/oracle/mid price, premium, day volume per coin |
| `positions` | 5 min | every open position of the tracked addresses: size, entry, leverage, **liquidation price**, uPnL |
| `accounts` | 5 min | account value, margin used, withdrawable of the tracked addresses |
| `leaderboard` | 6 h | the public leaderboard (top 2,000 by account value), kept whole for survivorship checks |
| `trades` | live | every trade on the configured coins, with buyer and seller addresses |
| `fills` | live | fills of the top tracked addresses; **liquidation fills are flagged** (`is_liquidation`) |
| `heartbeat`, `events` | each cycle | used to prove there were no gaps |

Tracked addresses = `extra_addresses` from config, plus the top `top_addresses` leaderboard
accounts by monthly PnL with at least $50k account value, refreshed every 6 hours.

**Liquidations:** as far as I know Hyperliquid has no public market-wide liquidation stream
[verify]. The recorder covers it three ways: liquidation fills of tracked addresses (exact),
liquidation prices of every tracked position (a "liquidation map"), and
`tradingbot.data.hyperliquid.analysis.infer_liquidations`, which flags positions that vanished
while the mark price crossed their liquidation price.

**Rate limits:** the recorder stays under `max_weight_per_minute = 600`, half of Hyperliquid's
per-IP budget of 1,200 [verify]. 300 addresses every 5 minutes uses about 120 weight a minute.

**Phase 0 gate (72 hours, no gaps):**

```bash
tb hl-status              # last 72 h; exit code 0 = PASS
tb hl-compact             # merge finished days' small files into one per day
```

**Keeping it running.** It needs an always-on machine with internet: your own computer if it
stays on, a Raspberry Pi, or a small cheap VPS.
- Linux with systemd: `deploy/tb-recorder.service` (instructions inside).
- macOS / Linux without systemd: `nohup deploy/run-recorder.sh > recorder.log 2>&1 &`.

Disk use depends mostly on trade volume; check the size of `data/raw/hyperliquid` after the first day.

**Reading the data** (DuckDB, no server needed):

```python
from tradingbot.storage import read_parquet_glob
pos = read_parquet_glob("data/raw/hyperliquid/positions/*/*.parquet",
                        "select coin, sum(szi * entry_px) as net_notional from t group by coin")
```

## 2. Historical data downloads

```bash
# XAUUSD ticks from Dukascopy -> daily tick files + 1-minute bars (resumable)
tb download dukascopy --symbol XAUUSD --start 2019-01-01 --end 2026-11-01

# Binance USDⓈ-M perps 1-minute klines and aggTrades (checksums verified)
tb download binance --symbol BTCUSDT --start 2020-01-01
tb download binance --symbol ETHUSDT --start 2020-01-01 --datatype aggTrades
tb download binance --symbol BTCUSDT --start 2024-01-01 --market spot

tb quality --source dukascopy --symbol XAUUSD   # missing minutes, duplicates, spikes, bad OHLC
```

Dukascopy is Dukascopy's own price feed, not Exness's, so spreads and prices will differ a little.
Downloading Binance's archive needs no account; trading there depends on your country.

## 3. Swappable venues

Every source is normalized to the same bar schema (UTC `ts` index, `open high low close volume`),
and research code loads bars through one function:

```python
from tradingbot.data.bars import load_bars
gold = load_bars("dukascopy", "XAUUSD", "5min", start="2024-01-01")
btc  = load_bars("binance_futures_um", "BTCUSDT", "15min")
```

Adding a venue means registering one loader with `@register_source("name")` in
`tradingbot/data/bars.py`. Rules never know which venue the bars came from.

## 4. ICT rules (`tradingbot.ict`)

Precise versions of the definitions in plan section 4.2. All session logic runs in New York time
(DST handled); data stays UTC. Every level and event carries the time it becomes **known**
(`known_from` / `confirmed_ts`), so research code can't accidentally peek ahead.

| Function | What it gives you |
|---|---|
| `label_sessions(bars)` | per bar: NY time, trading day (rolls at 18:00 NY), `in_asia`, `in_london_kz`, `in_ny_am_kz`, `in_ny_pm` |
| `session_ranges(bars)` | high/low/open/close of each session per day, known from the session's end |
| `previous_day_levels`, `previous_week_levels` | PDH/PDL, PWH/PWL |
| `session_levels(bars)` | Asia and London highs/lows as levels |
| `swings(bars, n)`, `swing_levels`, `equal_levels` | swing points known only `n` bars later; equal highs/lows |
| `detect_sweeps(bars, levels)` | trade beyond a level by ≥ `0.1 × ATR`, close back within 3 bars; breaks optional |
| `find_fvgs(bars)` | bullish/bearish FVGs with first touch, 50% (CE), invalidation and expiry times |

```python
import pandas as pd
from tradingbot.data.bars import load_bars
from tradingbot.ict import previous_day_levels, session_levels, detect_sweeps, find_fvgs, label_sessions

bars = load_bars("dukascopy", "XAUUSD", "5min", start="2023-01-01")
levels = pd.concat([previous_day_levels(bars), session_levels(bars)])
sweeps = detect_sweeps(bars, levels)
sweeps["session"] = label_sessions(bars).loc[sweeps["break_ts"], "session"].to_numpy()
london_asia_sweeps = sweeps[(sweeps.session == "london_kz") & sweeps.level_name.str.startswith("asia")]
fvgs = find_fvgs(bars, min_size_atr=0.3)
```

Defaults (sessions, ATR period, sweep size, FVG lifetime) are in `[ict]` in `config/default.toml`.
The plan says: adjust them once, then freeze them before testing.

## Layout

```
config/          default.toml (+ your local.toml)
data/            downloaded and recorded data (not in Git)
deploy/          systemd unit and restart script for the recorder
research/        event studies (next phase)
strategies/      strategy definitions (later)
live/            intentionally empty: no execution yet
src/tradingbot/
  data/          dukascopy.py, binance.py, bars.py, hyperliquid/
  ict/           sessions.py, levels.py, sweeps.py, fvg.py, indicators.py
  cli.py         the `tb` command
tests/           offline tests for everything above
```
