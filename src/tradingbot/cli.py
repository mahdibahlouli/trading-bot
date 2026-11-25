"""Command line entry point: ``tb <command>``. Run ``tb -h`` for help."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone

from .config import data_dir, load_config


def _date(s: str) -> date:
    return date.fromisoformat(s)


def cmd_download(args, cfg) -> int:
    root = data_dir(cfg)
    end = args.end or date.today()
    if args.source == "dukascopy":
        from .data.dukascopy import DukascopyDownloader

        d = DukascopyDownloader(root, cfg["dukascopy"].get("point_divisors"), cfg["dukascopy"].get("workers", 4))
        n = d.download(args.symbol, args.start, end, overwrite=args.overwrite)
    else:
        from .data.binance import BinanceArchive

        a = BinanceArchive(root, market=args.market or cfg["binance"]["market"])
        n = a.download(args.symbol, args.start, end, datatype=args.datatype, interval=args.interval, overwrite=args.overwrite)
    print(f"wrote {n} files under {root}")
    return 0


def cmd_record(args, cfg) -> int:
    from .data.hyperliquid.recorder import run_recorder
    from .data.hyperliquid.sink import ParquetSink

    hl = dict(cfg["hyperliquid"])
    if args.coins:
        hl["coins"] = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    if args.top is not None:
        hl["top_addresses"] = args.top
    if args.fill_addresses is not None:
        hl["fill_stream_addresses"] = args.fill_addresses
    root = data_dir(cfg)
    print(f"recording Hyperliquid public data to {root / 'raw' / 'hyperliquid'} (Ctrl+C to stop)")
    run_recorder(hl, ParquetSink(root))
    return 0


def cmd_hl_status(args, cfg) -> int:
    import pandas as pd

    from .data.hyperliquid.analysis import gap_report, load_stream

    root = data_dir(cfg)
    try:
        hb = load_stream(root, "heartbeat")
    except Exception as exc:
        print(f"no heartbeat data yet under {root} ({exc})")
        return 1
    rep = gap_report(hb, hours=args.hours)
    first = pd.to_datetime(hb["ts"], utc=True).min()
    print(f"recording since {first:%Y-%m-%d %H:%M} UTC; window: last {args.hours:g} h")
    print(rep.to_string(index=False))
    ok = bool(rep["pass"].all())
    print("PASS: no gaps" if ok else "FAIL: gaps found (see above)")
    return 0 if ok else 2


def cmd_hl_compact(args, cfg) -> int:
    from .data.hyperliquid.sink import compact_day

    root = data_dir(cfg)
    base = root / "raw" / "hyperliquid"
    today = datetime.now(timezone.utc).date().isoformat()
    n = 0
    for stream_dir in sorted(p for p in base.glob("*") if p.is_dir()):
        for day_dir in sorted(stream_dir.glob("date=*")):
            day = day_dir.name.split("=", 1)[1]
            if day >= today:
                continue  # the recorder may still be writing today
            if compact_day(root, stream_dir.name, day):
                n += 1
    print(f"compacted {n} day folders")
    return 0


def cmd_quality(args, cfg) -> int:
    from .data.bars import load_bars, quality_report

    bars = load_bars(args.source, args.symbol, "1min", args.start, args.end, root=data_dir(cfg))
    rep = quality_report(bars, expect_weekends=args.source.startswith("binance"))
    for k, v in rep.as_dict().items():
        print(f"{k:>18}: {v}")
    return 0


def cmd_sources(args, cfg) -> int:
    from .data.bars import sources

    print("\n".join(sources()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tb", description="Trading research data tools (no live trading).")
    p.add_argument("--config", help="extra TOML file applied over config/default.toml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download free historical data")
    d.add_argument("source", choices=["dukascopy", "binance"])
    d.add_argument("--symbol", required=True, help="e.g. XAUUSD (Dukascopy) or BTCUSDT (Binance)")
    d.add_argument("--start", type=_date, required=True, help="YYYY-MM-DD, inclusive")
    d.add_argument("--end", type=_date, help="YYYY-MM-DD, exclusive (default: today)")
    d.add_argument("--market", choices=["spot", "futures_um", "futures_cm"], help="Binance market")
    d.add_argument("--datatype", default="klines", choices=["klines", "aggTrades", "trades"], help="Binance file type")
    d.add_argument("--interval", default="1m", help="Binance kline interval")
    d.add_argument("--overwrite", action="store_true")
    d.set_defaults(fn=cmd_download)

    r = sub.add_parser("record", help="run the Hyperliquid public-data recorder until stopped")
    r.add_argument("venue", choices=["hyperliquid"])
    r.add_argument("--coins", help="comma-separated, e.g. BTC,ETH,SOL")
    r.add_argument("--top", type=int, help="leaderboard addresses to snapshot")
    r.add_argument("--fill-addresses", type=int, help="addresses to stream live fills for")
    r.set_defaults(fn=cmd_record)

    s = sub.add_parser("hl-status", help="check the recorder for gaps (Phase 0 gate: 72 h, no gaps)")
    s.add_argument("--hours", type=float, default=72)
    s.set_defaults(fn=cmd_hl_status)

    c = sub.add_parser("hl-compact", help="merge finished days' recorder part files")
    c.set_defaults(fn=cmd_hl_compact)

    q = sub.add_parser("quality", help="data-quality report for stored 1m bars")
    q.add_argument("--source", required=True, help="see `tb sources`")
    q.add_argument("--symbol", required=True)
    q.add_argument("--start")
    q.add_argument("--end")
    q.set_defaults(fn=cmd_quality)

    so = sub.add_parser("sources", help="list bar sources")
    so.set_defaults(fn=cmd_sources)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    return args.fn(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
