"""Buffered Parquet writer for recorder streams.

Rows are kept in memory and flushed to
``{root}/raw/hyperliquid/{stream}/date=YYYY-MM-DD/{HHMMSS}-{id}.parquet``
(partitioned by each row's UTC ``ts`` date). Each flush writes new files, so a crash
loses at most one flush interval and never corrupts earlier data.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ...storage import write_parquet

log = logging.getLogger(__name__)


class ParquetSink:
    def __init__(self, root: Path, prefix: str = "raw/hyperliquid", max_rows: int = 200_000):
        self.base = Path(root) / prefix
        self.max_rows = max_rows
        self.buffers: dict[str, list[dict]] = defaultdict(list)
        self.lock = threading.Lock()
        self.rows_written: dict[str, int] = defaultdict(int)

    def add(self, stream: str, rows: list[dict]) -> None:
        if not rows:
            return
        with self.lock:
            self.buffers[stream].extend(rows)
            big = len(self.buffers[stream]) >= self.max_rows
        if big:
            self.flush(stream)

    def pending(self, stream: str) -> int:
        with self.lock:
            return len(self.buffers.get(stream, []))

    def flush(self, stream: str | None = None) -> list[Path]:
        with self.lock:
            names = [stream] if stream else list(self.buffers)
            taken = {n: self.buffers.pop(n, []) for n in names}
        paths: list[Path] = []
        for name, rows in taken.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            stamp = pd.Timestamp.now(tz="UTC").strftime("%H%M%S")
            for day, part in df.groupby(df["ts"].dt.strftime("%Y-%m-%d")):
                path = self.base / name / f"date={day}" / f"{stamp}-{uuid.uuid4().hex[:8]}.parquet"
                write_parquet(part.reset_index(drop=True), path)
                paths.append(path)
            self.rows_written[name] += len(df)
            log.info("flushed %d %s rows", len(df), name)
        return paths


def compact_day(root: Path, stream: str, day: str, prefix: str = "raw/hyperliquid") -> Path | None:
    """Merge one finished day's part files into a single file (safe to re-run)."""
    d = Path(root) / prefix / stream / f"date={day}"
    parts = sorted(p for p in d.glob("*.parquet") if not p.name.startswith("compacted"))
    if len(parts) <= 1:
        return None
    frames = [pd.read_parquet(p) for p in parts]
    existing = d / "compacted.parquet"
    if existing.exists():
        frames.insert(0, pd.read_parquet(existing))
    df = pd.concat(frames, ignore_index=True).sort_values("ts", kind="stable")
    write_parquet(df, existing)
    for p in parts:
        p.unlink()
    return existing
