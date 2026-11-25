"""Parquet helpers. Files are written atomically (temp file, then rename)."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, path: str | os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def read_parquet_glob(pattern: str | os.PathLike, query: str | None = None) -> pd.DataFrame:
    """Read every Parquet file matching ``pattern`` with DuckDB.

    ``query`` may reference the files as the table ``t``, e.g. ``"select * from t where coin='BTC'"``.
    """
    import duckdb

    con = duckdb.connect()
    try:
        con.execute(
            f"create view t as select * from read_parquet('{Path(pattern).as_posix()}', union_by_name=true)"
        )
        return con.execute(query or "select * from t").df()
    finally:
        con.close()
