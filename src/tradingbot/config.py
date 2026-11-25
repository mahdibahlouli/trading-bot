"""Load settings from config/default.toml, overlaid by config/local.toml if present."""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.toml"
LOCAL_CONFIG = REPO_ROOT / "config" / "local.toml"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Return the merged configuration.

    ``path`` (or the ``TB_CONFIG`` environment variable) points to an extra TOML file
    applied on top of the defaults and ``config/local.toml``.
    """
    with open(DEFAULT_CONFIG, "rb") as fh:
        cfg = tomllib.load(fh)
    extra = [LOCAL_CONFIG]
    env_path = os.environ.get("TB_CONFIG")
    if env_path:
        extra.append(Path(env_path))
    if path:
        extra.append(Path(path))
    for p in extra:
        if p.exists():
            with open(p, "rb") as fh:
                cfg = _merge(cfg, tomllib.load(fh))
    return cfg


def data_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    d = Path(os.environ.get("TB_DATA_DIR", cfg.get("data_dir", "data")))
    if not d.is_absolute():
        d = REPO_ROOT / d
    return d
