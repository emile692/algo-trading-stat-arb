from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

def extract_field_yf(df: pd.DataFrame, field: str = "Close") -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = set(df.columns.get_level_values(0))
        lv1 = set(df.columns.get_level_values(1))
        if field in lv0:
            sub = df[field].copy()
        elif field in lv1:
            sub = df.xs(field, axis=1, level=1).copy()
        else:
            raise ValueError(f"Champ {field!r} introuvable.")
    else:
        if field in df.columns:
            sub = df[[field]].copy()
        else:
            raise ValueError(f"{field!r} introuvable en colonnes simples.")
    sub = sub.reindex(sorted(sub.columns), axis=1)
    return sub

def prepare_prices(prices: pd.DataFrame, tz: str | None = None) -> pd.DataFrame:
    p = prices.copy().sort_index()
    if not isinstance(p.index, pd.DatetimeIndex):
        p.index = pd.to_datetime(p.index, utc=True, errors="coerce")
    if p.index.tz is None:
        p.index = p.index.tz_localize("UTC")
    if tz:
        p.index = p.index.tz_convert(tz)
    p = p.ffill().bfill()
    p = p.dropna(axis=1, how="any")
    return p

def compute_returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    if log:
        ret = np.log(prices).diff()
    else:
        ret = prices.pct_change()
    return ret.dropna(how="all")

def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
