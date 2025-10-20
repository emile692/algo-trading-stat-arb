import pandas as pd
import numpy as np
from typing import List, Optional
from .config import Settings

def _select_cols(df: pd.DataFrame, tickers: List[str], suffix: Optional[str]) -> pd.DataFrame:
    cols = {}
    for t in tickers:
        col = f"{t}{suffix}" if suffix else t
        if col not in df.columns:
            raise KeyError(f"Colonne manquante: {col}")
        cols[t] = df[col]
    return pd.DataFrame(cols)

def _resample(df: pd.DataFrame, freq: str, how: str) -> pd.DataFrame:
    if df.index.freq == freq or freq is None:
        return df
    if how == "last":
        return df.resample(freq).last()
    if how == "mean":
        return df.resample(freq).mean()
    if how == "ohlc":
        # Si tu fournis déjà OHLC par ticker, adapte ici.
        return df.resample(freq).last()
    return df.resample(freq).last()

def load_prices(cfg: Settings) -> pd.DataFrame:
    """
    Charge un CSV multi-tickers, aligne l'index temporel, resample, et (optionnel) log-transform.
    Retourne un DataFrame indexé temps avec colonnes = tickers.
    """
    df = pd.read_csv(cfg.price_path)
    if cfg.date_col not in df.columns:
        raise KeyError(f"date_col '{cfg.date_col}' absent")
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], utc=True, errors="coerce")
    df = df.dropna(subset=[cfg.date_col]).set_index(cfg.date_col).sort_index()

    if cfg.tz:
        df = df.tz_convert(cfg.tz)

    px = _select_cols(df, cfg.tickers, cfg.price_col_suffix)

    # Optionnel: coupe la période
    if cfg.start:
        px = px.loc[cfg.start:]
    if cfg.end:
        px = px.loc[:cfg.end]

    # Resample régulier (1H ou 1D)
    px = _resample(px, cfg.freq_out, cfg.how)
    px = px.dropna(how="all")

    if cfg.log_prices:
        px = np.log(px)

    # Nettoyage final
    px = px.replace([np.inf, -np.inf], np.nan).dropna()
    return px

def get_pairs(tickers: List[str]) -> List[tuple]:
    """Toutes les paires non ordonnées (A,B) avec A!=B (sans doublons inversés)."""
    out = []
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            out.append((tickers[i], tickers[j]))
    return out
