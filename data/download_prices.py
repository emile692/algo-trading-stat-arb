# ============================================================
# data/download_prices.py
# ============================================================
import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

def _need_redownload(path_out: str, expected: list[str]) -> bool:
    p = Path(path_out)
    if not p.exists():
        return True
    try:
        df = pd.read_csv(p, nrows=1)
        cols = [c for c in df.columns if c != "Date"]
        # on vérifie que l'ensemble ET l'ordre correspondent
        return cols != list(expected)
    except Exception:
        return True

def download_prices(
    tickers: list[str],
    start: str = "2020-01-01",
    end: str | None = None,
    interval: str = "1h",
    path_out: str = "data/prices.csv",
    force: bool = False,
) -> pd.DataFrame:
    """
    Télécharge les prix ajustés pour `tickers` et sauvegarde un CSV
    aligné avec uniquement ces colonnes (dans le même ordre).
    """
    Path(path_out).parent.mkdir(parents=True, exist_ok=True)

    if not force and not _need_redownload(path_out, tickers):
        print(f"📁 {Path(path_out).name} déjà conforme, skip.")
        return pd.read_csv(path_out, parse_dates=["Date"], index_col="Date")

    print(f"⏳ Téléchargement {len(tickers)} tickers de {start} à {end or 'today'} ({interval}) → {path_out}")
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end or datetime.today().strftime("%Y-%m-%d"),
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    close_df = pd.DataFrame()
    missing = []
    for t in tickers:
        if isinstance(data.columns, pd.MultiIndex) and (t, "Close") in data.columns:
            close_df[t] = data[t]["Close"]
        elif "Close" in data.columns and len(tickers) == 1:
            close_df[t] = data["Close"]
        else:
            missing.append(t)

    if missing:
        print("⚠️ Tickeurs introuvables (colonne 'Close' absente) :", ", ".join(missing))

    close_df = close_df.reindex(columns=tickers)  # ordre = input
    close_df = close_df.dropna(how="all")
    close_df.index.name = "Date"
    close_df.to_csv(path_out)
    print(f"✅ Sauvegardé → {path_out} | shape={close_df.shape}")

    return close_df
