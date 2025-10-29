# ============================================================
# data/download_prices.py
# ============================================================
import os
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
from pathlib import Path
import yaml

# === Chargement de la config globale ===
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "params.yaml"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {}

DATA_PROVIDER = CONFIG.get("data_provider", "yahoo").lower()
API_KEY = os.getenv(f"{DATA_PROVIDER}_api_key")

# ============================================================

def _need_redownload(path_out: str, expected: list[str]) -> bool:
    p = Path(path_out)
    if not p.exists():
        return True
    try:
        df = pd.read_csv(p, nrows=1)
        cols = [c for c in df.columns if c != "Date"]
        return cols != list(expected)
    except Exception:
        return True

# ============================================================
# === Provider: Yahoo Finance (par défaut)
# ============================================================
def _download_yahoo(tickers, start, end, interval) -> pd.DataFrame:
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
    for t in tickers:
        if isinstance(data.columns, pd.MultiIndex) and (t, "Close") in data.columns:
            close_df[t] = data[t]["Close"]
        elif "Close" in data.columns and len(tickers) == 1:
            close_df[t] = data["Close"]
        else:
            print(f"⚠️ Données manquantes pour {t}")
    return close_df

# ============================================================
# === Provider: Finnhub
# ============================================================
def _download_finnhub(tickers, start, end, interval) -> pd.DataFrame:
    """
    Télécharge via Finnhub.io (requiert API key)
    interval: '1h' ou '1d'
    """
    if not API_KEY:
        raise ValueError("❌ Clé API Finnhub manquante (config: finnhub_api_key).")

    base = "https://finnhub.io/api/v1/stock/candle"
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end or datetime.today()).timestamp())

    res = {}
    for t in tickers:
        params = {
            "symbol": t,
            "resolution": "60" if interval == "1h" else "D",
            "from": start_ts,
            "to": end_ts,
            "token": API_KEY,
        }
        r = requests.get(base, params=params)
        if r.status_code != 200:
            print(f"⚠️ Erreur {r.status_code} pour {t}")
            continue
        data = r.json()
        if data.get("s") != "ok":
            print(f"⚠️ Pas de données pour {t}")
            continue
        df = pd.DataFrame({"Date": pd.to_datetime(data["t"], unit="s"), t: data["c"]})
        df = df.set_index("Date")
        res[t] = df[t]

    if not res:
        raise ValueError("❌ Aucun ticker n’a pu être téléchargé depuis Finnhub.")

    close_df = pd.concat(res, axis=1)
    return close_df

# ============================================================
# === Fonction principale (interface unifiée)
# ============================================================
def download_prices(
    tickers: list[str],
    start: str = "2020-01-01",
    end: str | None = None,
    interval: str = "1h",
    path_out: str = "data/prices.csv",
    force: bool = False,
) -> pd.DataFrame:
    Path(path_out).parent.mkdir(parents=True, exist_ok=True)

    if not force and not _need_redownload(path_out, tickers):
        print(f"📁 {Path(path_out).name} déjà conforme, skip.")
        return pd.read_csv(path_out, parse_dates=["Date"], index_col="Date")

    print(f"⏳ Téléchargement via provider = {DATA_PROVIDER.upper()} ({len(tickers)} tickers)")
    if DATA_PROVIDER == "finnhub":
        close_df = _download_finnhub(tickers, start, end, interval)
    elif DATA_PROVIDER == "yahoo":
        close_df = _download_yahoo(tickers, start, end, interval)
    else:
        raise ValueError(f"❌ Provider inconnu: {DATA_PROVIDER}")

    close_df = close_df.reindex(columns=tickers)
    close_df = close_df.dropna(how="all")
    close_df.index.name = "Date"
    close_df.to_csv(path_out)
    print(f"✅ Sauvegardé → {path_out} | shape={close_df.shape}")

    return close_df
