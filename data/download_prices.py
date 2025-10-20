# ============================================================
# data/download_prices.py
# ============================================================
import yfinance as yf
import pandas as pd
from datetime import datetime

def download_prices(tickers, start="2020-01-01", end=None, interval="1h", path_out="prices.csv"):
    """
    Télécharge les prix ajustés pour une liste de tickers Yahoo Finance
    et sauvegarde un CSV aligné sur un même index temporel.
    """
    print(f"⏳ Téléchargement {len(tickers)} tickers de {start} à {end or 'today'} ({interval})")
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end or datetime.today().strftime("%Y-%m-%d"),
        interval=interval,
        group_by='ticker',
        auto_adjust=True,
        progress=True,
        threads=True
    )

    # Harmoniser en un DataFrame plat
    close_df = pd.DataFrame()
    for t in tickers:
        if (t, 'Close') in data.columns:
            close_df[t] = data[t]['Close']
        elif 'Close' in data.columns:  # si single ticker
            close_df[t] = data['Close']
        else:
            print(f"⚠️ {t}: colonne 'Close' introuvable.")

    close_df = close_df.dropna(how="all")
    close_df.index.name = "Date"
    close_df.to_csv(path_out)
    print(f"✅ Sauvegardé → {path_out} | shape={close_df.shape}")

    return close_df

if __name__ == "__main__":

    tickers = [
        "GS", "JPM", "BK", "WFC", "MS", "C", "USB",  # banques
        "XOM", "CVX", "BP",                          # énergie
        "AAPL", "MSFT", "GOOGL", "AMZN"              # tech (pour tests cross-sector)
    ]
    df = download_prices(tickers, start="2024-01-01", interval="1h", path_out="prices.csv")
    print(df.tail())
