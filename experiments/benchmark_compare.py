# ============================================================
# experiments/benchmark_compare.py
# Compare la stratégie market-neutral au S&P 500
# ============================================================
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os


def compare_neutral_strategy(strategy_equity, start=None, end=None):
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt
    import os

    # === Déterminer la période automatique ===
    if start is None:
        start = strategy_equity.index.min().strftime("%Y-%m-%d")
    if end is None:
        end = strategy_equity.index.max().strftime("%Y-%m-%d")

    # === Télécharger le benchmark (S&P500) ===
    spy = yf.download("^GSPC", start="2021-01-01", end=end, progress=False)["Close"]
    spy.index = pd.to_datetime(spy.index)

    # === Harmonisation de l'index ===
    strategy_equity.index = pd.to_datetime(strategy_equity.index)
    try:
        if strategy_equity.index.tz is None:
            strategy_equity.index = strategy_equity.index.tz_localize("UTC")
        else:
            strategy_equity.index = strategy_equity.index.tz_convert("UTC")
    except Exception:
        pass

    spy.index = spy.index.tz_localize("UTC", nonexistent="shift_forward")

    # === Resampling daily & alignement ===
    strategy_equity = strategy_equity.resample("1D").last().dropna()
    spy = spy.resample("1D").last().dropna()

    # === Détection automatique du burn-in ===
    diff = strategy_equity.diff().fillna(0)
    first_move_idx = diff.ne(0).idxmax()  # première date où diff != 0
    burn_in_date = first_move_idx if pd.notna(first_move_idx) else strategy_equity.index[0]
    print(f"⏱ Burn-in détecté jusqu’au {burn_in_date.date()}")

    # Tronquer la stratégie et le SP500 à partir de cette date
    strategy_equity = strategy_equity.loc[burn_in_date:]
    spy = spy.loc[burn_in_date:]

    # === Calcul des rendements ===
    strat_ret = strategy_equity.pct_change().dropna().squeeze()
    spy_ret = spy.pct_change().dropna().squeeze()

    # Alignement strict
    strat_ret, spy_ret = strat_ret.align(spy_ret, join="inner")
    strategy_equity, spy = strategy_equity.align(spy, join="inner")

    if len(strat_ret) < 5:
        raise ValueError("❌ Trop peu de points communs entre stratégie et SPX.")

    # === Statistiques ===
    excess_ret = strat_ret
    sharpe = np.sqrt(252) * excess_ret.mean() / excess_ret.std()
    vol = excess_ret.std() * np.sqrt(252)
    max_dd = (strategy_equity / strategy_equity.cummax() - 1).min()

    n_years = (strategy_equity.index[-1] - strategy_equity.index[0]).days / 365.25
    cagr = (strategy_equity.iloc[-1] / strategy_equity.iloc[0]) ** (1 / n_years) - 1

    x = spy_ret.values.ravel()
    y = excess_ret.values.ravel()
    beta = np.cov(y, x)[0, 1] / np.var(x)
    corr = np.corrcoef(y, x)[0, 1]

    print("\n=== 📈 Comparaison marché neutre ===")
    print(f"Période               : {burn_in_date.date()} → {end}")
    print(f"CAGR                  : {cagr*100:.2f}%")
    print(f"Volatilité annuelle   : {vol*100:.2f}%")
    print(f"Sharpe ratio          : {sharpe:.2f}")
    print(f"Max Drawdown          : {max_dd*100:.2f}%")
    print(f"Corrélation SPX       : {corr:.3f}")
    print(f"Beta SPX              : {beta:.3f}")

    # === GRAPHIQUE COMPARATIF ===
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    # --- NAV (haut) ---
    ax1.plot(strategy_equity.index, strategy_equity / strategy_equity.iloc[0],
             label="Strategy (neutral)", lw=2)
    ax1.plot(spy.index, spy / spy.iloc[0],
             label="S&P 500 (SPX)", lw=1.8, color="orange")
    ax1.axvline(burn_in_date, color="gray", linestyle="--", lw=1)
    ax1.text(burn_in_date, ax1.get_ylim()[0], " Burn-in end", rotation=90,
             va="bottom", ha="left", color="gray", fontsize=9)
    ax1.set_title("Market-Neutral Strategy vs S&P 500")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Ajouter statistiques sur le graphe
    stats_text = (
        f"CAGR: {cagr*100:.1f}%\n"
        f"Sharpe: {sharpe:.2f}\n"
        f"MaxDD: {max_dd*100:.1f}%\n"
        f"Corr SPX: {corr:.2f}"
    )
    ax1.text(0.02, 0.05, stats_text, transform=ax1.transAxes,
             fontsize=10, bbox=dict(facecolor="white", alpha=0.7))

    # --- DRAWDOWN (bas) ---
    dd = strategy_equity / strategy_equity.cummax() - 1
    ax2.fill_between(dd.index, dd, 0, color="red", alpha=0.3)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.set_ylim(dd.min() * 1.1, 0)
    ax2.set_title(f"Max Drawdown: {max_dd*100:.2f}%")

    plt.tight_layout()

    # --- Sauvegarde automatique ---
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/benchmark_report.png", dpi=300)
    plt.show()




if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RESULTS_ROOT = PROJECT_ROOT / "results"
    FILE_ROOT = RESULTS_ROOT / "portfolio_equity_static.csv"

    df = pd.read_csv(FILE_ROOT, index_col=0, parse_dates=True)
    colname = [c for c in df.columns if "Portfolio" in c][0]
    strategy_equity = df[colname]

    # 🧮 Corrige le cas où equity = PnL cumulatif
    if strategy_equity.min() <= 0 or strategy_equity.iloc[0] < 100:
        strategy_equity = 10_000 + strategy_equity

    compare_neutral_strategy(strategy_equity)
