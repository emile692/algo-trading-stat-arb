import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import skew, kurtosis
import yaml
from utils.time_scaling import get_bars_per_year

# === ⚙️ CONFIGURATION ===
with open("config/params.yaml", "r") as f:
    params = yaml.safe_load(f)

FREQ = params.get("freq", "1H")
BARS_PER_YEAR = get_bars_per_year(FREQ)
INITIAL_CAPITAL = 10_000  # 💰 même capital de référence

RESULTS_DIR = Path("results")
FILES = {
    "Static": RESULTS_DIR / "portfolio_equity_static.csv",
    "WF": RESULTS_DIR / "portfolio_equity_wf.csv",
    "Mixte": RESULTS_DIR / "portfolio_equity_mixte.csv",
}


def compute_metrics(df, col, initial_capital=INITIAL_CAPITAL):
    equity = df[col].astype(float)
    equity = equity.loc[equity.ne(0).idxmax():]
    equity = equity[equity > 0]

    # NAV = capital initial + PnL pondéré
    equity_start = equity.iloc[0]
    nav = initial_capital + (equity - equity_start)

    # Retours = ΔPnL / capital initial
    pnl_diff = equity.diff().replace([np.inf, -np.inf], np.nan).dropna()
    returns = (pnl_diff / initial_capital).dropna()
    if len(returns) < 3 or returns.std() == 0:
        return None

    mean_ret = returns.mean()
    vol = returns.std()
    sharpe = np.sqrt(BARS_PER_YEAR) * mean_ret / vol
    downside = returns[returns < 0]
    sortino = np.sqrt(BARS_PER_YEAR) * mean_ret / downside.std() if downside.std() > 0 else np.nan

    # Valeur finale
    final_value = nav.iloc[-1]
    cash_gain = final_value - initial_capital
    total_return_pct = (final_value / initial_capital - 1) * 100

    # Drawdown et CAGR sur NAV
    nav_norm = nav / nav.iloc[0]
    max_dd = (nav_norm / nav_norm.cummax() - 1).min()
    days = (nav.index[-1] - nav.index[0]).days
    CAGR = (nav_norm.iloc[-1]) ** (252 / days) - 1 if days > 0 else np.nan

    # Risque et distribution
    VaR_95 = returns.quantile(0.05)
    best_week = returns.resample("W").sum().max()
    worst_week = returns.resample("W").sum().min()
    skewness = skew(returns)
    kurt = kurtosis(returns)

    dd = (nav_norm / nav_norm.cummax() - 1)
    recovery_period = 0
    if dd.min() < 0:
        last_max = dd[dd == 0].index[-1] if not dd[dd == 0].empty else dd.index[0]
        recovery_period = (dd.index[-1] - last_max).days

    pos_weeks = (returns.resample("W").sum() > 0).astype(int)
    neg_weeks = (returns.resample("W").sum() < 0).astype(int)
    max_consec_pos = pos_weeks.groupby((pos_weeks != pos_weeks.shift()).cumsum()).transform("size").max()
    max_consec_neg = neg_weeks.groupby((neg_weeks != neg_weeks.shift()).cumsum()).transform("size").max()

    return {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "CAGR": CAGR,
        "Volatility (ann.)": np.sqrt(BARS_PER_YEAR) * vol,
        "Portfolio Final Value (€)": final_value,
        "Portfolio Cash Gain (€)": cash_gain,
        "Portfolio Relative Return (%)": total_return_pct,
        "Max Drawdown": max_dd,
        "VaR 95%": VaR_95,
        "Best Weekly Return": best_week,
        "Worst Weekly Drawdown": worst_week,
        "Skewness": skewness,
        "Kurtosis": kurt,
        "Max Consecutive Pos Weeks": max_consec_pos,
        "Max Consecutive Neg Weeks": max_consec_neg,
        "Recovery Period (days)": recovery_period,
    }


metrics = {}
plt.figure(figsize=(12, 5))

for name, path in FILES.items():
    if not path.exists():
        print(f"⚠️ Fichier manquant : {path}")
        continue

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df.rename(columns={"Portfolio": "equity"}, inplace=True)
    df["equity"] = df["equity"].astype(float)

    if df["equity"].max() == 0:
        print(f"⚠️ {name}: equity vide (tout à 0)")
        continue

    df = df.loc[df["equity"].ne(0).idxmax():]
    df["equity_norm"] = df["equity"] / df["equity"].iloc[0]
    plt.plot(df.index, df["equity_norm"], label=name)

    result = compute_metrics(df, "equity")
    if result:
        metrics[name] = result
    else:
        print(f"⚠️ {name}: données insuffisantes pour calculer les métriques")

plt.title("Comparaison des portefeuilles Static vs WF vs Mixte (base 10 000 €)")
plt.xlabel("Temps")
plt.ylabel("Équity normalisée")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "compare_portfolios.png")
plt.show()

metrics_df = pd.DataFrame(metrics).T.round(4)
print("\n=== 📈 Résumé des portefeuilles ===")
print(metrics_df.T.round(2))
metrics_df.to_csv(RESULTS_DIR / "compare_portfolios_summary.csv")
print(f"💾 Sauvegardé → {RESULTS_DIR}/compare_portfolios_summary.csv")
