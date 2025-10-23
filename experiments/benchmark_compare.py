import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def compare_neutral_strategy(strategy_equity, start=None, end=None):
    # Benchmark = SPX pour corrélation, cash = 0% perf
    spy = yf.download("^GSPC", start=start, end=end)["Close"]

    # Uniformisation des timezones
    spy.index = pd.to_datetime(spy.index).tz_localize("UTC")
    strategy_equity.index = pd.to_datetime(strategy_equity.index).tz_convert("UTC")

    # Harmonisation de la fréquence (daily close)
    strategy_equity = strategy_equity.resample("1D").last().dropna()

    # === Calcul des rendements ===
    strat_ret = strategy_equity.pct_change().dropna()
    spy_ret = spy.pct_change().dropna()

    # Alignement des deux séries sur les dates communes
    strat_ret, spy_ret = strat_ret.align(spy_ret, join="inner")
    if len(strat_ret) == 0:
        raise ValueError("❌ Aucune date commune entre ta stratégie et le benchmark (SPX).")

    # === Metrics ===
    excess_ret = strat_ret  # benchmark cash = 0
    sharpe = np.sqrt(252) * excess_ret.mean() / excess_ret.std()
    cagr = (strategy_equity.iloc[-1] / strategy_equity.iloc[0]) ** (252 / len(strategy_equity)) - 1
    vol = excess_ret.std() * np.sqrt(252)
    max_dd = (strategy_equity / strategy_equity.cummax() - 1).min()

    beta = np.cov(excess_ret, spy_ret)[0, 1] / np.var(spy_ret)
    corr = excess_ret.corr(spy_ret)

    print("\n=== 📈 Comparaison marché neutre ===")
    print(f"CAGR                 : {cagr*100:.2f}%")
    print(f"Volatilité annuelle  : {vol*100:.2f}%")
    print(f"Sharpe ratio         : {sharpe:.2f}")
    print(f"Max Drawdown         : {max_dd*100:.2f}%")
    print(f"Corrélation SPX      : {corr:.3f}")
    print(f"Beta SPX             : {beta:.3f}")

    plt.figure(figsize=(10, 5))
    plt.plot(strategy_equity.index, strategy_equity / strategy_equity.iloc[0], label="Strategy (neutral)")
    plt.plot(spy.index, spy / spy.iloc[0], label="S&P 500 (SPX)")
    plt.legend()
    plt.title("Market-Neutral Strategy vs S&P 500")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    strategy_equity = pd.read_csv(
        "results/portfolio_equity_static.csv", index_col=0, parse_dates=True
    )["Portfolio"]
    compare_neutral_strategy(strategy_equity)
