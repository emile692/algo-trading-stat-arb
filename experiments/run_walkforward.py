# ============================================================
# experiments/run_walkforward.py — version avec sauvegarde equity réelle
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt
from core.config import Settings
from core.loader import load_prices, get_pairs
from core.hedge_ratio import fit_beta_expanding_monthly, fit_beta_static
from core.backtest import backtest_pair
import os

if __name__ == "__main__":
    # 1️⃣ Config du pipeline
    cfg = Settings(
        price_path="data/prices.csv",
        tickers=["GS", "JPM", "BK", "WFC", "MS", "C", "USB", "XOM", "CVX", "BP", "AAPL", "MSFT", "GOOGL", "AMZN"],
        date_col="Date",
        tz=None,
        freq_out="1h",
        price_col_suffix=None,
        log_prices=True,
        start="2024-01-01",
        end=None,
        how="last"
    )

    # 2️⃣ Chargement
    prices = load_prices(cfg)
    print(f"✅ Loaded prices: {prices.shape}")
    pairs = get_pairs(cfg.tickers)
    print(f"✅ {len(pairs)} paires générées")

    results = []
    os.makedirs("results/equities/static", exist_ok=True)
    os.makedirs("results/equities/wf", exist_ok=True)

    # 3️⃣ Backtest sur chaque paire
    for a, b in pairs:
        if a not in prices.columns or b not in prices.columns:
            print(f"⚠️ Skipped {a}-{b} (missing data)")
            continue

        y, x = prices[a].dropna(), prices[b].dropna()
        print(f"→ Backtest {a}-{b} ...", end=" ")

        try:
            # --- 1️⃣ Statique ---
            beta_static = fit_beta_static(y, x)
            beta_series_static = pd.Series(beta_static, index=y.index)
            res_static = backtest_pair(y, x, beta_series_static, window=24 * 45)
            res_static.update({"pair": f"{a}-{b}", "mode": "Static"})

            # Sauvegarde equity réelle
            res_static["equity"].to_csv(f"results/equities/static/{a}-{b}.csv")

            # --- 2️⃣ Walk-Forward expanding ---
            beta_series_wf = fit_beta_expanding_monthly(y, x, min_hist=200)
            res_wf = backtest_pair(y, x, beta_series_wf, window=24 * 45)
            res_wf.update({"pair": f"{a}-{b}", "mode": "WF"})

            # Sauvegarde equity réelle
            res_wf["equity"].to_csv(f"results/equities/wf/{a}-{b}.csv")

            # --- append both ---
            results.extend([res_static, res_wf])
            print(f"Done | S={res_static['sharpe']:.2f} / WF={res_wf['sharpe']:.2f}")

        except Exception as e:
            print(f"❌ Error {a}-{b}: {e}")
            continue

    # 4️⃣ Résumé des performances
    if not results:
        print("❌ Aucun résultat produit.")
        exit()

    df_perf = pd.DataFrame([{
        "Pair": r["pair"],
        "Mode": r["mode"],
        "Sharpe": r["sharpe"],
        "MaxDD": r["max_dd"],
        "FinalPnL": float(r["equity"].iloc[-1]),
        "Trades": r["trades"]
    } for r in results])

    df_pivot = df_perf.pivot(index="Pair", columns="Mode", values=["Sharpe", "MaxDD", "FinalPnL"])
    df_pivot.columns = [f"{metric}_{mode}" for metric, mode in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    df_pivot["ΔSharpe"] = df_pivot["Sharpe_WF"] - df_pivot["Sharpe_Static"]

    print("\n=== Comparaison Static vs Walk-Forward ===")
    print(df_pivot.round(3))

    # 5️⃣ Sauvegarde
    os.makedirs("results", exist_ok=True)
    df_perf.to_csv("results/walkforward_summary.csv", index=False)
    df_pivot.to_csv("results/comparison_static_vs_wf.csv", index=False)
    print("💾 Résultats sauvegardés → results/walkforward_summary.csv")
