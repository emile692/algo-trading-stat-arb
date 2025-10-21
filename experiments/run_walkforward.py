# ============================================================
# experiments/run_walkforward.py — univers-aware version + trade logger
# ============================================================
import os
from pathlib import Path
import pandas as pd

from core.config import Settings
from core.loader import load_prices, get_pairs
from core.hedge_ratio import fit_beta_expanding_monthly, fit_beta_static
from core.backtest import backtest_pair

# === Nouveau module pour log des exécutions ===
from datetime import datetime
from src.trading.execution_logger import ExecutionLogger

if __name__ == "__main__":
    # === 1️⃣ Lecture des variables d'environnement ===
    path_data = os.getenv("UNIVERSE_FILE", "data/prices.csv")
    universe_name = os.getenv("UNIVERSE_NAME", "UNSPECIFIED")
    assets_env = os.getenv("UNIVERSE_ASSETS", "")
    tickers = [t for t in assets_env.split(",") if t]

    if not tickers:
        tickers = [
            "GS", "JPM", "BK", "WFC", "MS", "C", "USB",
            "XOM", "CVX", "BP", "AAPL", "MSFT", "GOOGL", "AMZN"
        ]
        print("⚠️ Aucun UNIVERSE_ASSETS fourni, fallback vers tickers par défaut.")

    # === 2️⃣ Configuration du pipeline ===
    cfg = Settings(
        price_path=path_data,
        tickers=tickers,
        date_col="Date",
        tz=None,
        freq_out="1h",
        price_col_suffix=None,
        log_prices=True,
        start="2024-01-01",
        end=None,
        how="last"
    )

    print(f"✅ Chargement du dataset ({universe_name}) : {cfg.price_path}")
    prices = load_prices(cfg)
    prices = prices[[t for t in tickers if t in prices.columns]]

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().ffill().bfill()

    print(f"✅ Loaded prices: {prices.shape}")

    pairs = get_pairs(prices.columns)
    print(f"✅ {len(pairs)} paires générées pour {len(prices.columns)} actifs")

    results = []
    os.makedirs("results/equities/static", exist_ok=True)
    os.makedirs("results/equities/wf", exist_ok=True)

    # === 3️⃣ Initialisation du logger global pour la stratégie ===
    trade_logger_static = ExecutionLogger("Static", log_dir="results/trades")
    trade_logger_wf = ExecutionLogger("WF", log_dir="results/trades")

    # === 4️⃣ Backtest par paire ===
    for a, b in pairs:
        if a not in prices.columns or b not in prices.columns:
            print(f"⚠️ Skipped {a}-{b} (missing data)")
            continue

        y = pd.to_numeric(prices[a], errors="coerce").dropna()
        x = pd.to_numeric(prices[b], errors="coerce").dropna()
        y, x = y.align(x, join="inner")

        if len(y) < 200:
            print(f"⚠️ Skipped {a}-{b} (too few data points)")
            continue

        print(f"→ Backtest {a}-{b} ...", end=" ")

        try:
            # --- Statique ---
            beta_static = fit_beta_static(y, x)
            beta_series_static = pd.Series(beta_static, index=y.index)
            res_static = backtest_pair(y, x, beta_series_static, window=24 * 45)

            # 🔹 Si le backtest renvoie les trades individuels
            if "trades_df" in res_static:
                for _, t in res_static["trades_df"].iterrows():
                    trade_logger_static.log_trade(
                        pair=f"{a}-{b}",
                        side_x=t.get("side_x", "long"),
                        side_y=t.get("side_y", "short"),
                        vol_x=t.get("volume_x", 1),
                        vol_y=t.get("volume_y", 1),
                        entry_time=pd.to_datetime(t["entry_time"]),
                        exit_time=pd.to_datetime(t["exit_time"]),
                        entry_px_x=t.get("entry_px_x", y.loc[t["entry_time"]] if t["entry_time"] in y.index else None),
                        entry_px_y=t.get("entry_px_y", x.loc[t["entry_time"]] if t["entry_time"] in x.index else None),
                        exit_px_x=t.get("exit_px_x", y.loc[t["exit_time"]] if t["exit_time"] in y.index else None),
                        exit_px_y=t.get("exit_px_y", x.loc[t["exit_time"]] if t["exit_time"] in x.index else None),
                        pnl_x=t.get("pnl_x", 0),
                        pnl_y=t.get("pnl_y", 0),
                        reason=t.get("reason", "signal")
                    )

            res_static.update({"pair": f"{a}-{b}", "mode": "Static"})
            res_static["equity"].to_csv(f"results/equities/static/{a}-{b}.csv")

            # --- Walk-forward ---
            beta_series_wf = fit_beta_expanding_monthly(y, x, min_hist=200)
            res_wf = backtest_pair(y, x, beta_series_wf, window=24 * 45)

            # 🔹 Log complet des trades WF
            if "trades_df" in res_wf:
                for _, t in res_wf["trades_df"].iterrows():
                    trade_logger_wf.log_trade(
                        pair=f"{a}-{b}",
                        side_x=t.get("side_x", "long"),
                        side_y=t.get("side_y", "short"),
                        vol_x=t.get("volume_x", 1),
                        vol_y=t.get("volume_y", 1),
                        entry_time=pd.to_datetime(t["entry_time"]),
                        exit_time=pd.to_datetime(t["exit_time"]),
                        entry_px_x=t.get("entry_px_x", y.loc[t["entry_time"]] if t["entry_time"] in y.index else None),
                        entry_px_y=t.get("entry_px_y", x.loc[t["entry_time"]] if t["entry_time"] in x.index else None),
                        exit_px_x=t.get("exit_px_x", y.loc[t["exit_time"]] if t["exit_time"] in y.index else None),
                        exit_px_y=t.get("exit_px_y", x.loc[t["exit_time"]] if t["exit_time"] in x.index else None),
                        pnl_x=t.get("pnl_x", 0),
                        pnl_y=t.get("pnl_y", 0),
                        reason=t.get("reason", "signal")
                    )

            res_wf.update({"pair": f"{a}-{b}", "mode": "WF"})
            res_wf["equity"].to_csv(f"results/equities/wf/{a}-{b}.csv")

            results.extend([res_static, res_wf])
            print(f"Done | S={res_static['sharpe']:.2f} / WF={res_wf['sharpe']:.2f}")

        except Exception as e:
            print(f"❌ Error {a}-{b}: {e}")
            continue

    # === 5️⃣ Résumé et sauvegarde ===
    if not results:
        print("❌ Aucun résultat produit.")
        raise SystemExit(1)

    df_perf = pd.DataFrame([{
        "Pair": r["pair"],
        "Mode": r["mode"],
        "Sharpe": r["sharpe"],
        "MaxDD": r["max_dd"],
        "FinalPnL": float(r["equity"].iloc[-1]),
        "Trades": r["trades"],
        "Universe": universe_name
    } for r in results])

    df_pivot = df_perf.pivot(index="Pair", columns="Mode", values=["Sharpe", "MaxDD", "FinalPnL"])
    df_pivot.columns = [f"{metric}_{mode}" for metric, mode in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    if "Sharpe_Static" in df_pivot and "Sharpe_WF" in df_pivot:
        df_pivot["ΔSharpe"] = df_pivot["Sharpe_WF"] - df_pivot["Sharpe_Static"]

    print("\n=== Comparaison Static vs Walk-Forward ===")
    print(df_pivot.round(3))

    os.makedirs("results", exist_ok=True)
    summary_path = Path("results/walkforward_summary.csv")

    if summary_path.exists():
        df_existing = pd.read_csv(summary_path)
        df_perf = pd.concat([df_existing, df_perf], ignore_index=True)
        print(f"🧩 Ajout des résultats pour {universe_name} ({len(df_perf)} lignes totales)")

    df_perf.to_csv(summary_path, index=False)
    df_pivot.to_csv(f"results/comparison_static_vs_wf_{universe_name}.csv", index=False)
    print(f"💾 Résultats sauvegardés → results/walkforward_summary.csv (+ {universe_name})")
