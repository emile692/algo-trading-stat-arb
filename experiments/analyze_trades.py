# ============================================================
# experiments/analyze_trades.py
# ============================================================
"""
Analyse détaillée des trades Stat-Arb
-------------------------------------
1️⃣ Charge le fichier de trades (Static, WF ou Mixte)
2️⃣ Applique les pondérations du portefeuille final
3️⃣ Calcule les statistiques globales et par paire
4️⃣ Génère plusieurs visualisations
5️⃣ Exporte un résumé CSV complet et un log détaillé
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import numpy as np

# ============================================================
# ⚙️ CONFIGURATION GLOBALE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
TRADES_DIR = RESULTS_DIR / "trades"
CONFIG_PATH = PROJECT_ROOT / "config" / "params.yaml"

# ============================================================
# 🧠 FONCTIONS UTILITAIRES
# ============================================================

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_trades(mode: str) -> pd.DataFrame:
    trade_path = TRADES_DIR / f"trades_{mode}.csv"
    if not trade_path.exists():
        raise FileNotFoundError(f"❌ Fichier introuvable : {trade_path}")
    trades = pd.read_csv(trade_path, parse_dates=["EntryTime", "ExitTime"])
    print(f"🔍 {len(trades)} trades chargés depuis {trade_path}")

    print(
        f"📊 Échelle des PnL - Min: {trades['PnL_Total'].min():.2f} €, "
        f"Max: {trades['PnL_Total'].max():.2f} €, "
        f"Moyenne: {trades['PnL_Total'].mean():.2f} €"
    )
    return trades

def load_final_portfolio(mode: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"final_portfolio_{mode}.csv"
    return pd.read_csv(path) if path.exists() else None

def apply_weights(trades: pd.DataFrame, final_portfolio: pd.DataFrame | None, initial_capital: float) -> pd.DataFrame:
    if final_portfolio is not None:
        weights = dict(zip(final_portfolio["Pair"], final_portfolio["weight"]))
        trades = trades[trades["Pair"].isin(weights.keys())].copy()
        trades["weight"] = trades["Pair"].map(weights).fillna(0)
        print(f"🔎 Analyse restreinte aux {len(weights)} paires du portefeuille final.")
    else:
        trades["weight"] = 1.0 / trades["Pair"].nunique()
        print("⚠️ Aucun portefeuille final trouvé → pondération égale appliquée.")
    trades["PnL_adj"] = trades["PnL_Total"] * trades["weight"]
    return trades

def compute_global_stats(trades: pd.DataFrame, initial_capital: float, n_pairs: int) -> dict:
    total_pnl = trades["PnL_adj"].sum()
    return {
        "n_trades": len(trades),
        "total_pnl": total_pnl,
        "mean_pnl": trades["PnL_adj"].mean(),
        "win_rate": (trades["PnL_adj"] > 0).mean() * 100,
        "avg_duration": trades.get("Duration_h", pd.Series()).mean(),
        "max_pnl": trades["PnL_adj"].max(),
        "min_pnl": trades["PnL_adj"].min(),
        "capital_return_pct": total_pnl / initial_capital * 100,
    }

def plot_trade_distributions(trades: pd.DataFrame, mode: str):
    plt.figure(figsize=(8, 4))
    sns.histplot(trades["PnL_adj"], bins=60, kde=True, color="steelblue")
    plt.title(f"Distribution des PnL pondérés ({mode})")
    plt.xlabel("PnL (€)")
    plt.tight_layout()
    plt.show()

    pnl_by_pair = trades.groupby("Pair")["PnL_adj"].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    pnl_by_pair.head(20).plot(kind="bar", color="mediumseagreen")
    plt.title(f"Top 20 paires par PnL pondéré ({mode})")
    plt.ylabel("PnL cumulé (€)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def export_summary(trades: pd.DataFrame, mode: str):
    summary = trades.groupby("Pair").agg({
        "PnL_adj": ["sum", "mean", "count"],
        "weight": "first"
    })
    summary.columns = ["PnL_total", "PnL_moyen", "Nb_trades", "Poids"]
    summary["Win_rate"] = trades.groupby("Pair")["PnL_adj"].apply(lambda s: (s > 0).mean() * 100)
    summary = summary.sort_values("PnL_total", ascending=False)
    path = RESULTS_DIR / f"trades_summary_{mode}_pondéré.csv"
    summary.to_csv(path)
    print(f"\n💾 Résumé sauvegardé → {path}")
    return summary

# ============================================================
# 🧾 JOURNAL DÉTAILLÉ ET DÉCOMPOSÉ PAR JAMBE
# ============================================================

def export_trade_log(trades: pd.DataFrame, mode: str):
    """Affiche un journal complet et hiérarchisé (spread + détails par jambe)."""

    required_cols = [
        "Pair", "Side_X", "Side_Y",
        "EntryTime", "ExitTime",
        "EntryPrice_X", "EntryPrice_Y",
        "ExitPrice_X", "ExitPrice_Y",
        "Volume_X", "Volume_Y",
        "PnL_Total", "Duration_h"
    ]
    log_df = trades[[c for c in required_cols if c in trades.columns]].copy()
    log_df = log_df.sort_values("EntryTime").reset_index(drop=True)

    log_df["Side"] = np.where(
        log_df["Side_Y"].str.lower() == "long", "BUY spread", "SELL spread"
    )

    print("\n=== 🧾 Journal détaillé des trades (avec décomposition) ===")

    lines = []
    for _, r in log_df.iterrows():
        entry = r["EntryTime"].strftime("%Y-%m-%d")
        exit_ = r["ExitTime"].strftime("%Y-%m-%d") if pd.notna(r["ExitTime"]) else "OPEN"
        header = (
            f"📅 {entry} | {r['Side']:<4} {r['Pair']:<12} "
            f"| PnL={r['PnL_Total']:>8.2f} € | Durée={r['Duration_h']:>6.0f}h"
        )
        print(header)
        lines.append(header)

        # Détail jambe X
        asset_x, asset_y = r["Pair"].split("-")
        print(
            f"    ├── {asset_x:<5}: {r['Side_X'].upper():<6} "
            f"{r['EntryPrice_X']:.2f} → {r['ExitPrice_X']:.2f} | Vol={r['Volume_X']:.2f}"
        )
        print(
            f"    └── {asset_y:<5}: {r['Side_Y'].upper():<6} "
            f"{r['EntryPrice_Y']:.2f} → {r['ExitPrice_Y']:.2f} | Vol={r['Volume_Y']:.2f}"
        )

        lines.append(f"    ├── {asset_x:<5}: {r['Side_X'].upper():<6} "
                     f"{r['EntryPrice_X']:.2f} → {r['ExitPrice_X']:.2f} | Vol={r['Volume_X']:.2f}")
        lines.append(f"    └── {asset_y:<5}: {r['Side_Y'].upper():<6} "
                     f"{r['EntryPrice_Y']:.2f} → {r['ExitPrice_Y']:.2f} | Vol={r['Volume_Y']:.2f}")

    # Sauvegarde CSV à deux niveaux (spread + jambes)
    detailed_rows = []
    for _, r in log_df.iterrows():
        asset_x, asset_y = r["Pair"].split("-")
        detailed_rows.extend([
            {
                "Pair": r["Pair"],
                "Asset": asset_x,
                "Side": r["Side_X"],
                "EntryPrice": r["EntryPrice_X"],
                "ExitPrice": r["ExitPrice_X"],
                "Volume": r["Volume_X"],
                "PnL_Total_trade": r["PnL_Total"],
                "EntryTime": r["EntryTime"],
                "ExitTime": r["ExitTime"],
                "Duration_h": r["Duration_h"]
            },
            {
                "Pair": r["Pair"],
                "Asset": asset_y,
                "Side": r["Side_Y"],
                "EntryPrice": r["EntryPrice_Y"],
                "ExitPrice": r["ExitPrice_Y"],
                "Volume": r["Volume_Y"],
                "PnL_Total_trade": r["PnL_Total"],
                "EntryTime": r["EntryTime"],
                "ExitTime": r["ExitTime"],
                "Duration_h": r["Duration_h"]
            }
        ])

    log_path = RESULTS_DIR / f"trade_log_{mode}_hierarchique.csv"
    pd.DataFrame(detailed_rows).to_csv(log_path, index=False)
    print(f"\n💾 Log complet (spread + jambes) sauvegardé → {log_path}")


# ============================================================
# 🚀 MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyse des trades Stat-Arb.")
    parser.add_argument("--mode", choices=["Static", "WF", "Mixte"], default="WF")
    parser.add_argument("--capital", type=float, default=None)
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    initial_capital = args.capital or config.get("initial_capital", 10_000)
    mode = args.mode

    print(f"⚙️  Capital initial total : {initial_capital:,.0f} €")

    trades = load_trades(mode)
    final_portfolio = load_final_portfolio(mode)
    trades = apply_weights(trades, final_portfolio, initial_capital)
    n_pairs = len(final_portfolio) if final_portfolio is not None else trades["Pair"].nunique()

    stats = compute_global_stats(trades, initial_capital, n_pairs)
    print(f"\n=== 📊 Statistiques globales ({mode}) ===")
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            print(f"{k.replace('_',' ').capitalize():<25}: {v:,.2f}")
        else:
            print(f"{k.replace('_',' ').capitalize():<25}: {v}")

    pnl_by_pair = trades.groupby("Pair")["PnL_adj"].sum().sort_values(ascending=False)
    print("\n=== 🧩 PnL par paire (pondéré) ===")
    print(pnl_by_pair.head(15))

    plot_trade_distributions(trades, mode)
    export_trade_log(trades, mode)
    export_summary(trades, mode)

# ============================================================
if __name__ == "__main__":
    main()
