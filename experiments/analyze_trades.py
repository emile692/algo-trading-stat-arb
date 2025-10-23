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
5️⃣ Exporte un résumé CSV complet
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
    """Charge la configuration YAML du projet."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_trades(mode: str) -> pd.DataFrame:
    """Charge le fichier de trades correspondant au mode."""
    trade_path = TRADES_DIR / f"trades_{mode}.csv"
    if not trade_path.exists():
        raise FileNotFoundError(f"❌ Fichier introuvable : {trade_path}")

    trades = pd.read_csv(trade_path, parse_dates=["EntryTime", "ExitTime"])
    print(f"🔍 {len(trades)} trades chargés depuis {trade_path}")

    # Après avoir chargé les trades
    print(
        f"📊 Échelle des PnL - Min: {trades['PnL_Total'].min():.2f} €, Max: {trades['PnL_Total'].max():.2f} €, Moyenne: {trades['PnL_Total'].mean():.2f} €")

    # === Vérification du PnL latent enregistré ===
    summary_path = RESULTS_DIR / "walkforward_summary.csv"
    if summary_path.exists():
        df_summary = pd.read_csv(summary_path)
        if {"Pair", "PnL_Latent", "Mode"}.issubset(df_summary.columns):
            latent = df_summary[df_summary["Mode"] == mode][["Pair", "PnL_Latent"]]
            latent = latent.set_index("Pair")

            # 🔧 Normalisation du latent au capital global
            # Chaque paire a été testée avec 10k€, mais le portefeuille global a 10k€ total
            if not latent.empty:
                n_pairs = len(latent)
                latent["PnL_Latent"] = latent["PnL_Latent"] / n_pairs

            total_latent = latent["PnL_Latent"].sum()

            print(f"💤 PnL latent total ({mode}) : {total_latent:,.2f} €")
            if total_latent != 0:
                print(latent[latent["PnL_Latent"].abs() > 0]
                      .sort_values("PnL_Latent", ascending=False))

    return trades


def load_final_portfolio(mode: str) -> pd.DataFrame | None:
    """Charge le portefeuille final si disponible."""
    final_portfolio_path = RESULTS_DIR / f"final_portfolio_{mode}.csv"
    if final_portfolio_path.exists():
        return pd.read_csv(final_portfolio_path)
    return None


def apply_weights(trades: pd.DataFrame, final_portfolio: pd.DataFrame | None, initial_capital: float) -> pd.DataFrame:
    """Applique les pondérations du portefeuille final aux trades (PnL déjà en euros)."""
    if final_portfolio is not None:
        selected_pairs = final_portfolio["Pair"].tolist()
        weights = dict(zip(final_portfolio["Pair"], final_portfolio["weight"]))
        trades = trades[trades["Pair"].isin(selected_pairs)].copy()
        trades["weight"] = trades["Pair"].map(weights).fillna(0)
        print(f"🔎 Analyse restreinte aux {len(selected_pairs)} paires du portefeuille final.")
    else:
        trades["weight"] = 1.0 / trades["Pair"].nunique()
        print("⚠️ Aucun portefeuille final trouvé → pondération égale appliquée.")

    # CORRECTION : Supprimer la multiplication par initial_capital
    trades["PnL_Total"] = trades["PnL_Total"].astype(float)
    trades["Duration_h"] = trades["Duration_h"].astype(float)

    # PnL_adj = PnL_Total * poids (sans multiplication par capital)
    trades["PnL_adj"] = trades["PnL_Total"] * trades["weight"]

    return trades


def compute_global_stats(trades: pd.DataFrame, initial_capital: float, n_pairs: int) -> dict:
    """Calcule les statistiques globales du portefeuille."""
    total_pnl = trades["PnL_adj"].sum()

    # CORRECTION : Le capital total investi = capital_initial * nombre_de_paires
    portfolio_capital = initial_capital * n_pairs

    stats = {
        "n_trades": len(trades),
        "total_pnl": total_pnl,
        "mean_pnl": trades["PnL_adj"].mean(),
        "win_rate": (trades["PnL_adj"] > 0).mean() * 100,
        "avg_duration": trades["Duration_h"].mean(),
        "max_pnl": trades["PnL_adj"].max(),
        "min_pnl": trades["PnL_adj"].min(),
        "capital_return_pct": total_pnl / initial_capital * 100,
    }
    return stats


def plot_trade_distributions(trades: pd.DataFrame, mode: str):
    """Affiche les principales visualisations de performance."""
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

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="Duration_h", y="PnL_adj", data=trades, alpha=0.7)
    plt.title(f"Durée vs PnL pondéré ({mode})")
    plt.xlabel("Durée (heures)")
    plt.ylabel("PnL (€)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def export_summary(trades: pd.DataFrame, mode: str):
    """Génère et exporte le résumé des performances par paire."""
    summary = trades.groupby("Pair").agg({
        "PnL_adj": ["sum", "mean", "count"],
        "weight": "first"
    })
    summary.columns = ["PnL_total", "PnL_moyen", "Nb_trades", "Poids"]
    summary["Win_rate"] = trades.groupby("Pair")["PnL_adj"].apply(lambda s: (s > 0).mean() * 100)
    summary = summary.sort_values("PnL_total", ascending=False)

    summary_path = RESULTS_DIR / f"trades_summary_{mode}_pondéré.csv"
    summary.to_csv(summary_path)
    print(f"\n💾 Résumé sauvegardé → {summary_path}")
    return summary


# ============================================================
# 🚀 MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyse des trades Stat-Arb.")
    parser.add_argument("--mode", choices=["Static", "WF", "Mixte"], default="WF",
                        help="Mode d'analyse : Static, WF ou Mixte (défaut: WF)")
    parser.add_argument("--capital", type=float, default=None,
                        help="Capital initial à utiliser (par défaut: valeur de config/params.yaml)")
    args = parser.parse_args()

    # --- Chargement des paramètres globaux ---
    config = load_config(CONFIG_PATH)
    initial_capital = args.capital or config.get("initial_capital", 10_000)
    mode = args.mode

    print(f"⚙️  Capital initial total : {initial_capital:,.0f} €")

    # --- Pipeline principal ---
    trades = load_trades(mode)
    final_portfolio = load_final_portfolio(mode)
    trades = apply_weights(trades, final_portfolio, initial_capital)
    n_pairs = len(final_portfolio) if final_portfolio is not None else trades["Pair"].nunique()
    stats = compute_global_stats(trades, initial_capital, n_pairs)

    # --- Résumé global ---
    print(f"\n=== 📊 Statistiques globales ({mode}) ===")
    for k, v in stats.items():
        if "pnl" in k or "capital" in k or "duration" in k:
            print(f"{k.replace('_', ' ').capitalize():<25}: {v:,.2f}")
        else:
            print(f"{k.replace('_', ' ').capitalize():<25}: {v}")

    pnl_by_pair = trades.groupby("Pair")["PnL_adj"].sum().sort_values(ascending=False)
    print("\n=== 🧩 PnL par paire (pondéré) ===")
    print(pnl_by_pair.head(15))

    # --- Visualisations ---
    plot_trade_distributions(trades, mode)

    # --- Export résumé ---
    export_summary(trades, mode)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
