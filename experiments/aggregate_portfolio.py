# ============================================================
# experiments/aggregate_portfolio.py
# ============================================================
"""
Agrégation et analyse du portefeuille final Stat-Arb
-----------------------------------------------------
1️⃣ Charge le portefeuille final (Static, WF ou Mixte)
2️⃣ Récupère les équités individuelles pondérées
3️⃣ Construit et sauvegarde l’équité totale du portefeuille
4️⃣ Calcule les métriques de performance (Sharpe, PnL, etc.)
5️⃣ Affiche la courbe de NAV cumulée
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from utils.time_scaling import get_bars_per_year


# ============================================================
# ⚙️ CONFIGURATION UTILITAIRE
# ============================================================

def load_params(config_path: str = "config/params.yaml") -> dict:
    """Charge le fichier YAML de configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_equity_dirs(mode: str) -> list[Path]:
    """Retourne la liste des dossiers d’équités à utiliser selon le mode."""
    if mode.lower() == "mixte":
        return [Path("results/equities/static"), Path("results/equities/wf")]
    return [Path(f"results/equities/{mode.lower()}")]


def load_final_portfolio(mode: str) -> pd.DataFrame:
    """Charge le CSV du portefeuille final selon le mode."""
    path = Path(f"results/final_portfolio_{mode}.csv")
    if not path.exists():
        raise FileNotFoundError(f"❌ Fichier introuvable : {path}")
    print(f"📈 Chargement du portefeuille final : {path}")
    return pd.read_csv(path)


def load_and_weight_equities(final_df: pd.DataFrame, equity_dirs: list[Path],
                             initial_capital: float) -> list[pd.Series]:
    """Charge les équités des paires, les pondère et retourne une liste de séries."""
    equities = []

    for _, row in final_df.iterrows():
        pair = row["Pair"]
        weight = row["weight"]

        found_path = None
        for directory in equity_dirs:
            candidate = directory / f"{pair}.csv"
            if candidate.exists():
                found_path = candidate
                break

        if not found_path:
            print(f"⚠️ Équity manquante : {pair}")
            continue

        eq = pd.read_csv(found_path, index_col=0)
        eq.index = pd.to_datetime(eq.index)

        # On prend la première colonne numérique trouvée
        num_cols = eq.select_dtypes(include=["float", "int"]).columns
        if len(num_cols) == 0:
            print(f"⚠️ Aucune colonne numérique trouvée pour {pair}")
            continue

        equity_col = num_cols[0]
        # baseline par paire pour avoir un PnL en € (base 0)
        eq_euros = (eq[equity_col] - eq[equity_col].iloc[0]) * weight
        equities.append(eq_euros)

    if not equities:
        raise ValueError("❌ Aucune équity trouvée. Vérifie les CSV dans results/equities/...")

    return equities


def compute_performance(portfolio_eq: pd.Series, bars_per_year: int,
                        initial_capital: float) -> dict:
    """Calcule les métriques de performance du portefeuille."""

    # plus de coupe ni re-baseline : portfolio_eq est déjà un PnL cumulé pondéré
    nav = initial_capital + portfolio_eq

    pnl_diff = portfolio_eq.diff().replace([np.inf, -np.inf], np.nan).dropna()
    returns = pnl_diff / initial_capital

    sharpe = float("nan")
    if not returns.empty and returns.std() != 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(bars_per_year)

    final_value = nav.iloc[-1]
    gain_euros = final_value - initial_capital
    gain_pct = (final_value / initial_capital - 1) * 100

    return {
        "sharpe": sharpe,
        "final_value": final_value,
        "gain_euros": gain_euros,
        "gain_pct": gain_pct,
        "nav": nav,
    }


def plot_nav(nav: pd.Series, mode: str, initial_capital: float):
    """Affiche la courbe de valeur cumulée du portefeuille."""
    plt.figure(figsize=(10, 5))
    nav.plot(title=f"Portefeuille {mode} — Valeur cumulée (base {initial_capital:,.0f} €)", lw=2)
    plt.ylabel("Valeur (€)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 🚀 MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Agrégation du portefeuille final Stat-Arb.")
    parser.add_argument("--mode", choices=["Static", "WF", "Mixte"], default="Static",
                        help="Mode du portefeuille : Static, WF ou Mixte (défaut: Static)")
    parser.add_argument("--capital", type=float, default=10_000,
                        help="Capital initial simulé (défaut: 10 000 €)")
    args = parser.parse_args()

    # --- Paramètres globaux ---
    params = load_params("config/params.yaml")
    freq = params.get("freq", "1H")
    bars_per_year = get_bars_per_year(freq)

    mode = args.mode
    initial_capital = args.capital

    # --- Chargement du portefeuille ---
    final_df = load_final_portfolio(mode)
    equity_dirs = get_equity_dirs(mode)
    equities = load_and_weight_equities(final_df, equity_dirs, initial_capital)

    # --- Agrégation ---
    portfolio_eq = sum(equities)
    portfolio_eq.name = "Portfolio"

    output_path = Path(f"results/portfolio_equity_{mode.lower()}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio_eq.to_csv(output_path)
    print(f"💾 Sauvegardé → {output_path}")

    # --- Calculs de performance ---
    stats = compute_performance(portfolio_eq, bars_per_year, initial_capital)
    print(f"\n🏁 Sharpe global ({mode}) : {stats['sharpe']:.2f}")
    print(f"💰 Valeur finale : {stats['final_value']:,.2f} €")
    print(f"📈 Gain total : {stats['gain_euros']:,.2f} € ({stats['gain_pct']:.2f}%)")

    # --- Visualisation ---
    plot_nav(stats["nav"], mode, initial_capital)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
