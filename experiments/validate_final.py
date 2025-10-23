# ============================================================
# experiments/validate_final.py
# ============================================================
"""
Sélection finale des paires pour la stratégie stat-arb
------------------------------------------------------
1️⃣ Charge les résultats de backtest (walkforward_summary.csv ou final_candidates.csv)
2️⃣ Choisit la méthode cible : "Static", "WF" ou "Mixte"
3️⃣ Calcule un score composite (Sharpe ↑, MaxDD ↓, PnL ↑)
4️⃣ Exporte le top des paires retenues avec poids normalisés
"""

import argparse
import pandas as pd
from pathlib import Path


# ============================================================
# 🧠 FONCTIONS UTILITAIRES
# ============================================================

def load_results(input_path: Path) -> pd.DataFrame:
    """Charge et prépare le DataFrame selon le format du fichier."""
    df = pd.read_csv(input_path)

    # Cas 1 : déjà pivoté
    if any(col.startswith("Sharpe_") for col in df.columns):
        return df.copy()

    # Cas 2 : brut avec colonne 'Mode'
    if "Mode" in df.columns:
        df_pivot = df.pivot(index=["Universe", "Pair"], columns="Mode",
                            values=["Sharpe", "MaxDD", "FinalPnL"])
        df_pivot.columns = [f"{a}_{b}" for a, b in df_pivot.columns]
        return df_pivot.reset_index()

    raise ValueError(
        "❌ Le fichier ne contient ni les colonnes pivotées (Sharpe_WF, ...) "
        "ni une colonne 'Mode' pour effectuer le pivot."
    )


def compute_composite_score(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Calcule le score composite selon le mode choisi."""

    df = df.copy()

    if mode == "Mixte":
        # On prend la moyenne des perfs entre Static et WF
        df["Sharpe_Mixte"] = df[["Sharpe_Static", "Sharpe_WF"]].mean(axis=1)
        df["MaxDD_Mixte"] = df[["MaxDD_Static", "MaxDD_WF"]].mean(axis=1)
        df["FinalPnL_Mixte"] = df[["FinalPnL_Static", "FinalPnL_WF"]].mean(axis=1)

    # Sélection du bon suffixe (Static, WF, ou Mixte)
    df["score"] = (
        0.6 * df[f"Sharpe_{mode}"]
        + 0.3 * df[f"FinalPnL_{mode}"]
        - 0.1 * df[f"MaxDD_{mode}"]
    )

    # Filtre minimal
    df = df[
        (df[f"Sharpe_{mode}"] > 0) &
        (df[f"FinalPnL_{mode}"] > 0)
    ].copy()

    return df.sort_values("score", ascending=False).reset_index(drop=True)


def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les poids sur la base du score."""
    df = df.copy()
    df["weight"] = df["score"].clip(lower=0)
    if df["weight"].sum() > 0:
        df["weight"] /= df["weight"].sum()
    return df


# ============================================================
# 🚀 MAIN
# ============================================================

def main():
    # --- Arguments CLI ---
    parser = argparse.ArgumentParser(description="Sélection finale du portefeuille stat-arb.")
    parser.add_argument("--mode", choices=["Static", "WF", "Mixte"], default="WF",
                        help="Mode de sélection : 'Static', 'WF' ou 'Mixte' (par défaut: WF)")
    parser.add_argument("--input", type=str, default="results/walkforward_summary.csv",
                        help="Chemin vers le fichier de résultats CSV.")
    parser.add_argument("--top", type=int, default=10,
                        help="Nombre de paires à retenir (défaut: 10)")
    args = parser.parse_args()

    # --- Config ---
    input_path = Path(args.input)
    mode = args.mode
    top_n = args.top
    output_path = Path(f"results/final_portfolio_{mode}.csv")

    # --- Pipeline ---
    df = load_results(input_path)
    df = compute_composite_score(df, mode)
    df_top = df.head(top_n).copy()
    df_top = normalize_weights(df_top)

    # --- Export ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_top.to_csv(output_path, index=False)

    # --- Résumé ---
    print(f"\n=== 🏁 Portefeuille final ({mode}) ===")
    print(df_top[["Pair", f"Sharpe_{mode}", f"MaxDD_{mode}", f"FinalPnL_{mode}", "score", "weight"]])
    print(f"\n💾 Sauvegardé → {output_path}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
