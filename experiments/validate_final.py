# ============================================================
# experiments/validate_final.py
# ============================================================
"""
Sélection finale des paires pour la stratégie stat-arb
------------------------------------------------------
1️⃣ Charge les résultats de backtest (walkforward_summary.csv ou final_candidates.csv)
2️⃣ Choisit la méthode cible : "Static" ou "WF"
3️⃣ Calcule un score composite (Sharpe ↑, MaxDD ↓, PnL ↑)
4️⃣ Exporte le top des paires retenues avec poids normalisés
"""

import pandas as pd
from pathlib import Path

# === CONFIG ===
INPUT_PATH = Path("results/walkforward_summary.csv")  # ou final_candidates.csv
MODE = "WF"  # 🔧 "Static" ou "WF" selon ta stratégie finale
OUTPUT_PATH = Path(f"results/final_portfolio_{MODE}.csv")
TOP_N = 10   # nombre de paires à retenir

# === CHARGEMENT ===
df = pd.read_csv(INPUT_PATH)

# Le fichier est déjà en format large (pas besoin de pivot)
if "Sharpe_WF" in df.columns:
    df_pivot = df.copy()
else:
    # fallback si fichier brut
    if "Mode" in df.columns:
        df_pivot = df.pivot(index=["Universe", "Pair"], columns="Mode", values=["Sharpe", "MaxDD", "FinalPnL"])
        df_pivot.columns = [f"{a}_{b}" for a, b in df_pivot.columns]
        df_pivot = df_pivot.reset_index()
    else:
        raise ValueError("❌ Le fichier ne contient ni les colonnes pivotées (Sharpe_WF, ...) ni une colonne 'Mode'.")


# === CALCUL DU SCORE COMPOSITE ===
# pondérations : Sharpe 60%, PnL 30%, Drawdown 10% (pénalisant)
df_pivot["score"] = (
    0.6 * df_pivot[f"Sharpe_{MODE}"] +
    0.3 * df_pivot[f"FinalPnL_{MODE}"] -
    0.1 * df_pivot[f"MaxDD_{MODE}"]
)

# Filtre minimal (sécurité)
df_pivot = df_pivot[
    (df_pivot[f"Sharpe_{MODE}"] > 0) &
    (df_pivot[f"FinalPnL_{MODE}"] > 0)
].copy()

# Classement final
df_pivot = df_pivot.sort_values("score", ascending=False).reset_index(drop=True)
df_top = df_pivot.head(TOP_N).copy()

# Normalisation des poids (en fonction du score)
df_top["weight"] = df_top["score"].clip(lower=0)
if df_top["weight"].sum() > 0:
    df_top["weight"] /= df_top["weight"].sum()

# === EXPORT ===
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_top.to_csv(OUTPUT_PATH, index=False)

# === AFFICHAGE ===
print(f"\n=== 🏁 Portefeuille final ({MODE}) ===")
print(df_top[["Pair", f"Sharpe_{MODE}", f"MaxDD_{MODE}", f"FinalPnL_{MODE}", "score", "weight"]])
print(f"\n💾 Sauvegardé → {OUTPUT_PATH}")