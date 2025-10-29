# ============================================================
# experiments/half_life_analysis.py (version améliorée)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "equities" / "static"

# ============================================================
# 🔍 Fonction de calcul de half-life
# ============================================================
def estimate_half_life(z: pd.Series) -> float:
    """Calcule la half-life du spread normalisé z_t."""
    z = z.dropna()
    if len(z) < 10:
        return np.nan

    z_lag = z.shift(1).dropna()
    delta_z = z.diff().dropna()
    z_lag, delta_z = z_lag.align(delta_z, join="inner")

    try:
        model = sm.OLS(delta_z, sm.add_constant(z_lag)).fit()
        rho = model.params.iloc[1]
        half_life = -np.log(2) / rho if rho < 0 else np.inf
    except Exception:
        half_life = np.nan

    return half_life

# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    print("=== ⏳ Analyse de half-life pour les spreads ===")
    results = []

    for path in RESULTS_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(path)
            # Détection automatique de la colonne du zscore
            col_z = None
            for c in df.columns:
                if "0" in c.lower():
                    col_z = c
                    break
            if col_z is None:
                print(f"⚠️  Colonne zscore absente dans {path.name}")
                continue

            hl = estimate_half_life(df[col_z])
            results.append((path.stem, hl))
            status = "📈" if np.isfinite(hl) and hl < 2000 else "⚠️"
            print(f"{status} {path.stem:<20} → Half-life = {hl:8.2f} barres")

        except Exception as e:
            print(f"❌ Erreur sur {path.name}: {e}")

    # =========================================
    # 🧹 Nettoyage et analyse
    # =========================================
    res_df = pd.DataFrame(results, columns=["Pair", "Half_life_bars"])
    res_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    res_df["stationary"] = res_df["Half_life_bars"].notna()

    # Convertir en heures et jours
    res_df["Half_life_hours"] = res_df["Half_life_bars"]
    res_df["Half_life_days"] = res_df["Half_life_hours"] / 24

    # Compte et résumé
    n_total = len(res_df)
    n_stationary = res_df["stationary"].sum()
    pct_stationary = 100 * n_stationary / n_total if n_total > 0 else 0

    print("\n=== 📊 Statistiques globales ===")
    print(f"Total spreads analysés     : {n_total}")
    print(f"Spreads stationnaires (HL<∞): {n_stationary} → {pct_stationary:.1f}%")

    valid = res_df.dropna(subset=["Half_life_bars"])
    if not valid.empty:
        print("\n=== ⏱️ Moyenne (sur spreads stationnaires uniquement) ===")
        print(valid[["Half_life_bars", "Half_life_days"]].describe().T)
    else:
        print("\n⚠️ Aucun spread stationnaire détecté.")

    # Sauvegarde
    out_path = RESULTS_DIR / "half_life_summary_clean.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\n💾 Résumé propre sauvegardé → {out_path}")

# ============================================================
if __name__ == "__main__":
    main()
