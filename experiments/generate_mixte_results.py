from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "results" / "walkforward_summary.csv"
df = pd.read_csv(CSV_PATH)

# Pivot table pour avoir les deux modes sur une même ligne
df_pivot = df.pivot(index=["Universe", "Pair"], columns="Mode", values=["Sharpe", "MaxDD", "FinalPnL"])
df_pivot.columns = [f"{a}_{b}" for a, b in df_pivot.columns]
df_pivot = df_pivot.reset_index()

# === Génération du mode Mixte (50/50 entre Static et WF)
df_pivot["Sharpe_Mixte"] = 0.5 * (df_pivot["Sharpe_Static"] + df_pivot["Sharpe_WF"])
df_pivot["MaxDD_Mixte"] = 0.5 * (df_pivot["MaxDD_Static"] + df_pivot["MaxDD_WF"])
df_pivot["FinalPnL_Mixte"] = 0.5 * (df_pivot["FinalPnL_Static"] + df_pivot["FinalPnL_WF"])

# Sauvegarde pour validate_final
df_pivot.to_csv(CSV_PATH, index=False)
print("✅ Colonnes Mixte ajoutées → results/walkforward_summary.csv")
