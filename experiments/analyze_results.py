# ============================================================
# experiments/analyze_results.py (fix format long/pivot)
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1️⃣ Chargement du fichier
path = "results/walkforward_summary.csv"
if not os.path.exists(path):
    raise FileNotFoundError(f"Fichier introuvable : {path}")

df = pd.read_csv(path)
print(f"✅ Résultats chargés : {df.shape}")
print(df.head())

# === 2️⃣ Harmonisation du format (pivot si nécessaire)
if "Mode" in df.columns:
    # format long → pivot
    df = df.pivot(index="Pair", columns="Mode", values=["Sharpe", "MaxDD", "FinalPnL"])
    df.columns = [f"{metric}_{mode}" for metric, mode in df.columns]
    df = df.reset_index()
    print("🔁 Données pivotées (Static/WF dans des colonnes séparées).")

# === 3️⃣ Si ΔSharpe n'existe pas, le créer
if "Sharpe_Static" in df.columns and "Sharpe_WF" in df.columns:
    df["ΔSharpe"] = df["Sharpe_WF"] - df["Sharpe_Static"]
else:
    raise KeyError("Impossible de trouver les colonnes 'Sharpe_Static' et 'Sharpe_WF' après pivot.")

print(f"✅ Format final : {df.shape} colonnes = {df.columns.tolist()}")

# === 4️⃣ Statistiques globales
print("\n=== Statistiques globales ===")
print(df[["Sharpe_Static", "Sharpe_WF", "ΔSharpe"]].describe().round(3))

mean_delta = df["ΔSharpe"].mean()
improved = (df["ΔSharpe"] > 0).mean() * 100
print(f"\nMoyenne ΔSharpe = {mean_delta:.3f}")
print(f"% de paires améliorées par WF = {improved:.1f}%")


# === 4️⃣ Top / Bottom paires par ΔSharpe
top10 = df.sort_values("ΔSharpe", ascending=False).head(10)
bottom10 = df.sort_values("ΔSharpe", ascending=True).head(10)

print("\n=== Top 10 paires (WF > Static) ===")
print(top10[["Sharpe_Static", "Sharpe_WF", "ΔSharpe"]].round(3))
print("\n=== Bottom 10 paires (WF < Static) ===")
print(bottom10[["Sharpe_Static", "Sharpe_WF", "ΔSharpe"]].round(3))

# === 5️⃣ Graphiques comparatifs
plt.style.use("seaborn-v0_8-darkgrid")

# --- (a) Histogramme des ΔSharpe
plt.figure(figsize=(8,4))
sns.histplot(df["ΔSharpe"], bins=25, kde=True, color="dodgerblue")
plt.axvline(0, color="red", linestyle="--", linewidth=1)
plt.title("Distribution des ΔSharpe (WF - Static)")
plt.xlabel("ΔSharpe"); plt.ylabel("Nombre de paires")
plt.tight_layout()
plt.show()

# --- (b) Barplot top/bottom ΔSharpe
plt.figure(figsize=(10,6))
combined = pd.concat([
    top10.assign(Type="Top 10 (WF > Static)"),
    bottom10.assign(Type="Bottom 10 (WF < Static)")
])
sns.barplot(
    data=combined,
    x="ΔSharpe", y=combined.index,
    hue="Type", dodge=False, palette=["limegreen", "salmon"]
)
plt.title("Top / Bottom ΔSharpe par paire")
plt.xlabel("ΔSharpe (Sharpe_WF - Sharpe_Static)")
plt.ylabel("Paire")
plt.tight_layout()
plt.show()

# === 6️⃣ (Optionnel) Corrélation entre Sharpe Static et WF
plt.figure(figsize=(6,6))
sns.scatterplot(
    data=df,
    x="Sharpe_Static", y="Sharpe_WF",
    color="steelblue", s=40, alpha=0.8
)
plt.plot([-2,2], [-2,2], color="red", linestyle="--")
plt.title("Corrélation entre Sharpe Static et WF")
plt.xlabel("Sharpe Static")
plt.ylabel("Sharpe WF")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# === 7️⃣ Sauvegarde top paires
os.makedirs("results", exist_ok=True)
df.sort_values("Sharpe_WF", ascending=False).head(20).to_csv("results/top20_WF.csv")
df.sort_values("Sharpe_Static", ascending=False).head(20).to_csv("results/top20_Static.csv")
print("💾 Fichiers sauvegardés : top20_WF.csv & top20_Static.csv")
