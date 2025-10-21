import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from utils.time_scaling import get_bars_per_year

# === ⚙️ CONFIGURATION ===
with open("config/params.yaml", "r") as f:
    params = yaml.safe_load(f)

FREQ = params.get("freq", "1H")
BARS_PER_YEAR = get_bars_per_year(FREQ)
INITIAL_CAPITAL = 10_000  # 💰 Capital total simulé

MODE = "WF"  # 🔧 Choisis entre "Static", "WF" ou "Mixte"
final_path = f"results/final_portfolio_{MODE}.csv"
final = pd.read_csv(final_path)
print(f"📈 Chargement du portefeuille final : {final_path}")

# === 📂 Dossiers d'équités selon le mode ===
if MODE.lower() == "mixte":
    equity_dirs = [Path("results/equities/static"), Path("results/equities/wf")]
else:
    equity_dirs = [Path(f"results/equities/{MODE.lower()}")]

# === Agrégation des équités pondérées ===
equities = []
for _, row in final.iterrows():
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

    num_cols = eq.select_dtypes(include=["float", "int"]).columns
    if len(num_cols) == 0:
        print(f"⚠️ Aucune colonne numérique trouvée pour {pair}")
        continue

    equity_col = num_cols[0]
    # 🧮 Conversion en euros selon poids du portefeuille
    eq_euros = eq[equity_col] * (weight * INITIAL_CAPITAL)
    equities.append(eq_euros)

if not equities:
    raise ValueError("❌ Aucune équity trouvée. Vérifie que les CSV sont bien exportés dans results/equities/...")

# === 📊 Construction du portefeuille ===
portfolio_eq = sum(equities)
portfolio_eq.name = "Portfolio"

# === Sauvegarde brute ===
output_path = f"results/portfolio_equity_{MODE.lower()}.csv"
portfolio_eq.to_csv(output_path)

# === Calcul de performance ===
portfolio_eq = portfolio_eq.loc[portfolio_eq.ne(0).idxmax():]
portfolio_eq = portfolio_eq[portfolio_eq > 0]

# Convertit le PnL cumulatif pondéré en NAV réel
portfolio_eq_start = portfolio_eq.iloc[0]
nav = INITIAL_CAPITAL + (portfolio_eq - portfolio_eq_start)

# Retours = ΔPnL / capital
pnl_diff = portfolio_eq.diff().replace([np.inf, -np.inf], np.nan).dropna()
returns = pnl_diff / INITIAL_CAPITAL

if returns.empty or returns.std() == 0:
    sharpe = float("nan")
else:
    sharpe = returns.mean() / returns.std() * (BARS_PER_YEAR ** 0.5)

final_value = nav.iloc[-1]
gain_euros = final_value - INITIAL_CAPITAL
gain_pct = (final_value / INITIAL_CAPITAL - 1) * 100

print(f"🏁 Sharpe global ({MODE}) : {sharpe:.2f}")
print(f"💰 Valeur finale du portefeuille : {final_value:,.2f} €")
print(f"📈 Gain total : {gain_euros:,.2f} €  ({gain_pct:.2f}%)")
print(f"💾 Sauvegardé → {output_path}")

# === 📈 Visualisation ===
plt.figure(figsize=(10, 5))
nav.plot(title=f"Portefeuille {MODE} — Valeur cumulée (base 10 000 €)", lw=2)
plt.ylabel("Valeur (€)")
plt.grid(True)
plt.tight_layout()
plt.show()
