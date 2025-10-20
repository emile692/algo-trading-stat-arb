"""
Comparaison de portefeuilles Static β vs WF β(t) vs Mixte
Analyse pro : Sharpe, Drawdown, Corrélation inter-pf et attribution de risque
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

# === Répertoires ===
EQ_DIR_STATIC = "results/equities/static"
EQ_DIR_WF     = "results/equities/wf"
SEL_STATIC_CSV = "results/selected_pairs_static.csv"
SEL_WF_CSV     = "results/selected_pairs_wf.csv"

FREQ_PER_YEAR = 252 * 6.5  # heures de trading / an
TARGET_VOL = 0.10

plt.style.use("seaborn-v0_8-whitegrid")

# ===============================================================
# 1️⃣ Fonctions utilitaires
# ===============================================================
def load_equity(pair, mode):
    path = os.path.join(EQ_DIR_STATIC if mode == "static" else EQ_DIR_WF, f"{pair}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"⚠️ Equity manquante : {path}")
    s = pd.read_csv(path, index_col=0).iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    s.name = pair
    return s

def to_returns(eq):
    r = eq.diff().fillna(0.0)
    r.name = eq.name
    return r

def sharpe_ratio(returns):
    mu, vol = returns.mean(), returns.std()
    return np.sqrt(FREQ_PER_YEAR) * mu / vol if vol > 1e-12 else 0.0

def max_drawdown(equity):
    roll_max = equity.cummax()
    dd = roll_max - equity
    return float(dd.max()) if len(dd) else 0.0

def annualize_vol(returns):
    return returns.std() * np.sqrt(FREQ_PER_YEAR)

def volatility_scale(returns, target_vol=TARGET_VOL):
    vol = annualize_vol(returns)
    scale = target_vol / vol if vol > 1e-12 else 0.0
    return returns * scale

def corr_penalized_inverse_vol_weights(ret_df, alpha=1.0):
    vols = ret_df.std()
    inv_vol = 1.0 / vols.replace(0, np.nan)
    corr = ret_df.corr().fillna(0.0)
    avg_corr = corr.where(~np.eye(len(corr), dtype=bool)).mean().fillna(0.0)
    score = inv_vol / (1.0 + alpha * avg_corr.clip(lower=0))
    w = score / score.sum()
    w.name = "weight"
    return w

# ===============================================================
# 2️⃣ Construction des portefeuilles
# ===============================================================
def build_portfolio(pairs, mode):
    if not pairs:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    rets = []
    for p in pairs:
        eq = load_equity(p, mode)
        r = to_returns(eq)
        r_scaled = volatility_scale(r)
        rets.append(r_scaled.rename(p))
    ret_df = pd.concat(rets, axis=1).fillna(0.0)
    w = corr_penalized_inverse_vol_weights(ret_df)
    port_ret = (ret_df * w).sum(axis=1)
    equity = port_ret.cumsum()
    return equity, port_ret, w

# ===============================================================
# 3️⃣ Chargement des sélections
# ===============================================================
if not (os.path.exists(SEL_STATIC_CSV) and os.path.exists(SEL_WF_CSV)):
    raise FileNotFoundError("❌ Les fichiers de sélection sont manquants. Lance d'abord `python -m experiments.portfolio_analysis`.")

sel_static = pd.read_csv(SEL_STATIC_CSV)["Pair"].tolist()
sel_wf     = pd.read_csv(SEL_WF_CSV)["Pair"].tolist()

print(f"📊 {len(sel_static)} paires Static chargées : {', '.join(sel_static)}")
print(f"📊 {len(sel_wf)} paires WF     chargées : {', '.join(sel_wf)}")

# ===============================================================
# 4️⃣ Construction des 3 portefeuilles
# ===============================================================
eq_static, ret_static, w_static = build_portfolio(sel_static, "static")
eq_wf, ret_wf, w_wf = build_portfolio(sel_wf, "wf")

# Portefeuille mixte 50/50 (pondéré par risque)
ret_mix = 0.5 * ret_static + 0.5 * ret_wf
eq_mix = ret_mix.cumsum()

# ===============================================================
# 5️⃣ Analyse de performance
# ===============================================================
S_static, DD_static = sharpe_ratio(ret_static), max_drawdown(eq_static)
S_wf, DD_wf = sharpe_ratio(ret_wf), max_drawdown(eq_wf)
S_mix, DD_mix = sharpe_ratio(ret_mix), max_drawdown(eq_mix)

corr_pf = ret_static.corr(ret_wf)

print("\n=== 🧠 Statistiques comparatives ===")
print(f"Static : Sharpe={S_static:.2f} | MaxDD={DD_static:.3f}")
print(f"WF     : Sharpe={S_wf:.2f} | MaxDD={DD_wf:.3f}")
print(f"Mixte  : Sharpe={S_mix:.2f} | MaxDD={DD_mix:.3f}")
print(f"Corrélation inter-portefeuilles : {corr_pf:.2f}")

# ===============================================================
# 6️⃣ Plots
# ===============================================================
plt.figure(figsize=(13, 4))
plt.plot(eq_static, label=f"Static β | Sharpe={S_static:.2f}")
plt.plot(eq_wf, label=f"WF β(t) | Sharpe={S_wf:.2f}")
plt.plot(eq_mix, label=f"Mixte 50/50 | Sharpe={S_mix:.2f}", linestyle="--", color="black")
plt.title("Comparaison des portefeuilles Static vs WF vs Mixte")
plt.xlabel("Temps (heures)")
plt.ylabel("Équity cumulée (unités arbitraires)")
plt.legend()
plt.tight_layout()
plt.show()

# Heatmap des corrélations inter-portefeuilles
comb_df = pd.concat(
    [ret_static.rename("Static"), ret_wf.rename("WF"), ret_mix.rename("Mixte")],
    axis=1
)
plt.figure(figsize=(4.5, 3.5))
sns.heatmap(comb_df.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, cbar=False)
plt.title("Corrélation inter-portefeuilles")
plt.tight_layout()
plt.show()

# ===============================================================
# 7️⃣ Attribution du risque et diversification
# ===============================================================
var_static = ret_static.var()
var_wf = ret_wf.var()
covar = ret_static.cov(ret_wf)
var_mix = 0.25 * (var_static + var_wf + 2 * covar)
div_benefit = 1 - (var_mix / (0.5 * (var_static + var_wf)))

print("\n=== ⚙️ Diversification et risque ===")
print(f"Var(Static)={var_static:.2e} | Var(WF)={var_wf:.2e}")
print(f"Covar(Static,WF)={covar:.2e}")
print(f"Var(Mixte)={var_mix:.2e}")
print(f"Gain de diversification ≈ {div_benefit*100:.1f}%")

# ===============================================================
# 8️⃣ Sauvegarde des poids et retours
# ===============================================================
os.makedirs("results/compare", exist_ok=True)
pd.DataFrame({
    "Static": w_static,
    "WF": w_wf
}).fillna(0).to_csv("results/compare/weights_comparison.csv")

comb_df.to_csv("results/compare/returns_comparison.csv")

print("💾 Résultats sauvegardés → results/compare/")
