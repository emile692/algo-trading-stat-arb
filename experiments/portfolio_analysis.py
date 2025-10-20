# ============================================================
# experiments/portfolio_analysis.py — version pro (filtrage, sélection, risk-parity, corr penalty)
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

RESULTS_CSV = "results/walkforward_summary.csv"
EQ_DIR_STATIC = "results/equities/static"
EQ_DIR_WF = "results/equities/wf"

# ==========================
# Paramètres globaux
# ==========================
TOP_N             = 10
FREQ_PER_YEAR     = 252 * 6.5  # ~heures boursières/an si données 1h
TARGET_ANNU_VOL   = 0.10
MIN_SHARPE_STATIC = 0.30
MIN_SHARPE_WF     = 0.30
MAX_DD_CAP        = 3.50
MIN_TRADES        = 2
CORR_PENALTY_ALPHA = 1.0

plt.style.use("seaborn-v0_8-whitegrid")

# ==========================
# Fonctions utilitaires
# ==========================
def sharpe_ratio(returns, freq_per_year=FREQ_PER_YEAR):
    mu, vol = returns.mean(), returns.std()
    return (np.sqrt(freq_per_year) * mu / vol) if vol > 1e-12 else 0.0

def max_drawdown(equity):
    roll_max = equity.cummax()
    dd = roll_max - equity
    return float(dd.max()) if len(dd) else 0.0

def load_equity(pair, mode):
    path = os.path.join(EQ_DIR_STATIC if mode == "static" else EQ_DIR_WF, f"{pair}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"⚠️ Equity manquante: {path}")
    s = pd.read_csv(path, index_col=0).iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    s.name = pair
    return s

def to_returns(eq: pd.Series) -> pd.Series:
    s = eq.diff().fillna(0.0)
    s.name = eq.name
    return s

def annualize_vol(returns):
    return returns.std() * np.sqrt(FREQ_PER_YEAR)

def volatility_scale(returns, target_vol):
    vol = annualize_vol(returns)
    scale = target_vol / vol if vol > 1e-12 else 0.0
    return returns * scale

def corr_penalized_inverse_vol_weights(ret_df, alpha=CORR_PENALTY_ALPHA):
    vols = ret_df.std()
    inv_vol = 1.0 / vols.replace(0, np.nan)
    corr = ret_df.corr().fillna(0.0)
    avg_corr = corr.where(~np.eye(len(corr), dtype=bool)).mean().fillna(0.0)
    score = inv_vol / (1.0 + alpha * avg_corr.clip(lower=0))
    w = score / score.sum()
    w.name = "weight"
    return w

# ==========================
# Sélection gloutonne corr-aware (robuste)
# ==========================
def greedy_diversified_selection(candidates, mode, top_k):
    selected, remaining = [], candidates.index.tolist()
    if not remaining:
        print(f"⚠️ Aucun candidat initial pour {mode}.")
        return selected

    ret_map = {}
    for p in remaining:
        try:
            eq = load_equity(p, mode)
            r = to_returns(eq)
            ret_map[p] = r
        except Exception as e:
            print(f"⚠️ Erreur chargement {p}: {e}")
            continue

    while len(selected) < top_k and remaining:
        best_pair, best_score = None, -np.inf
        for p in remaining:
            test_set = selected + [p]
            valid = [x for x in test_set if x in ret_map and not ret_map[x].empty]
            if not valid:
                continue
            ret_df = pd.concat([ret_map[x] for x in valid], axis=1).fillna(0.0)
            if ret_df.shape[0] < 10:
                continue
            try:
                w = corr_penalized_inverse_vol_weights(ret_df)
                port_ret = (ret_df * w).sum(axis=1)
                s = sharpe_ratio(port_ret)
                avg_corr = ret_df.corr().where(~np.eye(len(valid), dtype=bool)).mean().mean()
                penalized = s / sqrt(max(avg_corr, 1e-6))
                if np.isfinite(penalized) and penalized > best_score:
                    best_score, best_pair = penalized, p
            except Exception as e:
                print(f"⚠️ Erreur sur {p}: {e}")
                continue

        if best_pair is None:
            if not selected and remaining:
                best_pair = remaining[0]
                print(f"⚠️ Sélection forcée du premier pair {best_pair} (init {mode}).")
            else:
                print(f"⚠️ Aucune paire supplémentaire trouvée à l’étape {len(selected)} ({mode}).")
                break

        selected.append(best_pair)
        remaining.remove(best_pair)
    return selected

# ==========================
# Chargement du résumé global
# ==========================
if not os.path.exists(RESULTS_CSV):
    raise FileNotFoundError(f"{RESULTS_CSV} introuvable. Lance d'abord python -m experiments.run_walkforward")

df = pd.read_csv(RESULTS_CSV)
if "Mode" in df.columns:
    df = df.pivot(index="Pair", columns="Mode", values=["Sharpe", "MaxDD", "FinalPnL", "Trades"])
    df.columns = [f"{m}_{c}" for m, c in df.columns]
    df = df.reset_index()

print(f"✅ Fichier chargé et pivoté : {df.shape}")

# ==========================
# Filtrage indépendant (Static vs WF)
# ==========================
def filter_candidates(df, min_sharpe_s, min_sharpe_w, max_dd, min_trades, mode="both"):
    if mode == "static":
        mask = (
            (df["Sharpe_Static"] >= min_sharpe_s) &
            (df["MaxDD_Static"] <= max_dd) &
            (df["Trades_Static"] >= min_trades)
        )
    elif mode == "wf":
        mask = (
            (df["Sharpe_WF"] >= min_sharpe_w) &
            (df["MaxDD_WF"] <= max_dd) &
            (df["Trades_WF"] >= min_trades)
        )
    else:
        mask = (
            (df["Sharpe_Static"] >= min_sharpe_s) &
            (df["Sharpe_WF"] >= min_sharpe_w)
        )
    return df.loc[mask].copy()

df_filt_static = filter_candidates(df, MIN_SHARPE_STATIC, MIN_SHARPE_WF, MAX_DD_CAP, MIN_TRADES, "static")
df_filt_wf = filter_candidates(df, MIN_SHARPE_STATIC, MIN_SHARPE_WF, MAX_DD_CAP, MIN_TRADES, "wf")

print(f"🔎 Candidats Static retenus : {len(df_filt_static)} / {len(df)}")
print(f"🔎 Candidats WF retenus     : {len(df_filt_wf)} / {len(df)}")

sel_static = greedy_diversified_selection(df_filt_static.set_index("Pair"), "static", top_k=min(TOP_N, len(df_filt_static)))
sel_wf = greedy_diversified_selection(df_filt_wf.set_index("Pair"), "wf", top_k=min(TOP_N, len(df_filt_wf)))

print(f"✅ Sélection Static ({len(sel_static)}): {', '.join(sel_static)}")
print(f"✅ Sélection WF     ({len(sel_wf)}): {', '.join(sel_wf)}")

# ==========================
# Construction du portefeuille
# ==========================
def build_portfolio(pairs, mode):
    if not pairs:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    rets = []
    for p in pairs:
        eq = load_equity(p, mode)
        r = to_returns(eq)
        r_scaled = volatility_scale(r, TARGET_ANNU_VOL)
        rets.append(r_scaled.rename(p))
    ret_df = pd.concat(rets, axis=1).fillna(0.0)
    w = corr_penalized_inverse_vol_weights(ret_df)
    port_ret = (ret_df * w).sum(axis=1)
    equity = port_ret.cumsum()
    return equity, port_ret

equity_static, pret_static = build_portfolio(sel_static, "static")
equity_wf, pret_wf = build_portfolio(sel_wf, "wf")

S_static, S_wf = sharpe_ratio(pret_static), sharpe_ratio(pret_wf)
DD_static, DD_wf = max_drawdown(equity_static), max_drawdown(equity_wf)

print("\n=== Statistiques Portefeuille (corr-aware, vol-scaled) ===")
print(f"Static : Sharpe={S_static:.2f} | MaxDD={DD_static:.2f} | N={len(sel_static)}")
print(f"WF     : Sharpe={S_wf:.2f} | MaxDD={DD_wf:.2f} | N={len(sel_wf)}")

# ==========================
# Plots et corrélations
# ==========================
plt.figure(figsize=(14, 4.5))
plt.plot(equity_static, label=f"Static β | Sharpe={S_static:.2f}", linewidth=2)
plt.plot(equity_wf, label=f"WF β(t) | Sharpe={S_wf:.2f}", linewidth=2)
plt.title(f"Portefeuilles égal-risque — Static vs WF (TOP_N={TOP_N})")
plt.xlabel("Temps (heures)")
plt.ylabel("Equity cumulée")
plt.legend()
plt.tight_layout()
plt.show()

def heatmap_corr(pairs, mode, title):
    if not pairs:
        return
    rets = [to_returns(load_equity(p, mode)) for p in pairs]
    ret_df = pd.concat(rets, axis=1).fillna(0.0)
    corr = ret_df.corr()
    avg_corr = corr.where(~np.eye(len(corr), dtype=bool)).mean().mean()
    plt.figure(figsize=(6.2, 4.8))
    sns.heatmap(corr, cmap="coolwarm", center=0, cbar=False)
    plt.title(f"{title} (moy corr={avg_corr:.2f})")
    plt.tight_layout()
    plt.show()

heatmap_corr(sel_static, "static", "Corrélation — Portefeuille Static")
heatmap_corr(sel_wf, "wf", "Corrélation — Portefeuille WF")

# ==========================
# Sauvegardes
# ==========================
os.makedirs("results", exist_ok=True)
pd.DataFrame({"Pair": sel_static, "Mode": "Static"}).to_csv("results/selected_pairs_static.csv", index=False)
pd.DataFrame({"Pair": sel_wf, "Mode": "WF"}).to_csv("results/selected_pairs_wf.csv", index=False)

print("💾 Sauvegardes → results/selected_pairs_*.csv")
