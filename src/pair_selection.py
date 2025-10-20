"""
pair_selection.py

Sélection de paires cointegrées + scoring pour stat-arb.

Entrées :
- DataFrame 'prices' : index datetime, colonnes = tickers (série de PRIX, ex: Close)
  OU DataFrame MultiIndex yfinance (('Close', TICKER), ...), à passer via extract_field_yf().

Sortie :
- DataFrame des paires avec colonnes :
  ['ticker_a','ticker_b','pvalue','beta','intercept','half_life',
   'spread_mean','spread_vol','n_obs','score','fdr_reject']

Dépendances : pandas, numpy, statsmodels, scipy (indirect)
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


# =========================
# Helpers I/O & Pré-traitement
# =========================

def extract_field_yf(df: pd.DataFrame, field: str = "Close") -> pd.DataFrame:
    """
    Extrait un champ ('Close', 'Adj Close', 'Open', ...) d'un DataFrame yfinance MultiIndex.
    Retourne un DataFrame simple : index = datetime, colonnes = tickers.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        # colonnes simples -> si 'field' est direct
        if field in df.columns:
            sub = df[field].to_frame() if not isinstance(df[field], pd.DataFrame) else df[field]
            return sub
        raise ValueError(f"Colonnes simples et champ {field!r} introuvable.")

    lv0 = set(df.columns.get_level_values(0))
    lv1 = set(df.columns.get_level_values(1))
    if field in lv0:
        sub = df[field].copy()
    elif field in lv1:
        sub = df.xs(field, axis=1, level=1).copy()
    else:
        raise ValueError(f"Champ {field!r} introuvable dans les colonnes MultiIndex.")

    # Tri des colonnes (optionnel)
    sub = sub.reindex(sorted(sub.columns), axis=1)
    return sub


def prepare_prices(prices: pd.DataFrame, tz: str | None = None) -> pd.DataFrame:
    """
    Nettoyage basique : tri par index, ffill/bfill, suppression colonnes avec NaN restants.
    Option : convertir le fuseau horaire.
    """
    p = prices.copy().sort_index()
    # si l'index n'est pas datetime, on tente de le convertir
    if not isinstance(p.index, pd.DatetimeIndex):
        p.index = pd.to_datetime(p.index, utc=True, errors="coerce")
    # timezone
    if p.index.tz is None:
        p.index = p.index.tz_localize("UTC")
    if tz:
        p.index = p.index.tz_convert(tz)

    p = p.ffill().bfill()
    p = p.dropna(axis=1, how="any")
    return p


# =========================
# Modèles/Stats
# =========================

def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> Tuple[float, float, sm.regression.linear_model.RegressionResultsWrapper]:
    """
    OLS : y = alpha + beta * x + eps
    Retourne (beta, alpha, model)
    """
    X = sm.add_constant(x.values)  # [const, x]
    model = sm.OLS(y.values, X).fit()
    beta = float(model.params[1])
    alpha = float(model.params[0])
    return beta, alpha, model



def engle_granger_min_pvalue(a: pd.Series, b: pd.Series, trend: str = "c") -> tuple[float, str]:
    # pval y~x
    p1 = coint(a, b, trend=trend)[1]
    # pval x~y
    p2 = coint(b, a, trend=trend)[1]
    if p1 <= p2:
        return float(p1), "y_on_x"
    else:
        return float(p2), "x_on_y"



def half_life(spread: pd.Series) -> float:
    """
    Estimation de la half-life via régression AR(1) sur Δs_t = a + γ * s_{t-1} + eps.
    HL ≈ -ln(2) / γ si γ < 0. Sinon -> inf (pas mean-reverting).
    """
    s = spread.dropna().values
    if s.size < 10:
        return np.nan

    ds = np.diff(s)
    s_lag = s[:-1]
    X = sm.add_constant(s_lag)
    try:
        res = sm.OLS(ds, X).fit()
        gamma = float(res.params[1])
    except Exception:
        return np.nan

    if gamma >= 0:
        return np.inf
    hl = -math.log(2.0) / gamma
    return max(1.0, hl)  # plancher à 1 barre


# =========================
# Scoring & FDR
# =========================

def score_pair(pvalue: float, hl: float, spread_vol: float,
               hl_min: float = 1.0, hl_max: float = 250.0) -> float:
    """
    Score simple :
    - pvalue petite -> meilleur (transformée -log10)
    - half-life ni trop courte ni trop longue (pénalise hors plage)
    - spread_vol non nulle (légère préférence pour un spread 'vivant')
    """
    # p-value (plus petit, mieux)
    pv_comp = -np.log10(max(pvalue, 1e-12)) if not np.isnan(pvalue) else -12.0

    # half-life (préférence milieu de plage)
    if np.isinf(hl) or np.isnan(hl):
        hl_comp = -5.0
    else:
        mid = (hl_min + hl_max) / 2.0
        hl_comp = -abs(np.log(hl + 1.0) - np.log(mid + 1.0))

    # volatilité du spread
    if np.isnan(spread_vol) or spread_vol <= 0:
        sv_comp = -5.0
    else:
        sv_comp = np.log(spread_vol)

    return 1.5 * pv_comp + 1.0 * hl_comp + 0.5 * sv_comp


def benjamini_hochberg(pvals: Iterable[float], alpha: float = 0.05) -> List[bool]:
    """
    Benjamini–Hochberg FDR (implémentation standard) :
    - trie p croissant
    - trouve k = max{i : p_(i) <= (i/m)*alpha}
    - rejette toutes les p <= p_(k)

    Retourne une liste de booléens alignée sur l'ordre initial.
    """
    p = np.asarray(list(pvals), dtype=float)
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    thresh = (np.arange(1, m + 1) / m) * alpha

    # indices où p_i <= thresh_i
    ok = p_sorted <= thresh
    if not np.any(ok):
        return [False] * m

    k = np.max(np.where(ok)[0])  # index du dernier vrai
    cutoff = p_sorted[k]

    rejected = p <= cutoff
    return rejected.tolist()


# =========================
# Analyse principale
# =========================

def analyze_pairs(prices: pd.DataFrame,
                  max_pairs: int | None = None,
                  min_lookback: int = 250,
                  pvalue_threshold: float = 0.05,
                  hl_min: float = 1.0,
                  hl_max: float = 250.0,
                  verbose: bool = False) -> pd.DataFrame:
    """
    Calcule les métriques pour toutes les combinaisons de colonnes (tickers) dans 'prices'.

    Params clés :
    - min_lookback : nb min d'observations communes exigées
    - pvalue_threshold : niveau FDR (Benjamini–Hochberg)
    - hl_min / hl_max : plage "raisonnable" pour half-life (en barres)

    Retour :
    DataFrame triée par fdr_reject desc, score desc, pvalue asc.
    """
    assert isinstance(prices.index, pd.DatetimeIndex), "Index datetime requis."
    tickers = list(prices.columns)
    combos = list(combinations(tickers, 2))

    if (max_pairs is not None) and (len(combos) > max_pairs):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(combos), size=max_pairs, replace=False)
        combos = [combos[i] for i in idx]

    results: List[dict] = []
    for a, b in combos:
        pair_df = pd.concat([prices[a], prices[b]], axis=1, keys=[a, b]).dropna()
        if pair_df.shape[0] < min_lookback:
            if verbose:
                print(f"skip {a}-{b}: overlap {pair_df.shape[0]} < {min_lookback}")
            continue

        y = pair_df[a]
        x = pair_df[b]

        # Cointegration
        pval, direction = engle_granger_min_pvalue(y, x)

        # Hedge ratio OLS
        try:
            beta, alpha, _ = ols_hedge_ratio(y, x)
        except Exception:
            beta, alpha = np.nan, np.nan

        # Spread & stats
        spread = y - (alpha + beta * x)
        hl = half_life(spread)
        sp_mean = float(spread.mean())
        sp_vol = float(spread.std(ddof=0))

        sc = score_pair(pval, hl, sp_vol, hl_min=hl_min, hl_max=hl_max)

        results.append(
            {
                "ticker_a": a,
                "ticker_b": b,
                "pvalue": float(pval) if not np.isnan(pval) else np.nan,
                "beta": float(beta) if not np.isnan(beta) else np.nan,
                "intercept": float(alpha) if not np.isnan(alpha) else np.nan,
                "half_life": float(hl) if not np.isnan(hl) else np.nan,
                "spread_mean": sp_mean,
                "spread_vol": sp_vol,
                "n_obs": int(pair_df.shape[0]),
                "score": float(sc),
            }
        )

    res = pd.DataFrame(results)
    if res.empty:
        return res

    # FDR Benjamini–Hochberg
    res["fdr_reject"] = benjamini_hochberg(res["pvalue"].fillna(1.0).to_list(),
                                           alpha=pvalue_threshold)

    res = res.sort_values(
        ["fdr_reject", "score", "pvalue"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return res


# =========================
# Exécution standalone (optionnel)
# =========================

if __name__ == "__main__":
    # Petit test sur données simulées
    def simulate_pair(n=1200, seed=0, beta=1.0):
        rng = np.random.default_rng(seed)
        f = np.cumsum(rng.normal(0, 0.01, size=n))
        x = 100 + f + rng.normal(0, 0.02, size=n)
        y = 50 + beta * f + rng.normal(0, 0.02, size=n)
        idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="H", tz="UTC")
        return pd.DataFrame({"X": x, "Y": y}, index=idx)

    sim = simulate_pair()
    res_demo = analyze_pairs(sim, min_lookback=500)
    print(res_demo.head())
