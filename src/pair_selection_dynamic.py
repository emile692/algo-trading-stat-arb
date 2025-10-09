"""
pair_selection_dynamic.py

Sélection & scoring de paires avec hedge ratio DYNAMIQUE (β_t) via Kalman Filter.
- calcule β_t, α_t avec un modèle état-espace simple
- construit le spread_t = y_t - (α_t + β_t x_t)
- mesure la half-life sur ce spread kalmanisé
- (optionnel) mesure la stabilité de co-intégration rolling (Engle-Granger)

Dépendances : pandas, numpy, statsmodels
Optionnel      : pykalman (recommandé) -> pip install pykalman
Fallback       : RLS (Recursive Least Squares) si pykalman indisponible
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


# =========================
# Utilitaires stats
# =========================

def half_life(spread: pd.Series) -> float:
    """
    HL via AR(1) sur Δs_t = a + γ s_{t-1} + ε_t
    HL ≈ -ln(2) / γ si γ < 0, sinon inf.
    """
    s = pd.Series(spread).dropna().values
    if s.size < 20:
        return np.nan
    ds = np.diff(s)
    X = sm.add_constant(s[:-1])
    try:
        res = sm.OLS(ds, X).fit()
        gamma = float(res.params[1])
    except Exception:
        return np.nan
    if gamma >= 0:
        return np.inf
    return max(1.0, -math.log(2.0) / gamma)


def rolling_cointegration_share(y: pd.Series, x: pd.Series, window: int = 250, step: int = 10,
                                trend: str = "c", p_thresh: float = 0.05) -> float:
    """
    Pourcentage de fenêtres (de longueur 'window') où Engle–Granger est significatif.
    """
    y = y.dropna()
    x = x.dropna()
    idx = y.index.intersection(x.index)
    y = y.loc[idx]; x = x.loc[idx]
    if len(y) < window + step:
        return np.nan
    pvals = []
    for i in range(0, len(y) - window + 1, step):
        yi = y.iloc[i:i+window]
        xi = x.iloc[i:i+window]
        try:
            p = coint(yi, xi, trend=trend)[1]
            pvals.append(p)
        except Exception:
            continue
    if not pvals:
        return np.nan
    return float((np.array(pvals) < p_thresh).mean())


# =========================
# Kalman Filter (β_t, α_t)
# =========================

def kalman_beta_alpha(y: pd.Series, x: pd.Series,
                      delta: float = 1e-4,
                      obs_var: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estime β_t et α_t via Kalman. Modèle :
      state_t   = state_{t-1} + w_t,  state = [beta, alpha]
      y_t       = beta_t * x_t + alpha_t + v_t
    - delta contrôle la variabilité de l'état (plus grand = β_t/α_t bougent plus)
    - obs_var variance d'observation (échelle du bruit sur y)

    Nécessite pykalman ; fallback RLS si non installé.
    """
    y = pd.Series(y).astype(float).dropna()
    x = pd.Series(x).astype(float).reindex(y.index).ffill().bfill()


    try:
        from pykalman import KalmanFilter  # type: ignore
        # Transition : état random walk
        transition_matrices = np.eye(2)
        # Observation matrix = [x_t, 1]
        observation_matrices = np.column_stack([x.values, np.ones(len(x))])[:, np.newaxis, :]
        # Transition covariance
        trans_cov = (delta / (1.0 - delta)) * np.eye(2)
        kf = KalmanFilter(
            transition_matrices=transition_matrices,
            observation_matrices=observation_matrices,
            initial_state_mean=np.zeros(2),
            initial_state_covariance=np.eye(2),
            transition_covariance=trans_cov,
            observation_covariance=obs_var
        )
        state_means, _ = kf.filter(y.values)
        beta_t = state_means[:, 0]
        alpha_t = state_means[:, 1]
        return beta_t, alpha_t

    except Exception:
        # --------- Fallback : Recursive Least Squares (RLS) ---------
        lam = 0.995  # forgetting factor
        theta = np.zeros(2)  # [beta, alpha]
        P = np.eye(2) * 1e3
        beta_t = np.zeros(len(x))
        alpha_t = np.zeros(len(x))
        for t, (xt, yt) in enumerate(zip(x.values, y.values)):
            phi = np.array([xt, 1.0])  # [x_t, 1]
            # Kt = P phi / (lam + phi' P phi)
            denom = lam + phi @ P @ phi
            K = (P @ phi) / denom
            # theta_t = theta_{t-1} + K (y - phi' theta_{t-1})
            err = yt - phi @ theta
            theta = theta + K * err
            # P_t = (I - K phi') P / lam
            P = (np.eye(2) - np.outer(K, phi)) @ P / lam
            beta_t[t], alpha_t[t] = theta[0], theta[1]
        return beta_t, alpha_t


# =========================
# Analyse dynamique principale
# =========================

@dataclass
class DynamicPairParams:
    # Kalman / RLS
    delta: float = 1e-4
    obs_var: float = 1.0
    # Rolling cointegration (optionnel)
    roll_window: int = 250
    roll_step: int = 10
    roll_trend: str = "c"
    roll_pthresh: float = 0.05
    # Filtrage des séries
    min_lookback: int = 250


def analyze_pairs_dynamic(prices: pd.DataFrame,
                          params: Optional[DynamicPairParams] = None,
                          compute_rolling_coint: bool = True) -> pd.DataFrame:
    """
    Pour chaque paire (A,B):
      - aligne y=A, x=B
      - calcule β_t, α_t (Kalman ou RLS fallback)
      - spread_t = y - (α_t + β_t x)
      - half-life(spread_t)
      - (optionnel) rolling cointegration share sur paires non-kalmanisées
    Retourne un DataFrame trié par half-life croissante (plus rapide = mieux).
    """
    if params is None:
        params = DynamicPairParams()

    tickers = list(prices.columns)
    from itertools import combinations
    pairs = list(combinations(tickers, 2))
    out: List[dict] = []

    for a, b in pairs:
        df = pd.concat([prices[a], prices[b]], axis=1, keys=[a, b]).dropna()
        if df.shape[0] < params.min_lookback:
            continue

        y = df[a]
        x = df[b]

        beta_t, alpha_t = kalman_beta_alpha(
            y, x, delta=params.delta, obs_var=params.obs_var
        )
        spread = y.values - (alpha_t + beta_t * x.values)
        hl = half_life(pd.Series(spread, index=df.index))
        sp_vol = float(np.nanstd(spread))
        sp_mean = float(np.nanmean(spread))

        roll_share = np.nan
        if compute_rolling_coint:
            try:
                roll_share = rolling_cointegration_share(
                    y, x,
                    window=params.roll_window,
                    step=params.roll_step,
                    trend=params.roll_trend,
                    p_thresh=params.roll_pthresh
                )
            except Exception:
                roll_share = np.nan

        out.append({
            "ticker_a": a,
            "ticker_b": b,
            "kalman_delta": params.delta,
            "kalman_obs_var": params.obs_var,
            "half_life_kalman": hl,
            "spread_mean_kalman": sp_mean,
            "spread_vol_kalman": sp_vol,
            "n_obs": int(df.shape[0]),
            "rolling_coint_share": roll_share
        })

    res = pd.DataFrame(out)
    if res.empty:
        return res

    # Classement : prioriser les spreads qui reviennent vite (HL courte),
    # puis bonne stabilité rolling, puis volatilité suffisante
    res = res.sort_values(
        by=["half_life_kalman", "rolling_coint_share", "spread_vol_kalman"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    return res
