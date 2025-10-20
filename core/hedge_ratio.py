# ============================================================
# core/hedge_ratio.py
# ============================================================
import pandas as pd
import statsmodels.api as sm

def fit_beta_static(y, x):
    """OLS sur toute la période."""
    y, x = y.align(x, join="inner")
    model = sm.OLS(y, sm.add_constant(x)).fit()
    return model.params.iloc[1]   # ✅ accès propre

def fit_beta_expanding_monthly(y, x, min_hist=200):
    """Estimation mensuelle du β sur fenêtre expanding."""
    month_ends = y.resample("ME").last().index
    beta_by_month = {}

    for t in month_ends:
        hist_y = y.loc[:t].dropna()
        hist_x = x.loc[:t].dropna()
        common = hist_y.index.intersection(hist_x.index)
        if len(common) < min_hist:
            continue
        model = sm.OLS(hist_y.loc[common], sm.add_constant(hist_x.loc[common])).fit()
        beta_by_month[t] = model.params.iloc[1]   # ✅ aussi ici

    beta_expanding = pd.Series(beta_by_month).sort_index()
    return beta_expanding.reindex(y.index, method="ffill").shift(1)
