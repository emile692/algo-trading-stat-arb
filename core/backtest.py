# ============================================================
# core/backtest.py
# ============================================================
import numpy as np
import pandas as pd
from math import sqrt

def backtest_pair(y, x, beta_series, window=24*45, entry_z=2.0, exit_z=0.5, annualize_factor=sqrt(252*6.5)):
    """
    Backtest d'une paire avec β(t) fourni (statique ou dynamique).
    Retourne equity, pnl, sharpe, drawdown, nb_trades.
    """
    # Alignement
    y, x = y.align(x, join="inner")
    beta_series = beta_series.reindex(y.index, method="ffill")

    # Spread
    spread = y - beta_series * x

    # Rolling stats pour z-score
    mu = spread.rolling(window, min_periods=int(window*0.7)).mean().shift(1)
    sd = spread.rolling(window, min_periods=int(window*0.7)).std().shift(1)
    z = (spread - mu) / (sd + 1e-8)

    # Signaux
    enter_long = z < -entry_z
    enter_short = z > entry_z
    exit_all = z.abs() < exit_z

    pos = pd.Series(0.0, index=z.index)
    for i in range(1, len(z)):
        pos.iloc[i] = pos.iloc[i-1]
        if exit_all.iloc[i]:
            pos.iloc[i] = 0.0
        elif pos.iloc[i] == 0.0:
            if enter_long.iloc[i]:
                pos.iloc[i] = 1.0
            elif enter_short.iloc[i]:
                pos.iloc[i] = -1.0

    pos = pos.shift(1).fillna(0.0)

    pnl = (spread.diff() * pos).fillna(0.0)
    equity = pnl.cumsum()

    vol = pnl.std()
    sharpe = (annualize_factor * pnl.mean() / vol) if vol > 0 else 0.0
    dd = (equity.cummax() - equity).max()
    trades = int((pos.diff().abs() > 0).sum())

    return {
        "equity": equity,
        "pnl": pnl,
        "sharpe": sharpe,
        "max_dd": dd,
        "trades": trades,
    }
