# ============================================================
# core/backtest.py — spread/logs + sizing & PnL en euros (prix réels)
# ============================================================
import numpy as np
import pandas as pd
from math import sqrt
from pathlib import Path
import yaml

# === Lecture du capital global depuis params.yaml ===
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "params.yaml"
with open(CONFIG_PATH, "r") as f:
    params = yaml.safe_load(f)
INITIAL_CAPITAL = params.get("initial_capital", 10_000)


def backtest_pair(
    y: pd.Series,
    x: pd.Series,
    beta_series: pd.Series,
    window: int = 24 * 45,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    annualize_factor: float = sqrt(252 * 6.5),
):
    """
    Backtest d'une paire.
    - Signal et z-score calculés sur les LOGS (log(y), log(x)).
    - Sizing et PnL calculés en EUROS sur PRIX RÉELS.
    - Allocation par trade = INITIAL_CAPITAL réparti ~dollar-neutral via |β|.
      w_y = 1/(1+|β|) sur 'y', w_x = |β|/(1+|β|) sur 'x'.

    Retour:
      equity (euros), pnl (euros/bar), sharpe, max_dd, trades, trades_df,
      + pnl_realized, pnl_latent, pnl_total
    """

    # === Alignement & noms ===
    y, x = y.align(x, join="inner")
    y.name = getattr(y, "name", None) or "Y"
    x.name = getattr(x, "name", None) or "X"
    pair_name = f"{y.name}-{x.name}"

    beta_series = beta_series.reindex(y.index, method="ffill")

    # === Spread & Z-score sur LOGS ===
    y_log = np.log(y)
    x_log = np.log(x)
    spread = y_log - beta_series * x_log
    mu = spread.rolling(window, min_periods=int(window * 0.7)).mean().shift(1)
    sd = spread.rolling(window, min_periods=int(window * 0.7)).std().shift(1)
    z = (spread - mu) / (sd + 1e-8)

    # === Règles d'entrée / sortie ===
    enter_long = z < -entry_z
    enter_short = z > entry_z
    exit_all = z.abs() < exit_z

    pos = pd.Series(0.0, index=z.index)
    trades = []
    current_trade = None

    dy, dx = y.diff().fillna(0.0), x.diff().fillna(0.0)
    pnl_bar = pd.Series(0.0, index=y.index)

    for i in range(1, len(z)):
        t = z.index[i]
        prev_pos = pos.iloc[i - 1]
        pos.iloc[i] = prev_pos

        # ---------- EXIT ----------
        if prev_pos != 0.0 and exit_all.iloc[i]:
            pos.iloc[i] = 0.0
            if current_trade is not None:
                exit_time = t
                exit_px_y, exit_px_x = y.iloc[i], x.iloc[i]
                sign_y = +1 if current_trade["side_y"] == "long" else -1
                sign_x = +1 if current_trade["side_x"] == "long" else -1
                qty_y, qty_x = current_trade["qty_y"], current_trade["qty_x"]

                pnl_y = sign_y * qty_y * (exit_px_y - current_trade["entry_px_y"])
                pnl_x = sign_x * qty_x * (exit_px_x - current_trade["entry_px_x"])
                pnl_total = pnl_y + pnl_x

                current_trade.update({
                    "exit_time": exit_time,
                    "exit_px_y": exit_px_y,
                    "exit_px_x": exit_px_x,
                    "pnl_y": pnl_y,
                    "pnl_x": pnl_x,
                    "pnl_total": pnl_total,
                    "reason": "exit_signal",
                    "duration_h": (exit_time - current_trade["entry_time"]).total_seconds() / 3600,
                })
                trades.append(current_trade)

                seg = y.loc[current_trade["entry_time"]:exit_time].index
                pnl_bar.loc[seg] += sign_y * qty_y * dy.loc[seg] + sign_x * qty_x * dx.loc[seg]
                current_trade = None

        # ---------- ENTER LONG ----------
        elif prev_pos == 0.0 and enter_long.iloc[i]:
            pos.iloc[i] = +1.0
            entry_time = t
            entry_px_y, entry_px_x = y.iloc[i], x.iloc[i]
            beta_t = float(beta_series.iloc[i])
            beta_abs = abs(beta_t)
            w_y = 1.0 / (1.0 + beta_abs)
            w_x = beta_abs / (1.0 + beta_abs)
            capital_y = INITIAL_CAPITAL * w_y
            capital_x = INITIAL_CAPITAL * w_x
            qty_y = capital_y / max(entry_px_y, 1e-12)
            qty_x = capital_x / max(entry_px_x, 1e-12)
            current_trade = {
                "pair": pair_name,
                "side_y": "long",
                "side_x": "short",
                "entry_time": entry_time,
                "entry_px_y": entry_px_y,
                "entry_px_x": entry_px_x,
                "beta_at_entry": beta_t,
                "w_y": w_y, "w_x": w_x,
                "qty_y": qty_y, "qty_x": qty_x,
            }

        # ---------- ENTER SHORT ----------
        elif prev_pos == 0.0 and enter_short.iloc[i]:
            pos.iloc[i] = -1.0
            entry_time = t
            entry_px_y, entry_px_x = y.iloc[i], x.iloc[i]
            beta_t = float(beta_series.iloc[i])
            beta_abs = abs(beta_t)
            w_y = 1.0 / (1.0 + beta_abs)
            w_x = beta_abs / (1.0 + beta_abs)
            capital_y = INITIAL_CAPITAL * w_y
            capital_x = INITIAL_CAPITAL * w_x
            qty_y = capital_y / max(entry_px_y, 1e-12)
            qty_x = capital_x / max(entry_px_x, 1e-12)
            current_trade = {
                "pair": pair_name,
                "side_y": "short",
                "side_x": "long",
                "entry_time": entry_time,
                "entry_px_y": entry_px_y,
                "entry_px_x": entry_px_x,
                "beta_at_entry": beta_t,
                "w_y": w_y, "w_x": w_x,
                "qty_y": qty_y, "qty_x": qty_x,
            }

    # ---------- FIN DU BACKTEST ----------
    # Option A : on laisse la position OUVERTE (non close)
    # -> pas d'append dans trades, on calcule juste le latent ci-dessous.

    # === Equity & métriques ===
    equity = pnl_bar.cumsum()
    vol = pnl_bar.std()
    sharpe = (annualize_factor * pnl_bar.mean() / (vol + 1e-12)) if vol > 0 else 0.0
    dd = (equity.cummax() - equity).max()
    trades_df = pd.DataFrame(trades)

    # === PnL latent à la fin ===
    latent_pnl = 0.0
    if current_trade is not None:
        exit_px_y, exit_px_x = y.iloc[-1], x.iloc[-1]
        sign_y = +1 if current_trade["side_y"] == "long" else -1
        sign_x = +1 if current_trade["side_x"] == "long" else -1
        qty_y, qty_x = current_trade["qty_y"], current_trade["qty_x"]
        latent_pnl_y = sign_y * qty_y * (exit_px_y - current_trade["entry_px_y"])
        latent_pnl_x = sign_x * qty_x * (exit_px_x - current_trade["entry_px_x"])
        latent_pnl = latent_pnl_y + latent_pnl_x

    realized_pnl = equity.iloc[-1]
    total_pnl = realized_pnl + latent_pnl

    return {
        "equity": equity,
        "pnl": pnl_bar,
        "sharpe": sharpe,
        "max_dd": dd,
        "trades": len(trades_df),
        "trades_df": trades_df,
        "pnl_realized": realized_pnl,
        "pnl_latent": latent_pnl,
        "pnl_total": total_pnl,
    }
