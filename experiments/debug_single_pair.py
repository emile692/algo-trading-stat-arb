# detailed_reconcile.py
import pandas as pd
import numpy as np
from pathlib import Path


def detailed_reconcile(mode="Static"):
    """Réconciliation détaillée trade par trade"""

    # Charger le portefeuille final
    final_df = pd.read_csv(f"../results/final_portfolio_{mode}.csv")

    results = []

    for _, row in final_df.iterrows():
        pair = row["Pair"]
        weight = row["weight"]

        # Équity PnL
        equity_path = Path(f"../results/equities/{mode.lower()}/{pair}.csv")
        if equity_path.exists():
            equity = pd.read_csv(equity_path, index_col=0)
            equity_col = equity.select_dtypes(include=["float", "int"]).columns[0]
            equity_pnl = (equity[equity_col].iloc[-1] - equity[equity_col].iloc[0]) * weight
        else:
            equity_pnl = 0

        # Trades PnL
        trades = pd.read_csv(f"../results/trades/trades_{mode}.csv")
        pair_trades = trades[trades["Pair"] == pair]
        trades_pnl = pair_trades["PnL_Total"].sum() * weight

        # Calcul détaillé trade par trade
        detailed_trades = []
        for _, trade in pair_trades.iterrows():
            detailed_trades.append({
                "entry_time": trade["EntryTime"],
                "exit_time": trade["ExitTime"],
                "pnl": trade["PnL_Total"] * weight,
                "duration": trade["Duration_h"]
            })

        results.append({
            "Pair": pair,
            "Weight": weight,
            "EquityPnL_w€": equity_pnl,
            "TradesPnL_w€": trades_pnl,
            "Delta€": equity_pnl - trades_pnl,
            "N_trades": len(pair_trades),
            "Trades_Details": detailed_trades
        })

    df_reconcile = pd.DataFrame(results)

    print("=== Reconciliation détaillée ===")
    for _, row in df_reconcile.iterrows():
        print(f"\n--- {row['Pair']} (w={row['Weight']:.3f}) ---")
        print(f"Equity PnL: {row['EquityPnL_w€']:8.2f}€")
        print(f"Trades PnL: {row['TradesPnL_w€']:8.2f}€")
        print(f"Delta:      {row['Delta€']:8.2f}€")

        for trade in row['Trades_Details']:
            print(f"  Trade: {trade['pnl']:6.1f}€ ({trade['duration']:5.1f}h)")

    total_equity = df_reconcile["EquityPnL_w€"].sum()
    total_trades = df_reconcile["TradesPnL_w€"].sum()

    print(f"\n--- Totals ---")
    print(f"Σ EquityPnL_w€ : {total_equity:8.2f}")
    print(f"Σ TradesPnL_w€ : {total_trades:8.2f}")
    print(f"Σ Delta€       : {total_equity - total_trades:8.2f}")

    return df_reconcile


if __name__ == "__main__":
    detailed_reconcile("Static")