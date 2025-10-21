import pandas as pd
from pathlib import Path
from datetime import datetime

class ExecutionLogger:
    def __init__(self, strategy_name: str, log_dir="results/trades"):
        self.strategy = strategy_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.file = self.log_dir / f"trades_{strategy_name}.csv"
        if not self.file.exists():
            pd.DataFrame(columns=[
                "Strategy", "Pair",
                "Side_X", "Side_Y",
                "Volume_X", "Volume_Y",
                "EntryTime", "ExitTime",
                "EntryPrice_X", "EntryPrice_Y",
                "ExitPrice_X", "ExitPrice_Y",
                "PnL_X", "PnL_Y", "PnL_Total",
                "Return_%", "Duration_h", "Reason"
            ]).to_csv(self.file, index=False)

    def log_trade(self, pair, side_x, side_y,
                  vol_x, vol_y,
                  entry_time, exit_time,
                  entry_px_x, entry_px_y,
                  exit_px_x, exit_px_y,
                  pnl_x, pnl_y, reason="signal"):

        pnl_total = pnl_x + pnl_y
        ret = pnl_total / abs(vol_x + vol_y) * 100 if (vol_x + vol_y) != 0 else 0
        duration_h = (exit_time - entry_time).total_seconds() / 3600

        row = {
            "Strategy": self.strategy,
            "Pair": pair,
            "Side_X": side_x,
            "Side_Y": side_y,
            "Volume_X": vol_x,
            "Volume_Y": vol_y,
            "EntryTime": entry_time,
            "ExitTime": exit_time,
            "EntryPrice_X": entry_px_x,
            "EntryPrice_Y": entry_px_y,
            "ExitPrice_X": exit_px_x,
            "ExitPrice_Y": exit_px_y,
            "PnL_X": pnl_x,
            "PnL_Y": pnl_y,
            "PnL_Total": pnl_total,
            "Return_%": ret,
            "Duration_h": duration_h,
            "Reason": reason,
        }
        pd.DataFrame([row]).to_csv(self.file, mode="a", header=False, index=False)
