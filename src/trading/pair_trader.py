from src.trading.execution_logger import ExecutionLogger
from datetime import datetime

logger = ExecutionLogger("WF")

# quand un trade se ferme :
logger.log_trade(
    pair="XOM-CVX",
    side_x="long", side_y="short",
    vol_x=100_000, vol_y=100_000,
    entry_time=datetime(2025,4,1,14),
    exit_time=datetime(2025,4,2,15),
    entry_px_x=104.2, entry_px_y=105.8,
    exit_px_x=106.0, exit_px_y=104.9,
    pnl_x=(106.0-104.2)*100_000/104.2,
    pnl_y=(105.8-104.9)*100_000/105.8,
    reason="take_profit"
)
