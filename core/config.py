from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class Settings:
    # data
    price_path: str                 # ex: "data/prices.csv"
    tickers: List[str]              # ex: ["GS","JPM","BK","WFC"]
    date_col: str = "Date"
    tz: Optional[str] = None        # ex: "America/New_York"
    freq_out: str = "1H"            # "1H" ou "1D"
    price_col_suffix: Optional[str] = None  # ex: "_close" si colonnes "GS_close"
    log_prices: bool = True

    # sample window
    start: Optional[str] = None     # ex: "2023-01-01"
    end: Optional[str] = None       # ex: "2025-10-01"

    # resampling
    how: str = "last"               # "last"|"mean"|"ohlc" (si tu as des barres fines)
