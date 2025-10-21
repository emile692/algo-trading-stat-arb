def get_bars_per_year(freq: str) -> int:
    freq = freq.upper()
    if freq == "1H":
        return int(252 * 6.5)
    if freq == "30MIN":
        return int(252 * 13)
    if freq == "15MIN":
        return int(252 * 26)
    if freq == "4H":
        return int(252 * 1.625)
    if freq == "1D":
        return 252
    raise ValueError(f"Fréquence non gérée : {freq}")
