from experiments.portfolio_analysis import load_equity, to_returns

eq = load_equity("AAPL-MSFT", "static")
print(eq.head())
r = to_returns(eq)
print(r.describe())
