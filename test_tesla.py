# Final Code for Trend-Follower 
# Here is the correct code for the strategy that produced the 2,259% return. 

# Import necessary libraries
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf
import pandas as pd

# Define the Trend-Following strategy
class TrendFollower(Strategy):
    # Optimal parameters from our best trend-following run
    n1 = 80
    n2 = 160
    stop_loss = 0.85 # Represents a 15% stop-loss

    def init(self):
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), self.data.Close)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), self.data.Close)

    def next(self):
        sl_price = self.data.Close[-1] * self.stop_loss

        if crossover(self.sma1, self.sma2) and not self.position:
            self.buy(sl=sl_price)
        elif crossover(self.sma2, self.sma1) and self.position:
            self.position.close()

# --- Configuration ---
ticker = 'TSLA'
start_date = '2015-01-01'
end_date = '2025-01-01'

# Download the data
print(f"Downloading {ticker} data...")
data = yf.download(ticker, start=start_date, end=end_date)
print("Data download complete.")

# Data Cleaning
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data.columns = [col.capitalize() for col in data.columns]

# --- Set up and Run the Backtest ---
bt = Backtest(data, TrendFollower, cash=10000, commission=.001)
stats = bt.run()

# --- Print and Plot Results ---
print(f"\n--- Backtest Results (Trend-Follower: {ticker}) ---")
print(stats)
bt.plot()