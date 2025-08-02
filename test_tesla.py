# Import necessary libraries
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf
import pandas as pd

# Define the trading strategy
class SmaCross(Strategy):
    n1 = 50  # Fast moving average
    n2 = 200 # Slow moving average

    def init(self):
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), self.data.Close)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), self.data.Close)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()

# Download Tesla (TSLA) data
print("Downloading TSLA data...")
data = yf.download('TSLA', start='2015-01-01', end='2025-01-01')
print("Data download complete.")

# --- NEW, MORE ROBUST FIX ---
# Flatten the column headers if they are a MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Capitalize column names for backtesting.py compatibility
data.columns = [col.capitalize() for col in data.columns]


# Set up and run the backtest
bt = Backtest(data, SmaCross, cash=10000, commission=.001)
stats = bt.run()

# Print the results
print("\n--- Backtest Results ---")
print(stats)

# Generate an interactive plot
bt.plot()