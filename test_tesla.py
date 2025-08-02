# Import necessary libraries
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf
import pandas as pd

# Define the IMPROVED trading strategy
class SmaCross(Strategy):
    n1 = 50  # Fast moving average
    n2 = 200 # Slow moving average

    def init(self):
        # Calculate the two moving averages
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), self.data.Close)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), self.data.Close)

    def next(self):
        # Define the stop-loss price (15% below the current price)
        stop_loss_price = self.data.Close[-1] * 0.85

        # If the fast SMA crosses above the slow SMA AND we don't have a position open
        if crossover(self.sma1, self.sma2) and not self.position:
            # Buy with a 15% stop-loss
            self.buy(sl=stop_loss_price)
            
        # If the fast SMA crosses below the slow SMA AND we have a position open
        elif crossover(self.sma2, self.sma1) and self.position:
            # Close the existing long position (don't open a short)
            self.position.close()

# Download Tesla (TSLA) data
print("Downloading TSLA data...")
data = yf.download('TSLA', start='2015-01-01', end='2025-01-01')
print("Data download complete.")

# --- Data Cleaning ---
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data.columns = [col.capitalize() for col in data.columns]

# Set up and run the backtest
bt = Backtest(data, SmaCross, cash=10000, commission=.001)
stats = bt.run()

# Print the results
print("\n--- Backtest Results (v2: Long-Only + Stop-Loss) ---")
print(stats)

# Generate an interactive plot
bt.plot()