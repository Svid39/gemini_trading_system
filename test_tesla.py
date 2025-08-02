# Import necessary libraries
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf
import pandas as pd

# Define the trading strategy
class SmaCross(Strategy):
    # Parameters to optimize
    n1 = 50
    n2 = 200
    stop_loss = 0.85 # Represents a 15% stop-loss (1.0 - 0.15)

    def init(self):
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), self.data.Close)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), self.data.Close)

    def next(self):
        # Calculate the stop-loss price based on the parameter
        sl_price = self.data.Close[-1] * self.stop_loss

        if crossover(self.sma1, self.sma2) and not self.position:
            # Use the stop-loss parameter in the buy order
            self.buy(sl=sl_price)
            
        elif crossover(self.sma2, self.sma1) and self.position:
            self.position.close()

# Download Tesla (TSLA) data
print("Downloading TSLA data...")
data = yf.download('TSLA', start='2015-01-01', end='2025-01-01')
print("Data download complete.")

# --- Data Cleaning ---
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data.columns = [col.capitalize() for col in data.columns]

# --- Set up and Run Optimization ---
bt = Backtest(data, SmaCross, cash=10000, commission=.001)

print("\nRunning optimization... this will take some time.")
stats = bt.optimize(
    n1=range(10, 81, 10),
    n2=range(100, 251, 20),
    stop_loss=[0.85, 0.90, 0.95],  # Test 15%, 10%, and 5% stop-losses
    maximize='Equity Final [$]',
    constraint=lambda p: p.n1 < p.n2)

# --- Print and Plot Results ---
print("\n--- Best Backtest Results (Optimized with Stop-Loss) ---")
print(stats)
print("\n--- Optimal Parameters ---")
print(stats._strategy)
print("\nPlotting the backtest with optimal parameters...")
bt.plot()