# Import necessary libraries
from backtesting import Backtest, Strategy
from backtesting.lib import crossover # <-- FIX: Add this import
import yfinance as yf
import pandas as pd
import numpy as np

# Mean-Reversion Strategy using Bollinger Bands
class MeanReversion(Strategy):
    # Parameters to optimize
    bb_period = 20
    bb_std_dev = 2.0
    stop_loss_pct = 0.05

    def init(self):
        price_series = pd.Series(self.data.Close)
        rolling_mean = price_series.rolling(window=self.bb_period).mean()
        rolling_std = price_series.rolling(window=self.bb_period).std()
        
        self.middle_band = self.I(lambda: rolling_mean)
        self.upper_band = self.I(lambda: rolling_mean + (rolling_std * self.bb_std_dev))
        self.lower_band = self.I(lambda: rolling_mean - (rolling_std * self.bb_std_dev))

    def next(self):
        sl_price = self.data.Close[-1] * (1 - self.stop_loss_pct)

        # Entry Condition: If price crosses below the lower band, buy
        if crossover(self.lower_band, self.data.Close) and not self.position:
            self.buy(sl=sl_price)
            
        # Exit Condition: Close if price crosses back above the middle band
        elif crossover(self.data.Close, self.middle_band) and self.position:
            self.position.close()

# Download Tesla (TSLA) data
print("Downloading TSLA data...")
data = yf.download('TSLA', start='2015-01-01', end='2025-01-01')
print("Data download complete.")

# Data Cleaning
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data.columns = [col.capitalize() for col in data.columns]

# --- Set up and Run Optimization ---
bt = Backtest(data, MeanReversion, cash=10000, commission=.001)

print("\nRunning optimization for Mean Reversion strategy with new exit...")
stats = bt.optimize(
    bb_period=[10, 20, 30, 40],
    bb_std_dev=[1.5, 2.0, 2.5],
    stop_loss_pct=list(np.arange(0.02, 0.16, 0.02)),  # 2% to 15% SL
    maximize='Sortino Ratio'
)

# Print and Plot Results
print("\n--- Backtest Results (Optimized Mean Reversion with Middle Band Exit) ---")
print(stats)
print("\n--- Optimal Parameters ---")
print(stats._strategy)
bt.plot()