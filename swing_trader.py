# Import necessary libraries
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf
import pandas as pd
import numpy as np

# Mean-Reversion Strategy using Bollinger Bands
class MeanReversion(Strategy):
    # Optimal parameters from our best mean-reversion run
    bb_period = 10
    bb_std_dev = 2.0
    stop_loss_pct = 0.08
    take_profit_pct = 0.04

    def init(self):
        price_series = pd.Series(self.data.Close)
        rolling_mean = price_series.rolling(window=self.bb_period).mean()
        rolling_std = price_series.rolling(window=self.bb_period).std()
        
        self.upper_band = self.I(lambda: rolling_mean + (rolling_std * self.bb_std_dev))
        self.lower_band = self.I(lambda: rolling_mean - (rolling_std * self.bb_std_dev))
        self.middle_band = self.I(lambda: rolling_mean) # Added for potential exit logic

    def next(self):
        sl_price = self.data.Close[-1] * (1 - self.stop_loss_pct)
        tp_price = self.data.Close[-1] * (1 + self.take_profit_pct)

        if crossover(self.lower_band, self.data.Close) and not self.position:
            self.buy(sl=sl_price, tp=tp_price)
            
        elif crossover(self.data.Close, self.upper_band) and self.position:
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
bt = Backtest(data, MeanReversion, cash=10000, commission=.001)
stats = bt.run()

# --- Print and Plot Results ---
print(f"\n--- Backtest Results (Mean-Reversion: {ticker}) ---")
print(stats)
bt.plot()