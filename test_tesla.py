# Import necessary libraries
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf
import pandas as pd

# RSI Helper Function
def rsi(array, n):
    """Calculate Relative Strength Index (RSI)"""
    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs_gain = gain.ewm(com=n-1, min_periods=n).mean()
    rs_loss = loss.abs().ewm(com=n-1, min_periods=n).mean()
    rs = rs_gain / rs_loss
    return 100 - 100 / (1 + rs)

# Define the trading strategy
class SmaCross(Strategy):
    # Use the optimal parameters from our best run
    n1 = 80
    n2 = 160
    stop_loss = 0.85 # Represents a 15% stop-loss
    rsi_period = 14
    rsi_dip_threshold = 40 # The "dip" level for RSI

    def init(self):
        # Indicators
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), self.data.Close)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), self.data.Close)
        self.rsi = self.I(rsi, self.data.Close, self.rsi_period)

    def next(self):
        sl_price = self.data.Close[-1] * self.stop_loss
        
        # --- NEW: "Buy the Dip" Entry Logic ---
        is_uptrend = self.sma1[-1] > self.sma2[-1]
        
        if is_uptrend and crossover(self.rsi, self.rsi_dip_threshold) and not self.position:
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

# --- Set up and Run the Backtest ---
bt = Backtest(data, SmaCross, cash=10000, commission=.001)
stats = bt.run()

# --- Print and Plot Results ---
print("\n--- Backtest Results (Buy the Dip with RSI) ---")
print(stats)
bt.plot()