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
        # Indicators for the primary asset (TSLA)
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), self.data.Close)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), self.data.Close)
        
        # --- NEW: Market filter indicators ---
        # These are passed in with the data
        self.spy_close = self.I(lambda: self.data.Spy_close, name="SPY_Close")
        self.spy_sma = self.I(lambda: self.data.Spy_sma, name="SPY_SMA")

    def next(self):
        # Define the stop-loss price
        sl_price = self.data.Close[-1] * self.stop_loss
        
        # --- NEW: Market filter condition ---
        is_market_healthy = self.spy_close[-1] > self.spy_sma[-1]

        # Check for buy signal and if the market is healthy
        if is_market_healthy and crossover(self.sma1, self.sma2) and not self.position:
            self.buy(sl=sl_price)
            
        elif crossover(self.sma2, self.sma1) and self.position:
            self.position.close()

# --- Step 1: Download Data for both TSLA and the Market (SPY) ---
print("Downloading TSLA and SPY data...")
tsla_data = yf.download('TSLA', start='2015-01-01', end='2025-01-01')
spy_data = yf.download('SPY', start='2015-01-01', end='2025-01-01')
print("Data download complete.")

# --- Step 2: Prepare the Data and Add the Filter ---
# Clean TSLA data
if isinstance(tsla_data.columns, pd.MultiIndex):
    tsla_data.columns = tsla_data.columns.get_level_values(0)
tsla_data.columns = [col.capitalize() for col in tsla_data.columns]

# Prepare SPY filter data
spy_filter = pd.DataFrame()
spy_filter['Spy_close'] = spy_data['Close']
spy_filter['Spy_sma'] = spy_data['Close'].rolling(200).mean()

# Merge the TSLA data with the SPY filter data
# This adds the 'Spy_close' and 'Spy_sma' columns to our main dataframe
data = pd.concat([tsla_data, spy_filter], axis=1)
data.dropna(inplace=True) # Remove rows with missing data (e.g., the first 200 days for the SPY SMA)

# --- Step 3: Run the Optimization ---
bt = Backtest(data, SmaCross, cash=10000, commission=.001)

print("\nRunning optimization with market filter... this may take some time.")
stats = bt.optimize(
    n1=range(10, 81, 10),
    n2=range(100, 251, 20),
    stop_loss=[0.85, 0.90, 0.95],  # Test 15%, 10%, and 5% stop-losses
    maximize='Max. Drawdown [%]', # Let's try to minimize drawdown this time
    constraint=lambda p: p.n1 < p.n2)

# --- Print and Plot Results ---
print("\n--- Best Backtest Results (Optimized with Market Filter) ---")
print(stats)
print("\n--- Optimal Parameters ---")
print(stats._strategy)
print("\nPlotting the backtest with optimal parameters...")
bt.plot()