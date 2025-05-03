import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .base_strategy import TradingStrategy

class EMACrossoverStrategy(TradingStrategy):
    def __init__(self, short_window=9, long_window=21):
        self.short_window = short_window
        self.long_window = long_window

    def calculate_signals(self, data):
        """Calculate EMA and generate trading signals"""
        # Calculate EMAs
        data['EMA_short'] = data['close'].ewm(span=self.short_window, adjust=False).mean()
        data['EMA_long'] = data['close'].ewm(span=self.long_window, adjust=False).mean()
        
        # Generate signals
        data['signal'] = 0
        
        # Only calculate signals where we have both EMAs
        mask = (data['EMA_short'].notna() & data['EMA_long'].notna())
        data.loc[mask, 'signal'] = np.where(
            data.loc[mask, 'EMA_short'] > data.loc[mask, 'EMA_long'],
            1,  # Buy signal
            -1  # Sell signal
        )
        
        # Generate position changes
        data['position'] = data['signal'].diff()
        return data

    def simulate_trades(self, data, initial_balance):
        """Simulate trading based on signals"""
        balance = initial_balance
        holdings = 0
        positions = []

        for index, row in data.iterrows():
            if row['position'] == 2:  # Buy signal
                if balance > 0:
                    holdings = balance / row['close']
                    balance = 0
                    positions.append({
                        'timestamp': index,
                        'type': 'BUY',
                        'price': row['close'],
                        'amount': holdings
                    })
            
            elif row['position'] == -2:  # Sell signal
                if holdings > 0:
                    balance = holdings * row['close']
                    positions.append({
                        'timestamp': index,
                        'type': 'SELL',
                        'price': row['close'],
                        'amount': holdings
                    })
                    holdings = 0

        return positions, balance, holdings

    def plot_results(self, data, positions):
        """Plot trading results"""
        plt.figure(figsize=(15, 10))
        
        # Plot price and EMAs
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['close'], label='Price')
        plt.plot(data.index, data['EMA_short'], label=f'EMA {self.short_window}')
        plt.plot(data.index, data['EMA_long'], label=f'EMA {self.long_window}')
        
        # Plot buy/sell signals
        buy_signals = data[data['position'] == 2]
        sell_signals = data[data['position'] == -2]
        
        plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', label='Sell Signal')
        
        plt.title('Price and EMA Crossover')
        plt.legend()
        plt.grid(True)
        
        # Plot portfolio value
        plt.subplot(2, 1, 2)
        
        # Create a DataFrame to track portfolio value over time
        portfolio_df = pd.DataFrame(index=data.index)
        portfolio_df['price'] = data['close']
        portfolio_df['position'] = data['position']
        
        # Initialize portfolio tracking
        balance = 10000  # Initial balance
        holdings = 0
        portfolio_value = []
        
        # Calculate portfolio value for each time period
        for index, row in portfolio_df.iterrows():
            if row['position'] == 2:  # Buy signal
                if balance > 0:
                    holdings = balance / row['price']
                    balance = 0
            elif row['position'] == -2:  # Sell signal
                if holdings > 0:
                    balance = holdings * row['price']
                    holdings = 0
            
            # Calculate current portfolio value
            if holdings > 0:
                portfolio_value.append(holdings * row['price'])
            else:
                portfolio_value.append(balance)
        
        # Plot portfolio value
        plt.plot(portfolio_df.index, portfolio_value, label='Portfolio Value')
        plt.title('Portfolio Value Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show() 