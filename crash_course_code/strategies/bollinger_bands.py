import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .base_strategy import TradingStrategy

class BollingerBandsStrategy(TradingStrategy):
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std

    def calculate_signals(self, data):
        """Calculate Bollinger Bands and generate trading signals"""
        # Calculate middle band (SMA)
        data['middle_band'] = data['close'].rolling(window=self.window).mean()
        
        # Calculate standard deviation
        data['std'] = data['close'].rolling(window=self.window).std()
        
        # Calculate upper and lower bands
        data['upper_band'] = data['middle_band'] + (data['std'] * self.num_std)
        data['lower_band'] = data['middle_band'] - (data['std'] * self.num_std)
        
        # Generate signals
        data['signal'] = 0
        
        # Buy signal when price touches lower band
        data.loc[data['close'] <= data['lower_band'], 'signal'] = 1
        
        # Sell signal when price touches upper band
        data.loc[data['close'] >= data['upper_band'], 'signal'] = -1
        
        # Generate position changes
        data['position'] = data['signal'].diff()
        return data

    def simulate_trades(self, data, initial_balance):
        """Simulate trading based on signals"""
        balance = initial_balance
        holdings = 0
        positions = []

        for index, row in data.iterrows():
            if row['position'] == 1:  # Buy signal
                if balance > 0:
                    holdings = balance / row['close']
                    balance = 0
                    positions.append({
                        'timestamp': index,
                        'type': 'BUY',
                        'price': row['close'],
                        'amount': holdings
                    })
            
            elif row['position'] == -1:  # Sell signal
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
        
        # Plot price and Bollinger Bands
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['close'], label='Price')
        plt.plot(data.index, data['middle_band'], label='Middle Band')
        plt.plot(data.index, data['upper_band'], label='Upper Band', linestyle='--')
        plt.plot(data.index, data['lower_band'], label='Lower Band', linestyle='--')
        
        # Plot buy/sell signals
        buy_signals = data[data['position'] == 1]
        sell_signals = data[data['position'] == -1]
        
        plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', label='Sell Signal')
        
        plt.title('Price and Bollinger Bands')
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
            if row['position'] == 1:  # Buy signal
                if balance > 0:
                    holdings = balance / row['price']
                    balance = 0
            elif row['position'] == -1:  # Sell signal
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