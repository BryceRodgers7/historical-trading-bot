import pandas as pd
import numpy as np
import ccxt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from strategies.sma_crossover import SMACrossoverStrategy

class TradingStrategy(ABC):
    """Base class for trading strategies"""
    @abstractmethod
    def calculate_signals(self, data):
        """Calculate trading signals for the given data"""
        pass

    @abstractmethod
    def simulate_trades(self, data, initial_balance):
        """Simulate trades based on the signals"""
        pass

    @abstractmethod
    def plot_results(self, data, positions):
        """Plot trading results"""
        pass

class TradingBot:
    def __init__(self, symbol='BTC/USDT', timeframe='1h', strategy=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy = strategy or SMACrossoverStrategy()
        self.exchange = ccxt.binanceus()  # Using Binance.US instead of Binance
        self.data = None
        self.positions = []
        self.initial_balance = 10000  # Starting with 10,000 USD
        self.balance = self.initial_balance
        self.holdings = 0

    def fetch_historical_data(self, start_date=None, end_date=None):
        """Fetch historical data from the exchange for a specific time window"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)  # Default to last 3 months
        if end_date is None:
            end_date = datetime.now()
        
        # Convert dates to timestamps
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        # Fetch data in chunks to handle rate limits
        all_ohlcv = []
        current_timestamp = start_timestamp
        
        while current_timestamp < end_timestamp:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=current_timestamp,
                    limit=1000  # Maximum candles per request
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                current_timestamp = ohlcv[-1][0] + 1  # Next timestamp after the last candle
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        if not all_ohlcv:
            raise Exception("No data fetched for the specified time period")
        
        self.data = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='ms')
        self.data.set_index('timestamp', inplace=True)
        
        # Filter data to ensure we only have data within our specified window
        self.data = self.data[(self.data.index >= start_date) & (self.data.index <= end_date)]
        
        return self.data

    def run_strategy(self):
        """Run the trading strategy"""
        # Calculate signals
        self.data = self.strategy.calculate_signals(self.data)
        
        # Simulate trades
        self.positions, self.balance, self.holdings = self.strategy.simulate_trades(
            self.data, self.initial_balance
        )
        
        # Calculate performance
        performance = self.calculate_performance()
        
        # Plot results
        self.strategy.plot_results(self.data, self.positions)
        
        return performance

    def calculate_performance(self):
        """Calculate trading performance metrics"""
        if not self.positions:
            return None

        # Calculate final portfolio value
        final_value = self.balance + (self.holdings * self.data['close'].iloc[-1])
        total_return = ((final_value - self.initial_balance) / self.initial_balance) * 100

        # Calculate trade statistics
        trades = pd.DataFrame(self.positions)
        if len(trades) > 0:
            # Ensure we have both buy and sell trades
            buy_trades = trades[trades['type'] == 'BUY']
            sell_trades = trades[trades['type'] == 'SELL']
            
            # Make sure we have equal number of buy and sell trades
            min_trades = min(len(buy_trades), len(sell_trades))
            if min_trades > 0:
                # Take only the first min_trades number of trades
                buy_prices = buy_trades['price'].values[:min_trades]
                sell_prices = sell_trades['price'].values[:min_trades]
                
                # Calculate winning trades (where sell price > buy price)
                winning_trades = sell_prices > buy_prices
                win_rate = (winning_trades.sum() / len(winning_trades)) * 100
            else:
                win_rate = 0
        else:
            win_rate = 0

        return {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return': total_return,
            'number_of_trades': len(self.positions) // 2,
            'win_rate': win_rate
        }

def main():
    # Initialize and run the trading bot with SMA crossover strategy
    bot = TradingBot(symbol='BTC/USDT', timeframe='1h')
    
    # Example: Fetch data for a specific 3-month window
    start_date = datetime(2024, 11, 28)
    end_date = datetime(2025, 2, 1)
    
    # Fetch historical data
    print(f"Fetching historical data from {start_date.date()} to {end_date.date()}...")
    bot.fetch_historical_data(start_date=start_date, end_date=end_date)
    
    # Run the strategy
    print("Running trading strategy...")
    performance = bot.run_strategy()
    
    # Display performance
    if performance:
        print("\nTrading Performance:")
        print(f"Initial Balance: ${performance['initial_balance']:.2f}")
        print(f"Final Value: ${performance['final_value']:.2f}")
        print(f"Total Return: {performance['total_return']:.2f}%")
        print(f"Number of Trades: {performance['number_of_trades']}")
        print(f"Win Rate: {performance['win_rate']:.2f}%")

if __name__ == "__main__":
    main() 