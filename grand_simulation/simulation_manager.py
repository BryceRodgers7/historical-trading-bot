import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from regime_simulation import RegimeSimulation
from simulation_summary import SimulationSummary
import ccxt

class SimulationManager:
    def __init__(self):
        self.simulations = {}
        self.results = {}
        self.exchange = ccxt.binanceus()
        self.sim_data = {}
        self.sim_parms = {'symbol': '', 'timeframe': '', 'start_date': '', 'end_date': ''}
        
    def add_simulation(self, name, symbol, timeframe, start_date, end_date, initial_balance=10000, 
                       ema_fast_window=9, ema_slow_window=21, bb_window=20, bb_std=2, lookback_window=100):
        """
        Add a new simulation scenario
        
        Parameters:
        name (str): Unique identifier for the simulation
        symbol (str): Symbol to simulate
        timeframe (str): Timeframe to simulate
        start_date (str or datetime): Start date to simulate (format: 'YYYY-MM-DD' or datetime object)
        end_date (str or datetime): End date to simulate (format: 'YYYY-MM-DD' or datetime object)
        initial_balance (float): Starting capital
        ema_fast_window (int): Fast EMA window
        ema_slow_window (int): Slow EMA window
        bb_window (int): Bollinger Bands window
        bb_std (float): Bollinger Bands standard deviation
        lookback_window (int): Window size for lookback calculations
        """
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        self.simulations[name] = {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'initial_balance': initial_balance,
            'ema_fast_window': ema_fast_window,
            'ema_slow_window': ema_slow_window,
            'bb_window': bb_window,
            'bb_std': bb_std,
            'lookback_window': lookback_window
        }
    
    def run_all_simulations(self):
        """Run all configured simulations and store results"""
        for name, config in self.simulations.items():
            
            # do not re-fetch data if the same symbol, timeframe, start_date, and end_date
            if (self.sim_parms['symbol'] != config['symbol'] 
                or self.sim_parms['timeframe'] != config['timeframe'] 
                or self.sim_parms['start_date'] != config['start_date'] 
                or self.sim_parms['end_date'] != config['end_date']):
                
                self.sim_data = self.fetch_historical_data(config['symbol'], config['timeframe'], config['start_date'], config['end_date'])

            self.sim_parms = config

            sim = RegimeSimulation(name=name, 
                                 symbol=config['symbol'], 
                                 timeframe=config['timeframe'], 
                                 start_date=config['start_date'], 
                                 end_date=config['end_date'], 
                                 initial_balance=config['initial_balance'],
                                 ema_fast_window=config['ema_fast_window'],
                                 ema_slow_window=config['ema_slow_window'],
                                 bb_window=config['bb_window'],
                                 bb_std=config['bb_std'],
                                 lookback_window=config['lookback_window'])
            
            print(f"Running simulation: {name}")

            results = sim.run_simulation(self.sim_data)
            self.results[name] = {
                'data': results[0],
                'trades': results[1],
                'performance': results[2],
                'regime_performance': results[3]
            }
    
    def get_summary(self, summary_type='basic'):
        """
        Get summary of all simulation results
        
        Parameters:
        summary_type (str): Type of summary to return ('basic', 'returns', or 'timeframe')
        
        Returns:
        pd.DataFrame: Summary DataFrame
        """
        summary = SimulationSummary(self.results)
        
        if summary_type == 'basic':
            return summary.get_basic_summary()
        elif summary_type == 'returns':
            return summary.get_returns_summary()
        elif summary_type == 'timeframe':
            return summary.get_timeframe_comparison()
        else:
            raise ValueError(f"Unknown summary type: {summary_type}")
    
    def plot_equity_curves(self):
        """Plot equity curves for all simulations"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 8))
        for name, result in self.results.items():
            if not result['data'].empty:
                plt.plot(result['data'].index, result['data']['equity'], 
                        label=f'{name} (Return: {result["performance"].get("total_return", 0):.2f}%)')
        
        plt.title('Equity Curves Comparison')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_regime_distribution(self):
        """Plot regime distribution for all simulations"""
        import matplotlib.pyplot as plt
        
        regime_counts = {}
        for name, result in self.results.items():
            if not result['data'].empty:
                regime_counts[name] = result['data']['regime'].value_counts()
        
        df = pd.DataFrame(regime_counts).fillna(0)
        df.plot(kind='bar', figsize=(15, 8))
        plt.title('Regime Distribution Across Simulations')
        plt.xlabel('Regime')
        plt.ylabel('Count')
        plt.legend(title='Simulation')
        plt.grid(True)
        plt.show()

    def fetch_historical_data(self, symbol='BTC/USDT', timeframe='1h', start_date=None, end_date=None):
        """Fetch historical data from the exchange"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)
        if end_date is None:
            end_date = datetime.now()
            
        # Validate dates
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise ValueError("start_date and end_date must be datetime objects")
            
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
            
        # Ensure dates are timezone-aware
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        # Convert dates to timestamps (milliseconds)
        try:
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
        except (OverflowError, OSError) as e:
            raise ValueError(f"Invalid date range: {e}")
        
        # Fetch data in chunks
        all_ohlcv = []
        current_timestamp = start_timestamp
        
        while current_timestamp < end_timestamp:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_timestamp,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                current_timestamp = ohlcv[-1][0] + 1
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        if not all_ohlcv:
            raise Exception("No data fetched for the specified time period")
        
        # Convert to DataFrame
        data = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        
        # Convert index to timezone-aware datetime
        data.index = data.index.tz_localize('UTC')
        
        # Filter data to ensure we only have data within our specified window
        mask = (data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))
        data = data[mask]
        
        return data

# Example usage:
if __name__ == "__main__":
    
    # Create simulation manager
    manager = SimulationManager()
    
    # Add different market scenarios
    manager.add_simulation(
        'Bull Market 4h',
        'BTC/USDT',
        '4h',
        datetime(2023, 1, 1),
        datetime(2023, 3, 31),
        initial_balance=10000
    )

    manager.add_simulation(
        'Bull Market 4h ema12',
        'BTC/USDT',
        '4h',
        datetime(2023, 1, 1),
        datetime(2023, 3, 31),
        initial_balance=10000,
        ema_fast_window=12,
        ema_slow_window=24
    )
    
    manager.add_simulation(
        'Bull Market 1h',
        'BTC/USDT',
        '1h',
        datetime(2023, 1, 1),
        datetime(2023, 3, 31),
        initial_balance=10000
    )

    manager.add_simulation(
        'Bull Market 1h ema12',
        'BTC/USDT',
        '1h',
        datetime(2023, 1, 1),
        datetime(2023, 3, 31),
        initial_balance=10000,
        ema_fast_window=12,
        ema_slow_window=24
    )

    manager.add_simulation(
        'Bull Market 1h ema20',
        'BTC/USDT',
        '1h',
        datetime(2023, 1, 1),
        datetime(2023, 3, 31),
        initial_balance=10000,
        ema_fast_window=20,
        ema_slow_window=50
    )
    
    manager.add_simulation(
        'Bull Market 15m',
        'BTC/USDT',
        '15m',
        datetime(2023, 1, 1),
        datetime(2023, 3, 31),
        initial_balance=10000
    )

    manager.add_simulation(
        'Bull Market 15m ema12',
        'BTC/USDT',
        '15m',
        datetime(2023, 1, 1),
        datetime(2023, 3, 31),
        initial_balance=10000,
        ema_fast_window=12,
        ema_slow_window=24
    )

    manager.add_simulation(
        'Bull Market 15m ema20',
        'BTC/USDT',
        '15m',
        datetime(2023, 1, 1),
        datetime(2023, 3, 31),
        initial_balance=10000,
        ema_fast_window=20,
        ema_slow_window=50
    )
    
    manager.add_simulation(
        'Bear Market 4h',
        'BTC/USDT',
        '4h',
        datetime(2023, 4, 1),
        datetime(2023, 6, 30),
        initial_balance=10000
    )

    manager.add_simulation(
        'Bear Market 4h ema12',
        'BTC/USDT',
        '4h',
        datetime(2023, 4, 1),
        datetime(2023, 6, 30),
        initial_balance=10000,
        ema_fast_window=12,
        ema_slow_window=24
    )
    
    manager.add_simulation(
        'Bear Market 1h',
        'BTC/USDT',
        '1h',
        datetime(2023, 4, 1),
        datetime(2023, 6, 30),
        initial_balance=10000
    )

    manager.add_simulation(
        'Bear Market 1h ema12',
        'BTC/USDT',
        '1h',
        datetime(2023, 4, 1),
        datetime(2023, 6, 30),
        initial_balance=10000,
        ema_fast_window=12,
        ema_slow_window=24
    )

    manager.add_simulation(
        'Bear Market 1h ema20',
        'BTC/USDT',
        '1h',
        datetime(2023, 4, 1),
        datetime(2023, 6, 30),
        initial_balance=10000,
        ema_fast_window=20,
        ema_slow_window=50
    )
    
    manager.add_simulation(
        'Bear Market 15m',
        'BTC/USDT',
        '15m',
        datetime(2023, 4, 1),
        datetime(2023, 6, 30),
        initial_balance=10000
    )

    manager.add_simulation(
        'Bear Market 15m ema12',
        'BTC/USDT',
        '15m',
        datetime(2023, 4, 1),
        datetime(2023, 6, 30),
        initial_balance=10000,
        ema_fast_window=12,
        ema_slow_window=24
    )

    manager.add_simulation(
        'Bear Market 15m ema20',
        'BTC/USDT',
        '15m',
        datetime(2023, 4, 1),
        datetime(2023, 6, 30),
        initial_balance=10000,
        ema_fast_window=20,
        ema_slow_window=50
    )
    
    manager.add_simulation(
        'High Volatility 4h',
        'BTC/USDT',
        '4h',
        datetime(2023, 7, 1),
        datetime(2023, 9, 30),
        initial_balance=10000
    )

    manager.add_simulation(
        'High Volatility 4h ema12',
        'BTC/USDT',
        '4h',
        datetime(2023, 7, 1),
        datetime(2023, 9, 30),
        initial_balance=10000,
        ema_fast_window=12,
        ema_slow_window=24
    )
    
    manager.add_simulation(
        'High Volatility 1h',
        'BTC/USDT',
        '1h',
        datetime(2023, 7, 1),
        datetime(2023, 9, 30),
        initial_balance=10000
    )

    manager.add_simulation(
        'High Volatility 1h ema12',
        'BTC/USDT',
        '1h',
        datetime(2023, 7, 1),
        datetime(2023, 9, 30),
        initial_balance=10000,
        ema_fast_window=12,
        ema_slow_window=24
    )

    manager.add_simulation(
        'High Volatility 1h ema20',
        'BTC/USDT',
        '1h',
        datetime(2023, 7, 1),
        datetime(2023, 9, 30),
        initial_balance=10000,
        ema_fast_window=20,
        ema_slow_window=50
    )
    
    manager.add_simulation(
        'High Volatility 15m',
        'BTC/USDT',
        '15m',
        datetime(2023, 7, 1),
        datetime(2023, 9, 30),
        initial_balance=10000
    )

    manager.add_simulation(
        'High Volatility 15m ema12',
        'BTC/USDT',
        '15m',
        datetime(2023, 7, 1),
        datetime(2023, 9, 30),
        initial_balance=10000,
        ema_fast_window=12,
        ema_slow_window=24
    )

    manager.add_simulation(
        'High Volatility 15m ema20',  
        'BTC/USDT',
        '15m',
        datetime(2023, 7, 1),
        datetime(2023, 9, 30),
        initial_balance=10000,
        ema_fast_window=20,
        ema_slow_window=50
    )
    
    manager.add_simulation(
        'Low Volatility 4h',
        'BTC/USDT',
        '4h',
        datetime(2023, 10, 1),
        datetime(2023, 12, 31),
        initial_balance=10000
    )

    manager.add_simulation(
        'Low Volatility 4h ema12',
        'BTC/USDT',
        '4h',
        datetime(2023, 10, 1),
        datetime(2023, 12, 31),
        initial_balance=10000,
        ema_fast_window=12,
        ema_slow_window=24
    )
    
    manager.add_simulation(
        'Low Volatility 1h',
        'BTC/USDT',
        '1h',
        datetime(2023, 10, 1),
        datetime(2023, 12, 31),
        initial_balance=10000
    )

    manager.add_simulation(
        'Low Volatility 1h ema12',
        'BTC/USDT',
        '1h',
        datetime(2023, 10, 1),
        datetime(2023, 12, 31),
        initial_balance=10000,
        ema_fast_window=12,
        ema_slow_window=24
    )

    manager.add_simulation(
        'Low Volatility 1h ema20',
        'BTC/USDT',
        '1h',
        datetime(2023, 10, 1),
        datetime(2023, 12, 31),
        initial_balance=10000,
        ema_fast_window=20,
        ema_slow_window=50
    )
    
    manager.add_simulation(
        'Low Volatility 15m',
        'BTC/USDT',
        '15m',
        datetime(2023, 10, 1),
        datetime(2023, 12, 31),
        initial_balance=10000
    )

    manager.add_simulation(
        'Low Volatility 15m ema12',
        'BTC/USDT',
        '15m',
        datetime(2023, 10, 1),
        datetime(2023, 12, 31),
        initial_balance=10000,
        ema_fast_window=12,
        ema_slow_window=24
    )

    manager.add_simulation(
        'Low Volatility 15m ema20',
        'BTC/USDT',
        '15m',
        datetime(2023, 10, 1),
        datetime(2023, 12, 31),
        initial_balance=10000,
        ema_fast_window=20,
        ema_slow_window=50
    )
    
    # Run simulations
    manager.run_all_simulations()
    
    # Get and display results
    print("\nBasic Summary:")
    print(manager.get_summary('basic'))
    
    print("\nReturns Summary (sorted by total return):")
    print(manager.get_summary('returns'))
    
    print("\nTimeframe Comparison:")
    print(manager.get_summary('timeframe'))
    
    # Plot results
    manager.plot_equity_curves()
    manager.plot_regime_distribution() 